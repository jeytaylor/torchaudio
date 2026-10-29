// Fused amplitude<->decibel conversion kernels for torchaudio (TA-08).
// Mirrors functional.amplitude_to_DB and DB_to_amplitude bit-for-bit on the
// elementwise ops; the top_db path does a per-batch max-clamp reduction.
// Registered under INDEPENDENT op names `_fused_amp_db` / `_fused_db2amp` so the
// original functional implementations remain the reference (single-process A/B).
#include <libtorchaudio/utils.h>
#include <torch/csrc/stable/library.h>
#include <torch/csrc/stable/ops.h>
#include <torch/csrc/stable/tensor.h>
#include <torch/headeronly/core/Dispatch_v2.h>
#include "fused_amp_db.h"

#include <cmath>
#include <limits>
#include <vector>

namespace {

using torch::stable::Tensor;

// amplitude_to_DB elementwise (no top_db), parallel over flat n
template <typename scalar_t>
void amp_to_db_flat(
    const scalar_t* in,
    scalar_t* out,
    int64_t n,
    scalar_t multiplier,
    scalar_t amin,
    scalar_t db_multiplier) {
  if (n <= 65536) {
    // Small N: serial (avoid thread-launch overhead on tiny inputs).
    for (int64_t i = 0; i < n; i++) {
      scalar_t x = in[i];
      scalar_t c = x < amin ? amin : x;
      out[i] = multiplier * std::log10(c) - multiplier * db_multiplier;
    }
    return;
  }
  torch::stable::parallel_for(
      0, n, 4096,
      [&](int64_t begin, int64_t end) {
        for (int64_t i = begin; i < end; i++) {
          scalar_t x = in[i];
          scalar_t c = x < amin ? amin : x;
          out[i] = multiplier * std::log10(c) - multiplier * db_multiplier;
        }
      });
}

// Top-db path. Layout already flattened to batch*packed*freq*time, contiguous.
// amax over (packed,freq,time) per batch element.
template <typename scalar_t>
void amp_to_db_topdb(
    const scalar_t* in,
    scalar_t* out,
    int64_t batch,
    int64_t packed,
    int64_t freq,
    int64_t time,
    scalar_t multiplier,
    scalar_t amin,
    scalar_t db_multiplier,
    scalar_t top_db) {
  int64_t inner = packed * freq * time;
  std::vector<scalar_t> xdb(batch * inner);

  amp_to_db_flat(in, xdb.data(), batch * inner, multiplier, amin, db_multiplier);

  torch::stable::parallel_for(
      0, batch, 4,
      [&](int64_t begin, int64_t end) {
        for (int64_t b = begin; b < end; b++) {
          const scalar_t* base = &xdb[b * inner];
          scalar_t* obase = &out[b * inner];
          // SEMANTIC FIX: torch.amax propagates NaN -> if any NaN in this
          // batch, the whole batch becomes NaN (torch.max(x, NaN)=NaN).
          // Our scalar >-reduction would silently ignore NaN, diverging from
          // the serial path. Detect NaN and propagate for semantic parity.
          bool has_nan = false;
          for (int64_t j = 0; j < inner; j++) {
            if (std::isnan(base[j])) { has_nan = true; break; }
          }
          if (has_nan) {
            for (int64_t j = 0; j < inner; j++)
              obase[j] = std::numeric_limits<scalar_t>::quiet_NaN();
            continue;
          }
          scalar_t mx = base[0];
          for (int64_t j = 1; j < inner; j++)
            if (base[j] > mx) mx = base[j];
          scalar_t cutoff = mx - top_db;
          for (int64_t j = 0; j < inner; j++)
            obase[j] = base[j] > cutoff ? base[j] : cutoff;
        }
      });
}

// DB_to_amplitude: ref * pow(pow(10, 0.1*x), power)  (all in float32 like torch)
// ROUND-2: collapse the two pow() into one fast exp2-based path
//   10^(0.1*x*power) = exp2(0.1*x*power*log2(10))
// plus a small-N serial branch.
template <typename scalar_t>
void db_to_amp_flat(
    const scalar_t* in,
    scalar_t* out,
    int64_t n,
    scalar_t ref,
    scalar_t power) {
  const scalar_t k = (scalar_t)(0.1 * std::log2(10.0) * (double)power);
  if (n <= 65536) {
    for (int64_t i = 0; i < n; i++) {
      scalar_t x = in[i];
      out[i] = ref * std::exp2(x * k);
    }
    return;
  }
  torch::stable::parallel_for(
      0, n, 4096,
      [&](int64_t begin, int64_t end) {
        for (int64_t i = begin; i < end; i++) {
          scalar_t x = in[i];
          out[i] = ref * std::exp2(x * k);
        }
      });
}

}  // namespace

Tensor fused_amp_db_cpu(
    Tensor x,
    double multiplier,
    double amin,
    double db_multiplier,
    double top_db,       // NAN => no top_db
    Tensor out) {
  THO_DISPATCH_V2(
      x.scalar_type(),
      "fused_amp_db",
      [&] {
        const scalar_t* in = x.const_data_ptr<scalar_t>();
        scalar_t* o = out.mutable_data_ptr<scalar_t>();
        int64_t numel = 1;
        for (auto s : x.sizes()) numel *= s;
        if (std::isnan(top_db)) {
          amp_to_db_flat<scalar_t>(
              in, o, numel, (scalar_t)multiplier, (scalar_t)amin,
              (scalar_t)db_multiplier);
        } else {
          // replicate batch reshape: intended for 4D/3D/2D spectrogram
          int64_t rank = (int64_t)x.sizes().size();
          int64_t s1 = rank >= 3 ? x.size(rank - 3) : 1;  // packed_channels
          int64_t freq = x.size(rank - 2);
          int64_t time = x.size(rank - 1);
          int64_t packed = rank >= 3 ? s1 : 1;
          int64_t batch = numel / (packed * freq * time);
          amp_to_db_topdb<scalar_t>(
              in, o, batch, packed, freq, time, (scalar_t)multiplier,
              (scalar_t)amin, (scalar_t)db_multiplier, (scalar_t)top_db);
        }
      },
      AT_FLOATING_TYPES);
  return out;
}

Tensor fused_amp_db_generic(
    Tensor x,
    double multiplier,
    double amin,
    double db_multiplier,
    double top_db,
    Tensor out) {
  return fused_amp_db_cpu(x, multiplier, amin, db_multiplier, top_db, out);
}

Tensor fused_db2amp_cpu(
    Tensor x,
    double ref,
    double power,
    Tensor out) {
  THO_DISPATCH_V2(
      x.scalar_type(),
      "fused_db2amp",
      [&] {
        const scalar_t* in = x.const_data_ptr<scalar_t>();
        scalar_t* o = out.mutable_data_ptr<scalar_t>();
        int64_t numel = 1;
        for (auto s : x.sizes()) numel *= s;
        db_to_amp_flat<scalar_t>(in, o, numel, (scalar_t)ref, (scalar_t)power);
      },
      AT_FLOATING_TYPES);
  return out;
}

Tensor fused_db2amp_generic(
    Tensor x,
    double ref,
    double power,
    Tensor out) {
  return fused_db2amp_cpu(x, ref, power, out);
}

STABLE_TORCH_LIBRARY_FRAGMENT(torchaudio, m) {
  m.def(
      "_fused_amp_db("
      "Tensor x,"
      "float multiplier,"
      "float amin,"
      "float db_multiplier,"
      "float top_db,"
      "Tensor(a!) out) -> Tensor(a!)");
  m.def(
      "_fused_db2amp("
      "Tensor x,"
      "float ref,"
      "float power,"
      "Tensor(a!) out) -> Tensor(a!)");
}

STABLE_TORCH_LIBRARY_IMPL(torchaudio, CPU, m) {
  m.impl("_fused_amp_db", TORCH_BOX(&fused_amp_db_cpu));
  m.impl("_fused_db2amp", TORCH_BOX(&fused_db2amp_cpu));
}

STABLE_TORCH_LIBRARY_IMPL(torchaudio, CompositeExplicitAutograd, m) {
  m.impl("_fused_amp_db", TORCH_BOX(&fused_amp_db_generic));
  m.impl("_fused_db2amp", TORCH_BOX(&fused_db2amp_generic));
}
