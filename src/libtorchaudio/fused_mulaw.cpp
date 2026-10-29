// Fused mu-law encoding kernel for torchaudio (TA-07).
// Single-pass elementwise fusion of the mu-law encoder, mirroring
//   functional.mu_law_encoding : sign(x)*log1p(mu*|x|)/log1p(mu) -> int64
// Registered under an INDEPENDENT op name `_fused_mulaw` so the original
// functional implementation remains the reference (single-process A/B compare).
//
// ROUND-2 FIX: was a single-threaded scalar loop (0.02x on large N vs the
// parallel/vectorized torch reference). Now parallelized via parallel_for,
// keeping the EXACT per-element float32 computation (so the int64 output and
// the flash-attention correctness verdict are unchanged). A small-N serial
// branch avoids thread-launch overhead on tiny inputs.
#include <libtorchaudio/utils.h>
#include <torch/csrc/stable/library.h>
#include <torch/csrc/stable/ops.h>
#include <torch/csrc/stable/tensor.h>
#include <torch/headeronly/core/Dispatch_v2.h>
#include "fused_mulaw.h"

#include <cmath>

namespace {

using torch::stable::Tensor;

template <typename scalar_t>
void host_mulaw_encode(
    const scalar_t* in,
    int64_t* out,
    int64_t n,
    scalar_t mu,
    scalar_t log_mu) {
  if (n <= 65536) {
    // Small N: serial (avoid thread-launch overhead).
    for (int64_t i = 0; i < n; i++) {
      scalar_t v = in[i];
      scalar_t sign =
          (v < 0) ? static_cast<scalar_t>(-1.0) : static_cast<scalar_t>(1.0);
      scalar_t y = sign * std::log1p(mu * std::fabs(v)) / log_mu;
      out[i] = static_cast<int64_t>(((y + 1) / 2) * mu +
                                    static_cast<scalar_t>(0.5));
    }
    return;
  }
  // Large N: parallelize over flat n; per-element math identical to serial.
  torch::stable::parallel_for(
      0, n, 4096,
      [&](int64_t begin, int64_t end) {
        for (int64_t i = begin; i < end; i++) {
          scalar_t v = in[i];
          scalar_t sign =
              (v < 0) ? static_cast<scalar_t>(-1.0) : static_cast<scalar_t>(1.0);
          scalar_t y = sign * std::log1p(mu * std::fabs(v)) / log_mu;
          out[i] = static_cast<int64_t>(((y + 1) / 2) * mu +
                                        static_cast<scalar_t>(0.5));
        }
      });
}

} // namespace

Tensor fused_mulaw_cpu(Tensor x, int64_t quantization_channels, Tensor out) {
  // total element count across all dims
  int64_t n = 1;
  { auto sz = x.sizes();
    for (int64_t i = 0; i < (int64_t)sz.size(); i++) n *= sz[i]; }
  int64_t* o = out.mutable_data_ptr<int64_t>();
  THO_DISPATCH_V2(
      x.scalar_type(),
      "fused_mulaw",
      [&] {
        const scalar_t* in = x.const_data_ptr<scalar_t>();
        scalar_t mu = static_cast<scalar_t>(quantization_channels - 1);
        scalar_t log_mu = std::log1p(mu);
        host_mulaw_encode<scalar_t>(in, o, n, mu, log_mu);
      },
      AT_FLOATING_TYPES);
  return out;
}

Tensor fused_mulaw_generic(Tensor x, int64_t quantization_channels, Tensor out) {
  return fused_mulaw_cpu(x, quantization_channels, out);
}

STABLE_TORCH_LIBRARY_FRAGMENT(torchaudio, m) {
  m.def(
      "_fused_mulaw("
      "Tensor x,"
      "int quantization_channels,"
      "Tensor(a!) out) -> Tensor(a!)");
}

STABLE_TORCH_LIBRARY_IMPL(torchaudio, CPU, m) {
  m.impl("_fused_mulaw", TORCH_BOX(&fused_mulaw_cpu));
}

STABLE_TORCH_LIBRARY_IMPL(torchaudio, CompositeExplicitAutograd, m) {
  m.impl("_fused_mulaw", TORCH_BOX(&fused_mulaw_generic));
}
