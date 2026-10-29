// Fused sliding-window cepstral mean (and optional variance) normalization
// kernel for torchaudio (TA-09).
// Mirrors functional.sliding_window_cmn exactly (window boundary logic + sums),
// but in a single C++ kernel to eliminate the Python per-frame dispatch overhead.
// Registered under an INDEPENDENT op name `_fused_sliding_window_cmn` so the original
// functional implementation remains the reference (single-process A/B compare).
//
// Signature: caller provides output tensor (Tensor(a!), float, same shape).
#include <libtorchaudio/utils.h>
#include <torch/csrc/stable/library.h>
#include <torch/csrc/stable/ops.h>
#include <torch/csrc/stable/tensor.h>
#include <torch/csrc/inductor/aoti_torch/generated/c_shim_cpu.h>
#include <torch/headeronly/core/Dispatch_v2.h>
#include "fused_cmn.h"

#include <vector>
#include <cmath>

namespace {

using torch::stable::Tensor;

inline Tensor stable_mul(const Tensor& a, const Tensor& b) {
  const auto num_args = 2;
  std::array<StableIValue, num_args> stack{
      torch::stable::detail::from(a),
      torch::stable::detail::from(b)};
  TORCH_ERROR_CODE_CHECK(torch_call_dispatcher(
      "aten::mul", "Tensor", stack.data(), TORCH_ABI_VERSION));
  return torch::stable::detail::to<torch::stable::Tensor>(stack[0]);
}

inline Tensor stable_cumsum(const Tensor& self, int64_t dim) {
  const auto num_args = 3;
  std::array<StableIValue, num_args> stack{
      torch::stable::detail::from(self),
      torch::stable::detail::from(dim),
      torch::stable::detail::from(std::nullopt)};
  TORCH_ERROR_CODE_CHECK(torch_call_dispatcher(
      "aten::cumsum", "", stack.data(), TORCH_ABI_VERSION));
  return torch::stable::detail::to<torch::stable::Tensor>(stack[0]);
}

template <typename scalar_t>
void host_cmn(
    const torch::stable::Tensor& specgram,
    const scalar_t* in,
    scalar_t* out,
    int64_t channels,
    int64_t num_frames,
    int64_t num_feats,
    int64_t cmn_window,
    int64_t min_cmn_window,
    bool center,
    bool norm_vars) {
  // per-channel running sums
  std::vector<scalar_t> cur_sum(channels * num_feats, scalar_t(0));
  std::vector<scalar_t> cur_sumsq(channels * num_feats, scalar_t(0));

  auto sum_at = [&](int64_t c) { return cur_sum.data() + c * num_feats; };
  auto sq_at = [&](int64_t c) { return cur_sumsq.data() + c * num_feats; };

  int64_t last_window_start = -1;
  int64_t last_window_end = -1;

  for (int64_t t = 0; t < num_frames; t++) {
    int64_t window_start = 0, window_end = 0;
    if (center) {
      window_start = t - cmn_window / 2;
      window_end = window_start + cmn_window;
    } else {
      window_start = t - cmn_window;
      window_end = t + 1;
    }
    if (window_start < 0) {
      window_end -= window_start;
      window_start = 0;
    }
    if (!center) {
      if (window_end > t) {
        window_end = (t + 1 > min_cmn_window) ? (t + 1) : min_cmn_window;
      }
    }
    if (window_end > num_frames) {
      window_start -= window_end - num_frames;
      window_end = num_frames;
      if (window_start < 0) {
        window_start = 0;
      }
    }

    if (last_window_start == -1) {
      // Initial window: match torch's EXACT blocked reduction so the running
      // sum is bit-identical to functional.sliding_window_cmn.
      //   Python: input_part = specgram[:, window_start : window_end - window_start, :]
      //           cur_sum    += torch.sum(input_part, 1)
      int64_t slice_stop = window_end - window_start;
      int64_t rank = (int64_t)specgram.sizes().size();
      int64_t frames_dim = rank - 2;
      {
        torch::stable::Tensor part = torch::stable::narrow(
            const_cast<torch::stable::Tensor&>(specgram),
            frames_dim, window_start, slice_stop - window_start);
        torch::stable::Tensor csum = torch::stable::contiguous(
            torch::stable::sum(
                part,
                torch::headeronly::IntHeaderOnlyArrayRef({frames_dim}),
                false,
                std::nullopt));
        const scalar_t* csum_data = csum.const_data_ptr<scalar_t>();
        for (int64_t c = 0; c < channels; c++) {
          scalar_t* cs = sum_at(c);
          for (int64_t j = 0; j < num_feats; j++)
            cs[j] += csum_data[c * num_feats + j];
        }
        if (norm_vars) {
          // Python: cur_sumsq += torch.cumsum(input_part**2, 1)[:, -1, :]
          torch::stable::Tensor sqr = stable_mul(part, part);
          torch::stable::Tensor ccs = stable_cumsum(sqr, frames_dim);
          int64_t L = slice_stop - window_start;
          torch::stable::Tensor last =
              torch::stable::contiguous(torch::stable::select(ccs, frames_dim, L - 1));
          const scalar_t* cc_data = last.const_data_ptr<scalar_t>();
          for (int64_t c = 0; c < channels; c++) {
            scalar_t* sq = sq_at(c);
            for (int64_t j = 0; j < num_feats; j++)
              sq[j] += cc_data[c * num_feats + j];
          }
        }
      }
    } else {
      if (window_start > last_window_start) {
        for (int64_t c = 0; c < channels; c++) {
          scalar_t* cs = sum_at(c);
          scalar_t* sq = sq_at(c);
          const scalar_t* rm = in + (c * num_frames + last_window_start) * num_feats;
          for (int64_t j = 0; j < num_feats; j++) {
            cs[j] -= rm[j];
            if (norm_vars)
              sq[j] -= rm[j] * rm[j];
          }
        }
      }
      if (window_end > last_window_end) {
        for (int64_t c = 0; c < channels; c++) {
          scalar_t* cs = sum_at(c);
          scalar_t* sq = sq_at(c);
          const scalar_t* add = in + (c * num_frames + last_window_end) * num_feats;
          for (int64_t j = 0; j < num_feats; j++) {
            cs[j] += add[j];
            if (norm_vars)
              sq[j] += add[j] * add[j];
          }
        }
      }
    }

    int64_t window_frames = window_end - window_start;
    last_window_start = window_start;
    last_window_end = window_end;

    if (norm_vars && window_frames != 1) {
      // ROUND-2b: compute variance + inverse-std inline in scalar C++ instead
      // of re-entering the torch dispatcher (from_blob + pow + contiguous) on
      // every frame. Same math as functional:
      //   variance = cur_sumsq/w - (cur_sum/w)^2 ;  inv = pow(variance, -0.5)
      //   cmn = (spec - cur_sum/w) * inv
      const scalar_t win = static_cast<scalar_t>(window_frames);
      const scalar_t win2 = win * win;
      for (int64_t c = 0; c < channels; c++) {
        scalar_t* cs = sum_at(c);
        scalar_t* sq = sq_at(c);
        scalar_t* outc = out + (c * num_frames + t) * num_feats;
        const scalar_t* spec_row = in + (c * num_frames + t) * num_feats;
        for (int64_t j = 0; j < num_feats; j++) {
          scalar_t var = sq[j] / win - (cs[j] * cs[j]) / win2;
          scalar_t inv = std::pow(var, static_cast<scalar_t>(-0.5));
          scalar_t v = spec_row[j] - cs[j] / win;
          outc[j] = v * inv;
        }
      }
    } else {
      for (int64_t c = 0; c < channels; c++) {
        scalar_t* cs = sum_at(c);
        scalar_t* sq = sq_at(c);
        scalar_t* outc = out + (c * num_frames + t) * num_feats;
        const scalar_t* spec_row = in + (c * num_frames + t) * num_feats;
        for (int64_t j = 0; j < num_feats; j++) {
          scalar_t v = spec_row[j] - cs[j] / static_cast<scalar_t>(window_frames);
          if (norm_vars) {
            outc[j] = scalar_t(0); // window_frames == 1
          } else {
            outc[j] = v;
          }
        }
      }
    }
  }
}

} // namespace

Tensor fused_cmn_cpu(
    Tensor specgram,
    int64_t cmn_window,
    int64_t min_cmn_window,
    bool center,
    bool norm_vars,
    Tensor out) {
  int64_t rank = (int64_t)specgram.sizes().size();
  int64_t num_frames = specgram.size(rank - 2);
  int64_t num_feats = specgram.size(rank - 1);
  int64_t n_channels = 1;
  for (int64_t i = 0; i + 2 < rank; i++)
    n_channels *= specgram.size(i);

  THO_DISPATCH_V2(
      specgram.scalar_type(),
      "fused_cmn",
      [&] {
        const scalar_t* in = specgram.const_data_ptr<scalar_t>();
        scalar_t* o = out.mutable_data_ptr<scalar_t>();
        host_cmn<scalar_t>(
            specgram, in, o, n_channels, num_frames, num_feats,
            cmn_window, min_cmn_window, center, norm_vars);
      },
      AT_FLOATING_TYPES);
  return out;
}

Tensor fused_cmn_generic(
    Tensor specgram,
    int64_t cmn_window,
    int64_t min_cmn_window,
    bool center,
    bool norm_vars,
    Tensor out) {
  return fused_cmn_cpu(specgram, cmn_window, min_cmn_window, center, norm_vars, out);
}

STABLE_TORCH_LIBRARY_FRAGMENT(torchaudio, m) {
  m.def(
      "_fused_sliding_window_cmn("
      "Tensor specgram,"
      "int cmn_window,"
      "int min_cmn_window,"
      "bool center,"
      "bool norm_vars,"
      "Tensor(a!) out) -> Tensor(a!)");
}

STABLE_TORCH_LIBRARY_IMPL(torchaudio, CPU, m) {
  m.impl("_fused_sliding_window_cmn", TORCH_BOX(&fused_cmn_cpu));
}

STABLE_TORCH_LIBRARY_IMPL(torchaudio, CompositeExplicitAutograd, m) {
  m.impl("_fused_sliding_window_cmn", TORCH_BOX(&fused_cmn_generic));
}
