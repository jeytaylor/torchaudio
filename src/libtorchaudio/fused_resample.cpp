// Fused polyphase sinc resampler kernel for torchaudio (TA-06).
// Fuses functional._apply_sinc_resample_kernel (the per-call multi-op chain):
//   pad(width, width+orig_freq) + conv1d(kernel, stride=orig_freq)
//   + transpose + reshape + [.., :target_length]
// into a single parallel polyphase-FIR pass over the output samples.
//
// Kernel precomputation (_get_sinc_resample_kernel) is NOT hot (transforms
// cache it), so it stays in Python; this op consumes the generated kernel.
//
// Independent op name `_fused_resample`; reference remains the original.
// Signature: out tensor (Tensor(a!), float) with leading shape + (tgt_len).
#include <libtorchaudio/utils.h>
#include <torch/csrc/stable/library.h>
#include <torch/csrc/stable/ops.h>
#include <torch/csrc/stable/tensor.h>
#include <torch/headeronly/core/Dispatch_v2.h>
#include "fused_resample.h"

#include <cmath>
#include <vector>
#include <type_traits>
#if defined(__aarch64__)
#pragma GCC push_options
#pragma GCC target("arch=armv9-a+sve2")
#include <arm_sve.h>
#endif

namespace {

using torch::stable::Tensor;

// in        : packed waveform (num_wavs, orig_len)
// kernel    : (new_freq, 1, K) filterbank rows (float), K = width + orig_freq
// out       : (num_wavs, tgt_len)
// All freq/width already gcd-reduced by the caller (matches reference).
template <typename scalar_t>
void host_resample(
    const scalar_t* in,
    const scalar_t* kern,
    scalar_t* out,
    int64_t num_wavs,
    int64_t orig_len,
    int64_t new_freq,
    int64_t orig_freq,
    int64_t width,
    int64_t K,
    int64_t tgt_len) {
  int64_t L = orig_len + 2 * width + orig_freq;  // padded input length
  std::vector<scalar_t> pad(num_wavs * L, scalar_t(0));
  for (int64_t n = 0; n < num_wavs; n++) {
    const scalar_t* src = in + n * orig_len;
    scalar_t* dst = &pad[n * L];
    for (int64_t i = 0; i < orig_len; i++)
      dst[width + i] = src[i];
  }

  // O = number of conv output positions = floor((L - K)/orig_freq) + 1
  int64_t O = (L - K) / orig_freq + 1;

  // ROUND-2b: parallelize over conv positions (O) instead of flat outputs.
  // Removes per-output div/mod (idx/total_out, m/new_freq, m%new_freq), reuses
  // the input row across all new_freq output phases (cache), and writes the
  // contiguous output block [p*new_freq, min(p*new_freq+new_freq,tgt_len)).
  torch::stable::parallel_for(
      0, num_wavs * O, 16,
      [&](int64_t begin, int64_t end) {
        for (int64_t idx = begin; idx < end; idx++) {
          int64_t n = idx / O;
          int64_t p = idx % O;
          const scalar_t* row = &pad[n * L + p * orig_freq];
          scalar_t* obase = out + n * tgt_len;
          int64_t o0 = p * new_freq;
          int64_t o1 = o0 + new_freq;
          if (o1 > tgt_len) o1 = tgt_len;
          for (int64_t o = o0; o < o1; o++) {
            const scalar_t* f = kern + (o - o0) * K;  // filter row = o % new_freq
            scalar_t s = scalar_t(0);
#if defined(__ARM_FEATURE_SVE)
            if constexpr (std::is_same_v<scalar_t, float>) {
                svfloat32_t acc = svdup_f32(0.0f);
                int64_t k = 0;
                for (; k + 8 <= K; k += 8) {
                    svfloat32_t r  = svld1_f32(svptrue_b32(), row + k);
                    svfloat32_t fi = svld1_f32(svptrue_b32(), f + k);
                    acc = svmad_f32(acc, r, fi);
                }
                s = svaddv_f32(svptrue_b32(), acc);
                for (; k < K; k++) s += row[k] * f[k];
            } else
#endif
            {
                for (int64_t k = 0; k < K; k++)
                    s += row[k] * f[k];
            }
            obase[o] = s;
          }
        }
      });
}

}  // namespace

Tensor fused_resample_cpu(
    Tensor waveform,
    Tensor kernel,
    int64_t orig_freq,   // gcd-reduced
    int64_t new_freq,    // gcd-reduced
    int64_t width,
    Tensor out) {
  int64_t rank = (int64_t)waveform.sizes().size();
  int64_t orig_len = waveform.size(rank - 1);
  int64_t num_wavs = 1;
  for (int64_t i = 0; i + 1 < rank; i++)
    num_wavs *= waveform.size(i);

  int64_t K = kernel.size(-1);
  int64_t tgt_len =
      (int64_t)std::ceil((double)new_freq * (double)orig_len / (double)orig_freq);

  THO_DISPATCH_V2(
      waveform.scalar_type(),
      "fused_resample",
      [&] {
        const scalar_t* in = waveform.const_data_ptr<scalar_t>();
        const scalar_t* kern = kernel.const_data_ptr<scalar_t>();
        scalar_t* o = out.mutable_data_ptr<scalar_t>();
        host_resample<scalar_t>(
            in, kern, o, num_wavs, orig_len, new_freq, orig_freq,
            width, K, tgt_len);
      },
      AT_FLOATING_TYPES);
  return out;
}

Tensor fused_resample_generic(
    Tensor waveform,
    Tensor kernel,
    int64_t orig_freq,
    int64_t new_freq,
    int64_t width,
    Tensor out) {
  return fused_resample_cpu(waveform, kernel, orig_freq, new_freq, width, out);
}

STABLE_TORCH_LIBRARY_FRAGMENT(torchaudio, m) {
  m.def(
      "_fused_resample("
      "Tensor waveform,"
      "Tensor kernel,"
      "int orig_freq,"
      "int new_freq,"
      "int width,"
      "Tensor(a!) out) -> Tensor(a!)");
}

STABLE_TORCH_LIBRARY_IMPL(torchaudio, CPU, m) {
  m.impl("_fused_resample", TORCH_BOX(&fused_resample_cpu));
}

STABLE_TORCH_LIBRARY_IMPL(torchaudio, CompositeExplicitAutograd, m) {
  m.impl("_fused_resample", TORCH_BOX(&fused_resample_generic));
}

#if defined(__aarch64__)
#pragma GCC pop_options
#endif
