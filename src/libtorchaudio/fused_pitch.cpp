// Fused pitch-frequency detection kernel for torchaudio (TA-10).
// Mirrors functional.detect_pitch_frequency EXACTly:
//   - _compute_nccf: per-lag normalized cross-correlation over frames
//     (eliminates the Python `for lag in range(1, lags+1)` loop + per-lag
//      batch of torch dispatches: unfold / mul / sum / vector_norm / stack)
//   - _find_max_per_frame: near-enough-max + lag_min/half combine
//   - _median_smoothing: replicate-left centered median filter
//   - frequency conversion: sample_rate / (EPSILON + indices)
// Registered under INDEPENDENT op name `_fused_detect_pitch_frequency` so the original
// functional implementation remains the reference (single-process A/B).
//
// Signature: caller provides output tensor (Tensor(a!), float, same leading
// shape + (num_frames)). Matches the _fused_sliding_window_cmn/_fused_mulaw out-param style.
#include <libtorchaudio/utils.h>
#include <torch/csrc/stable/library.h>
#include <torch/csrc/stable/ops.h>
#include <torch/csrc/stable/tensor.h>
#include <torch/headeronly/core/Dispatch_v2.h>
#include "fused_pitch.h"

#include <algorithm>
#include <cmath>
#include <vector>

namespace {

using torch::stable::Tensor;

constexpr double kEpsilon = 1e-9;

// NCCF for one (channel, frame): all lags.
// w       : padded waveform scan for this channel (length padded_len)
// start   : frame start index
// out_row : output row (lags) for this frame
template <typename scalar_t>
void nccf_one_frame(
    const scalar_t* w,
    int64_t start,
    int64_t frame_size,
    int64_t lags,
    scalar_t* out_row) {
  for (int64_t lag = 1; lag <= lags; lag++) {
    double acc = 0.0, acc1 = 0.0, acc2 = 0.0;
    int64_t base = start;
    for (int64_t n = 0; n < frame_size; n++) {
      double s1 = (double)w[base + n];
      double s2 = (double)w[base + lag + n];
      acc += s1 * s2;
      acc1 += s1 * s1;
      acc2 += s2 * s2;
    }
    double denom =
        (kEpsilon + std::sqrt(acc1)) * (kEpsilon + std::sqrt(acc1)) *
        (kEpsilon + std::sqrt(acc2)) * (kEpsilon + std::sqrt(acc2));
    out_row[lag - 1] = (denom != 0.0)
        ? static_cast<scalar_t>(acc / denom)
        : static_cast<scalar_t>(0.0);
  }
}

template <typename scalar_t>
void host_pitch(
    const scalar_t* in,
    scalar_t* out,
    int64_t channels,
    int64_t padded_len,
    int64_t num_frames,
    int64_t frame_size,
    int64_t lags,
    int64_t lag_min,
    int64_t win_length,
    scalar_t sample_rate) {
  // nccf layout: (channels, num_frames, lags)
  std::vector<scalar_t> nccf(channels * num_frames * lags, scalar_t(0));
  std::vector<int64_t> raw_idx(channels * num_frames, 0);

  // Pass 1: NCCF per frame (independent, parallel over all channels*frames)
  torch::stable::parallel_for(
      0, channels * num_frames, 8,
      [&](int64_t begin, int64_t end) {
        for (int64_t idx = begin; idx < end; idx++) {
          int64_t b = idx / num_frames;
          int64_t i = idx % num_frames;
          const scalar_t* w = in + b * padded_len;
          scalar_t* row = &nccf[(b * num_frames + i) * lags];
          nccf_one_frame(w, i * frame_size, frame_size, lags, row);
        }
      });

  int64_t half_size = lags / 2;
  int64_t half_end = std::min(half_size, lags);

  // Pass 2: find_max_per_frame -> raw index per frame (parallel)
  torch::stable::parallel_for(
      0, channels * num_frames, 8,
      [&](int64_t begin, int64_t end) {
        for (int64_t idx = begin; idx < end; idx++) {
          int64_t b = idx / num_frames;
          int64_t i = idx % num_frames;
          const scalar_t* row = &nccf[(b * num_frames + i) * lags];

          scalar_t best_val = row[lag_min];
          int64_t best_rel = 0;
          for (int64_t l = lag_min + 1; l < lags; l++) {
            if (row[l] > best_val) { best_val = row[l]; best_rel = l - lag_min; }
          }
          scalar_t half_val = row[lag_min];
          int64_t half_rel = 0;
          for (int64_t l = lag_min + 1; l < half_end; l++) {
            if (row[l] > half_val) { half_val = row[l]; half_rel = l - lag_min; }
          }
          bool use_half = half_val > static_cast<scalar_t>(0.99) * best_val;
          int64_t rel = use_half ? half_rel : best_rel;
          raw_idx[b * num_frames + i] = rel + lag_min + 1;
        }
      });

  // Pass 3: median_smoothing (replicate-left) + frequency conversion.
  // Note: median filtering REDUCES frame count:
  //   out_frames = num_frames + pad_length - win_length + 1
  // (matches reference: indices padded to (num_frames+pad_length) then
  //  unfolded with win_length step 1 -> (num_frames+pad_length-win_length+1)).
  int64_t pad_length = (win_length - 1) / 2;
  int64_t padded_frames = num_frames + pad_length;
  int64_t out_frames = num_frames + pad_length - win_length + 1;
  std::vector<int64_t> med_in(channels * padded_frames, 0);
  for (int64_t b = 0; b < channels; b++) {
    const int64_t* ib = &raw_idx[b * num_frames];
    int64_t* mb = &med_in[b * padded_frames];
    for (int64_t j = 0; j < pad_length; j++)
      mb[j] = ib[0];
    for (int64_t j = 0; j < num_frames; j++)
      mb[pad_length + j] = ib[j];
  }
  int64_t med_rank = (win_length - 1) / 2;  // lower middle for even/odd

  torch::stable::parallel_for(
      0, channels * out_frames, 8,
      [&](int64_t begin, int64_t end) {
        std::vector<int64_t> win(win_length);
        for (int64_t idx = begin; idx < end; idx++) {
          int64_t b = idx / out_frames;
          int64_t i = idx % out_frames;
          const int64_t* mb = &med_in[b * padded_frames];
          for (int64_t k = 0; k < win_length; k++)
            win[k] = mb[i + k];
          std::nth_element(win.begin(), win.begin() + med_rank, win.end());
          double idxd = (double)win[med_rank];
          out[idx] = static_cast<scalar_t>(sample_rate) *
                     static_cast<scalar_t>(1.0 / (kEpsilon + idxd));
        }
      });
}

}  // namespace

Tensor fused_pitch_cpu(
    Tensor waveform,
    int64_t sample_rate,
    double frame_time,
    int64_t win_length,
    int64_t freq_low,
    int64_t freq_high,
    Tensor out) {
  int64_t rank = (int64_t)waveform.sizes().size();
  int64_t wlen = waveform.size(rank - 1);
  int64_t n_channels = 1;
  for (int64_t i = 0; i + 1 < rank; i++)
    n_channels *= waveform.size(i);

  int64_t lags = (int64_t)std::ceil((double)sample_rate / (double)freq_low);
  int64_t frame_size = (int64_t)std::ceil((double)sample_rate * frame_time);
  int64_t num_frames = (int64_t)std::ceil((double)wlen / (double)frame_size);
  int64_t p = lags + num_frames * frame_size - wlen;
  int64_t padded_len = wlen + p;
  int64_t lag_min = (int64_t)std::ceil((double)sample_rate / (double)freq_high);

  THO_DISPATCH_V2(
      waveform.scalar_type(),
      "fused_pitch",
      [&] {
        std::vector<scalar_t> padded(n_channels * padded_len, scalar_t(0));
        const scalar_t* in = waveform.const_data_ptr<scalar_t>();
        for (int64_t b = 0; b < n_channels; b++)
          for (int64_t j = 0; j < wlen; j++)
            padded[b * padded_len + j] = in[b * wlen + j];

        scalar_t* o = out.mutable_data_ptr<scalar_t>();
        host_pitch<scalar_t>(
            padded.data(), o, n_channels, padded_len, num_frames,
            frame_size, lags, lag_min, win_length,
            static_cast<scalar_t>(sample_rate));
      },
      AT_FLOATING_TYPES);
  return out;
}

Tensor fused_pitch_generic(
    Tensor waveform,
    int64_t sample_rate,
    double frame_time,
    int64_t win_length,
    int64_t freq_low,
    int64_t freq_high,
    Tensor out) {
  return fused_pitch_cpu(waveform, sample_rate, frame_time, win_length, freq_low, freq_high, out);
}

STABLE_TORCH_LIBRARY_FRAGMENT(torchaudio, m) {
  m.def(
      "_fused_detect_pitch_frequency("
      "Tensor waveform,"
      "int sample_rate,"
      "float frame_time,"
      "int win_length,"
      "int freq_low,"
      "int freq_high,"
      "Tensor(a!) out) -> Tensor(a!)");
}

STABLE_TORCH_LIBRARY_IMPL(torchaudio, CPU, m) {
  m.impl("_fused_detect_pitch_frequency", TORCH_BOX(&fused_pitch_cpu));
}

STABLE_TORCH_LIBRARY_IMPL(torchaudio, CompositeExplicitAutograd, m) {
  m.impl("_fused_detect_pitch_frequency", TORCH_BOX(&fused_pitch_generic));
}
