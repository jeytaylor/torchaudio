#pragma once
#include <torch/csrc/stable/tensor.h>

/**
 * @brief detect_pitch_frequency：基音频率检测，单趟融合 padding + NCCF 自相关 + 峰值检索 + 中值平滑。
 * @param waveform   [in]  输入波形 (..., time)
 * @param sample_rate [in] 采样率
 * @param frame_time [in]  帧时长（秒）
 * @param win_length [in]  窗长
 * @param freq_low   [in]  频率下界（检测范围）
 * @param freq_high  [in]  频率上界（检测范围）
 * @param out        [in/out] 预分配输出（与 waveform 前导 shape + (num_frames,)），原地写入并返回
 * @return 填充后的 out 张量
 * @exception 无
 * @constraint 无
 */
torch::stable::Tensor fused_pitch_cpu(
    torch::stable::Tensor waveform,
    int64_t sample_rate,
    double frame_time,
    int64_t win_length,
    int64_t freq_low,
    int64_t freq_high,
    torch::stable::Tensor out);
