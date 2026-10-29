#pragma once
#include <torch/csrc/stable/tensor.h>

/**
 * @brief resample：多相 sinc 插值重采样内核，把 pad + conv1d + 转置 + 截取融合为单趟并行 pass。
 * @param waveform  [in]  输入波形 (..., time)，float32 普通 CPU 张量
 * @param kernel    [in]  预生成的 sinc 滤波器组 (new_freq, 1, K)，K = kernel.size(-1)
 * @param orig_freq [in]  gcd 约分后的原始采样率
 * @param new_freq  [in]  gcd 约分后的目标采样率
 * @param width     [in]  滤波核半宽
 * @param out       [in/out] 预分配输出 (..., tgt_len)，原地写入并返回
 * @return 填充后的 out 张量
 * @exception 无
 * @constraint 频率须为整数且已 gcd 约分；kernel 由 _get_sinc_resample_kernel 预计算；
 *             tgt_len = ceil(new_freq*len/orig_freq)；float 走 SVE 向量化
 */
torch::stable::Tensor fused_resample_cpu(
    torch::stable::Tensor waveform,
    torch::stable::Tensor kernel,
    int64_t orig_freq,
    int64_t new_freq,
    int64_t width,
    torch::stable::Tensor out);
