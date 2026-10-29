#pragma once
#include <torch/csrc/stable/tensor.h>

/**
 * @brief mulaw：μ-law 编码，逐元素量化到 int64（sign(x)*log1p(mu*|x|)/log1p(mu)）。
 * @param x                   [in]  输入波形，浮点张量
 * @param quantization_channels [in] 量化级数 q，mu = q - 1
 * @param out                 [in/out] 预分配 int64 输出（与 x 同 shape），原地写入并返回
 * @return 填充后的 out 张量
 * @exception 无
 * @constraint 输出为 int64；log_mu = log1p(mu)（float 直接 log1pf，无 double 提升）
 */
torch::stable::Tensor fused_mulaw_cpu(
    torch::stable::Tensor x,
    int64_t quantization_channels,
    torch::stable::Tensor out);
