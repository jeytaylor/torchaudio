#pragma once
#include <torch/csrc/stable/tensor.h>

/**
 * @brief amplitude_to_DB：幅值转 dB（clamp + log10 + 缩放），可选 top_db 逐 batch 归约夹下限。
 * @param x            [in]  输入幅值谱张量
 * @param multiplier   [in]  缩放系数（功率 10 / 幅值 20）
 * @param amin         [in]  clamp 下限
 * @param db_multiplier [in] dB 常数
 * @param top_db       [in]  最小负截止值；NaN 表示不启用 top_db 路径
 * @param out          [in/out] 预分配输出，原地写入并返回
 * @return 填充后的 out 张量
 * @exception 无
 * @constraint top_db 为 NaN 时跳过归约；float/double 均支持
 */
torch::stable::Tensor fused_amp_db_cpu(
    torch::stable::Tensor x,
    double multiplier,
    double amin,
    double db_multiplier,
    double top_db,
    torch::stable::Tensor out);

/**
 * @brief DB_to_amplitude：dB 转幅值（单趟 ref*exp2(x*k)，k 为循环不变式系数）。
 * @param x     [in]  dB 值张量
 * @param ref   [in]  参考缩放值
 * @param power [in]  幂次（1：功率，0.5：幅值）
 * @param out   [in/out] 预分配输出，原地写入并返回
 * @return 填充后的 out 张量
 * @exception 无
 * @constraint 无
 */
torch::stable::Tensor fused_db2amp_cpu(
    torch::stable::Tensor x,
    double ref,
    double power,
    torch::stable::Tensor out);
