#pragma once
#include <torch/csrc/stable/tensor.h>

/**
 * @brief sliding_window_cmn：滑窗倒谱均值（可选方差）归一化，增量累加器 O(1) 更新。
 * @param specgram       [in]  输入频谱 (..., time, freq)
 * @param cmn_window     [in]  滑窗帧数
 * @param min_cmn_window [in]  最小窗（仅 center=False 时生效）
 * @param center         [in]  是否以当前帧为中心
 * @param norm_vars      [in]  是否归一化方差
 * @param out            [in/out] 预分配输出（与 specgram 同 shape），原地写入并返回
 * @return 填充后的 out 张量
 * @exception 无
 * @constraint 增量累加顺序与 reference 逐位一致；window_frames==1 且 norm_vars 时输出 0
 */
torch::stable::Tensor fused_cmn_cpu(
    torch::stable::Tensor specgram,
    int64_t cmn_window,
    int64_t min_cmn_window,
    bool center,
    bool norm_vars,
    torch::stable::Tensor out);
