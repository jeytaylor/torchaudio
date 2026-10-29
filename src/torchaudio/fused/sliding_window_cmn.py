#!/usr/bin/env python
# coding=utf-8
# Copyright: (c) Huawei Technologies Co., Ltd. All rights reserved.

"""torchaudio.fused.sliding_window_cmn：滑窗倒谱均值/方差归一化（融合内核封装）。

底层算子：torch.ops.torchaudio._fused_sliding_window_cmn
"""
import torch

__all__ = ["sliding_window_cmn"]


def sliding_window_cmn(specgram, cmn_window=600, min_cmn_window=100,
                       center=False, norm_vars=False):
    """
    对频谱做滑窗倒谱均值（可选方差）归一化。

    Args:
        specgram: 输入频谱 (..., time, freq)
        cmn_window: 滑窗帧数
        min_cmn_window: 最小窗（仅 center=False 生效）
        center: 是否以当前帧为中心
        norm_vars: 是否归一化方差

    Returns:
        Tensor: 与 specgram 同 shape
    """
    out = torch.empty_like(specgram)
    return torch.ops.torchaudio._fused_sliding_window_cmn(
        specgram, int(cmn_window), int(min_cmn_window), bool(center), bool(norm_vars), out)
