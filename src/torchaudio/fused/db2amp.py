#!/usr/bin/env python
# coding=utf-8
# Copyright: (c) Huawei Technologies Co., Ltd. All rights reserved.

"""torchaudio.fused.DB_to_amplitude：dB 转幅值（融合内核高层封装）。

底层算子：torch.ops.torchaudio._fused_db2amp
"""
import torch

__all__ = ["DB_to_amplitude"]


def DB_to_amplitude(x, ref, power):
    """
    把 dB 值转回幅值/功率。

    Args:
        x: dB 值张量
        ref: 参考缩放值
        power: 幂次（1：功率，0.5：幅值）

    Returns:
        Tensor: 与 x 同 shape
    """
    out = torch.empty_like(x)
    return torch.ops.torchaudio._fused_db2amp(x, ref, power, out)
