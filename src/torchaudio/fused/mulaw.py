#!/usr/bin/env python
# coding=utf-8
# Copyright: (c) Huawei Technologies Co., Ltd. All rights reserved.

"""torchaudio.fused.mu_law_encoding：μ-law 编码（融合内核高层封装）。

底层算子：torch.ops.torchaudio._fused_mulaw
"""
import torch

__all__ = ["mu_law_encoding"]


def mu_law_encoding(x, quantization_channels):
    """
    对信号进行 μ-law 编码，量化到 int64。

    Args:
        x: 输入波形，浮点
        quantization_channels: 量化级数 q，mu = q - 1

    Returns:
        Tensor: int64，与 x 同 shape
    """
    out = torch.empty_like(x, dtype=torch.int64)
    return torch.ops.torchaudio._fused_mulaw(x, int(quantization_channels), out)
