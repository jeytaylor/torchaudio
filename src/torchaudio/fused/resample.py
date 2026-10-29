#!/usr/bin/env python
# coding=utf-8
# Copyright: (c) Huawei Technologies Co., Ltd. All rights reserved.

"""torchaudio.fused.resample：多相 sinc 插值重采样（融合内核高层封装）。

底层算子：torch.ops.torchaudio._fused_resample
"""
import math

import torch
import torchaudio.functional as F
from torchaudio.functional.functional import _get_sinc_resample_kernel

__all__ = ["resample"]


def resample(waveform, orig_freq, new_freq, lowpass_filter_width=6,
             rolloff=0.99, resampling_method="sinc_interp_hann", beta=None):
    """
    按带限 sinc 插值把波形重采样到新采样率，走融合 C++ 内核。

    Args:
        waveform: 输入信号，维度 (..., time)
        orig_freq: 原始采样率
        new_freq: 目标采样率
        lowpass_filter_width: 低通滤波核宽度（越宽越锐利但越慢）
        rolloff: 滚降系数（相对 Nyquist 的比例，默认 0.99）
        resampling_method: sinc_interp_hann / sinc_interp_kaiser
        beta: kaiser 窗形状参数（仅 kaiser 有效）

    Returns:
        Tensor: (..., ceil(new_freq*T/orig_freq)) 的重采样结果

    Raises:
        ValueError: orig_freq/new_freq 非正，或重采样方法非法
    """
    if orig_freq <= 0.0 or new_freq <= 0.0:
        raise ValueError("Original frequency and desired frequency should be positive")
    if orig_freq == new_freq:
        return waveform
    gcd = math.gcd(int(orig_freq), int(new_freq))
    kernel, width = _get_sinc_resample_kernel(
        int(orig_freq), int(new_freq), gcd, lowpass_filter_width, rolloff,
        resampling_method, beta, waveform.device, waveform.dtype)
    kernel = kernel.to(waveform.dtype).contiguous()
    of_r, nf_r = int(orig_freq) // gcd, int(new_freq) // gcd
    tgt = int(math.ceil(nf_r * waveform.shape[-1] / of_r))
    out = waveform.new_empty(list(waveform.shape[:-1]) + [tgt])
    return torch.ops.torchaudio._fused_resample(waveform, kernel, of_r, nf_r, width, out)
