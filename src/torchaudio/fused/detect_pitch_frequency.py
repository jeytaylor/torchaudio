#!/usr/bin/env python
# coding=utf-8
# Copyright: (c) Huawei Technologies Co., Ltd. All rights reserved.

"""torchaudio.fused.detect_pitch_frequency：基音频率检测（融合内核高层封装）。

底层算子：torch.ops.torchaudio._fused_detect_pitch_frequency
"""
import math

import torch

__all__ = ["detect_pitch_frequency"]


def detect_pitch_frequency(waveform, sample_rate, frame_time=0.01,
                           win_length=30, freq_low=85, freq_high=3400):
    """
    检测信号基音频率。

    Args:
        waveform: 输入波形 (..., time)
        sample_rate: 采样率
        frame_time: 帧时长（秒）
        win_length: 窗长
        freq_low: 频率下界
        freq_high: 频率上界

    Returns:
        Tensor: (..., num_frames) 的基音频率
    """
    wlen = waveform.shape[-1]
    frame_size = int(math.ceil(sample_rate * frame_time))
    num_frames = int(math.ceil(wlen / frame_size))
    pad_length = (win_length - 1) // 2
    out_frames = num_frames + pad_length - win_length + 1
    out = waveform.new_empty(list(waveform.shape[:-1]) + [out_frames])
    return torch.ops.torchaudio._fused_detect_pitch_frequency(
        waveform, int(sample_rate), float(frame_time), int(win_length),
        int(freq_low), int(freq_high), out)
