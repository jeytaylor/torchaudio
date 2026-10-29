#!/usr/bin/env python
# coding=utf-8
# Copyright: (c) Huawei Technologies Co., Ltd. All rights reserved.
"""torchaudio.fused：融合算子高层 Python 接口（独立命名空间，不覆盖原 functional 实现）。

每个函数封装一个 torch.ops.torchaudio._fused_* 底层算子：
- resample           -> _fused_resample
- mu_law_encoding    -> _fused_mulaw
- amplitude_to_DB    -> _fused_amp_db
- DB_to_amplitude    -> _fused_db2amp
- sliding_window_cmn -> _fused_sliding_window_cmn
- detect_pitch_frequency -> _fused_detect_pitch_frequency
"""
from .resample import resample
from .mulaw import mu_law_encoding
from .amp_db import amplitude_to_DB
from .db2amp import DB_to_amplitude
from .sliding_window_cmn import sliding_window_cmn
from .detect_pitch_frequency import detect_pitch_frequency

__all__ = [
    "resample",
    "mu_law_encoding",
    "amplitude_to_DB",
    "DB_to_amplitude",
    "sliding_window_cmn",
    "detect_pitch_frequency",
]
