#!/usr/bin/env python
# coding=utf-8
# Copyright: (c) Huawei Technologies Co., Ltd. All rights reserved.

"""torchaudio.fused.amplitude_to_DB：幅值转 dB（融合内核高层封装）。

底层算子：torch.ops.torchaudio._fused_amp_db
"""
import torch

__all__ = ["amplitude_to_DB"]


def amplitude_to_DB(x, multiplier, amin, db_multiplier, top_db=None):
    """
    把幅值/功率谱转为 dB，可选按 batch 的 top_db 下限夹取。

    Args:
        x: 输入频谱
        multiplier: 缩放系数（功率 10 / 幅值 20）
        amin: clamp 下限
        db_multiplier: dB 常数
        top_db: 最小负截止值；None 表示不启用

    Returns:
        Tensor: 与 x 同 shape 的 dB 输出

    约束条件：
    1. **amin > 0**
       amin 是防止 log10 输入为 0 的下界，必须大于 0；
       如果 amin <=0，当 x=0 时 max(x,amin)=0 → log10(0) → -inf，出现 NaN。
       工程常用 1e-10 / 1e-8，不能设 0。
    2. **multiplier 只能选 10 或 20（业务上强约定）**
       没有语法报错，但业务上不能乱填；填错 dB 数值完全不对，模型推理精度掉点。
       这是业务语义约束，不是代码断言约束。
         - 输入是功率谱（power=magnitude²）→ multiplier=10
         - 输入是幅度谱（magnitude）→ multiplier=20
    3. **top_db 为 None 或 top_db > 0**
       top_db 是相对最大幅值的动态范围阈值；
         - top_db <=0 无意义：会把所有值钳位成同一个值，特征失效；
         - top_db=None：关闭动态范围截断。
    4. **db_multiplier > 0**
       几乎永远固定 = 1.0；如果 ≤0，ref 计算异常，dB 值域错乱；
       业务场景几乎不会修改此参数。
    """
    top_val = float("nan") if top_db is None else float(top_db)
    out = torch.empty_like(x)
    return torch.ops.torchaudio._fused_amp_db(
        x, multiplier, amin, db_multiplier, top_val, out)
