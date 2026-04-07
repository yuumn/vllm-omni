from __future__ import annotations

from collections.abc import Iterable
from functools import lru_cache
from math import prod
from typing import TYPE_CHECKING, Any

import torch
import torch.nn as nn
import torch.nn.functional as F

import os, json

from vllm.model_executor.layers.linear import (
    ColumnParallelLinear,
    QKVParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
)


# def replace_linear_with_duallora(model, patterns, rank, alpha1, alpha2, use_fp8=False):
#     """
#     先冻结model所有参数，
#     将匹配patterns的nn.Linear替换为DualLoRALinear
#     不匹配的替换为 LinearFP8Wrapper
#     """
#     def to_fp8(tensor):
#         # 仅在 use_fp8 时转换
#         if use_fp8:
#             return tensor.to(dtype=torch.float8_e4m3fn)
#         return tensor
    
#     # # 冻结全部参数
#     # for p in model.parameters():
#     #     p.requires_grad = False

#     def _replace_module(parent, name_prefix=""):
#         for name, module in list(parent.named_children()):
#             full_name = f"{name_prefix}{name}"
#             if isinstance(module, ColumnParallelLinear) or 
#                isinstance(module, ReplicatedLinear) or
#                isinstance(module, RowParallelLinear):
#                 module.weight.data = to_fp8(module.weight.data)
#                 if module.bias is not None:
#                     module.bias.data = to_fp8(module.bias.data)

#                 if any(p in full_name for p in patterns):
#                     # 替换为 DualLoRALinear
#                     new_module = DualLoRALinear(module, rank, alpha1, alpha2)
#                     setattr(parent, name, new_module)
#                     print(f"[lora] {full_name} -> DualLoRALinear")
#                 else: 
#                     # 替换为自动精度转换的 FP8LinearWrapper
#                     new_module = LinearFP8Wrapper(module)
#                     setattr(parent, name, new_module)
#                     print(f"[cast] {full_name} -> LinearFP8Wrapper (fp8={use_fp8})")
#             elif isinstance(module, QKVParallelLinear):
#                 pass

#             else:
#                 _replace_module(module, name_prefix=full_name + ".")
    
#     _replace_module(model)



class LinearFP8Wrapper(torch.nn.Module):
    """普通 Linear forward 时自动 cast 到输入 dtype"""
    def __init__(self, base_linear):
        super().__init__()
        self.base_linear = base_linear
        # assert isinstance(linear, torch.nn.Linear)
        self.weight = base_linear.weight
        self.bias = base_linear.bias
        # # 冻结参数
        # self.weight.requires_grad = False
        # if self.bias is not None:
        #     self.bias.requires_grad = False

    def forward(self, x):
        self.base_linear = self.base_linear.to(device=x.device, dtype=x.dtype)
        return self.base_linear(x)
        # w = self.weight.to(dtype=x.dtype)
        # b = self.bias.to(dtype=x.dtype) if self.bias is not None else None
        # return torch.nn.functional.linear(x, w, b)

class DualLoRALinear(torch.nn.Module):
    """
    对序列前一半/后一半分别使用不同 LoRA 的 Linear 包装器。
    原 Linear 权重原地共享，不额外复制。
    """
    def __init__(self, linear, rank, alpha1, alpha2):
        super().__init__()
        # assert isinstance(linear, torch.nn.Linear)
        self.rank    = rank
        self.alpha1  = alpha1
        self.alpha2  = alpha2
        self.linear  = linear

        dev = linear.weight.device
        dt  = torch.bfloat16

        # 两套低秩矩阵，参数量 2 * (in * rank + rank * out)
        self.lora_A1 = torch.nn.Linear(linear.input_size, rank, bias=False).to(device=dev, dtype=dt)
        self.lora_B1 = torch.nn.Linear(rank, linear.output_size, bias=False).to(device=dev, dtype=dt)
        self.lora_A2 = torch.nn.Linear(linear.input_size, rank, bias=False).to(device=dev, dtype=dt)
        self.lora_B2 = torch.nn.Linear(rank, linear.output_size, bias=False).to(device=dev, dtype=dt)

        self.scaling1 = alpha1 /  max(1, rank)
        self.scaling2 = alpha2 /  max(1, rank)

        torch.nn.init.normal_(self.lora_A1.weight, std=1.0 / rank)
        torch.nn.init.zeros_(self.lora_B1.weight)
        torch.nn.init.normal_(self.lora_A2.weight, std=1.0 / rank)
        torch.nn.init.zeros_(self.lora_B2.weight)

    def _base_linear(self, x):
        # 仅在计算里把 float8 权重/偏置转成 x.dtype（bf16）
        w = self.linear.weight.detach().to(dtype=x.dtype)
        b = None
        if self.linear.bias is not None:
            b = self.linear.bias.detach().to(dtype=x.dtype)
        return torch.nn.functional.linear(x, w, b)

    def forward(self, x):
        self.linear = self.linear.to(device=x.device, dtype=x.dtype)
        y = self._base_linear(x)
        return y 
        # self.lora_A1 = self.lora_A1.to(x.device)
        # self.lora_B1 = self.lora_B1.to(x.device)
        # self.lora_A2 = self.lora_A2.to(x.device)
        # self.lora_B2 = self.lora_B2.to(x.device)
        # if x.ndim == 2:
        #     y = self._base_linear(x)  # (B, C_out)
        #     delta1 = self.lora_B1(self.lora_A1(x)) * self.scaling1
        #     delta2 = self.lora_B2(self.lora_A2(x)) * self.scaling2
        #     y1 = y + delta1
        #     y2 = y + delta2
        #     return torch.stack([y1, y2], dim=1)  # (B, 2, C_out)
        # else:
        #     B, L2, _ = x.shape
        #     assert L2 % 2 == 0, "sequence length must be even"
        #     L = L2 // 2

        #     y = self._base_linear(x)                           # [B, 2L, C_out]
        #     x1 = x[:, :L, :]                             # [B, L, C_in]
        #     x2 = x[:, L:, :]                             # [B, L, C_in]

        #     delta1 = self.lora_B1(self.lora_A1(x1)) * self.scaling1   # [B, L, C_out]
        #     delta2 = self.lora_B2(self.lora_A2(x2)) * self.scaling2   # [B, L, C_out]

        #     y[:, :L] += delta1
        #     y[:, L:] += delta2
        #     return y

class DualLoRAQKVLinear(torch.nn.Module):
    """
    对序列前一半/后一半分别使用不同 LoRA 的 Linear 包装器。
    原 Linear 权重原地共享，不额外复制。
    """
    def __init__(self, linear, rank, alpha1, alpha2):
        super().__init__()
        # assert isinstance(linear, torch.nn.Linear)
        self.rank    = rank
        self.alpha1  = alpha1
        self.alpha2  = alpha2
        self.linear  = linear

        dev = linear.weight.device
        dt  = torch.bfloat16

        # 两套低秩矩阵，参数量 2 * (in * rank + rank * out)
        self.q_lora_A1 = torch.nn.Linear(linear.input_size, rank, bias=False).to(device=dev, dtype=dt)
        self.q_lora_B1 = torch.nn.Linear(rank, linear.output_size, bias=False).to(device=dev, dtype=dt)
        self.q_lora_A2 = torch.nn.Linear(linear.input_size, rank, bias=False).to(device=dev, dtype=dt)
        self.q_lora_B2 = torch.nn.Linear(rank, linear.output_size, bias=False).to(device=dev, dtype=dt)

        self.k_lora_A1 = torch.nn.Linear(linear.input_size, rank, bias=False).to(device=dev, dtype=dt)
        self.k_lora_B1 = torch.nn.Linear(rank, linear.output_size, bias=False).to(device=dev, dtype=dt)
        self.k_lora_A2 = torch.nn.Linear(linear.input_size, rank, bias=False).to(device=dev, dtype=dt)
        self.k_lora_B2 = torch.nn.Linear(rank, linear.output_size, bias=False).to(device=dev, dtype=dt)

        self.v_lora_A1 = torch.nn.Linear(linear.input_size, rank, bias=False).to(device=dev, dtype=dt)
        self.v_lora_B1 = torch.nn.Linear(rank, linear.output_size, bias=False).to(device=dev, dtype=dt)
        self.v_lora_A2 = torch.nn.Linear(linear.input_size, rank, bias=False).to(device=dev, dtype=dt)
        self.v_lora_B2 = torch.nn.Linear(rank, linear.output_size, bias=False).to(device=dev, dtype=dt)

        self.scaling1 = alpha1 /  max(1, rank)
        self.scaling2 = alpha2 /  max(1, rank)

        torch.nn.init.normal_(self.q_lora_A1.weight, std=1.0 / rank)
        torch.nn.init.zeros_(self.q_lora_B1.weight)
        torch.nn.init.normal_(self.q_lora_A2.weight, std=1.0 / rank)
        torch.nn.init.zeros_(self.q_lora_B2.weight)

    def _base_linear(self, x):
        # 仅在计算里把 float8 权重/偏置转成 x.dtype（bf16）
        w = self.linear.weight.detach().to(dtype=x.dtype)
        b = None
        if self.linear.bias is not None:
            b = self.linear.bias.detach().to(dtype=x.dtype)
        return torch.nn.functional.linear(x, w, b)
    
    def forward(self, x):
        self.linear = self.linear.to(device=x.device, dtype=x.dtype)
        # y = self._base_linear(x)
        return self.linear(x)
        if x.ndim == 2:
            y = self._base_linear(x)  # (B, C_out)
            delta1 = self.lora_B1(self.lora_A1(x)) * self.scaling1
            delta2 = self.lora_B2(self.lora_A2(x)) * self.scaling2
            y1 = y + delta1
            y2 = y + delta2
            return torch.stack([y1, y2], dim=1)  # (B, 2, C_out)
        else:
            B, L2, _ = x.shape
            assert L2 % 2 == 0, "sequence length must be even"
            L = L2 // 2

            y = self._base_linear(x)                           # [B, 2L, C_out]
            x1 = x[:, :L, :]                             # [B, L, C_in]
            x2 = x[:, L:, :]                             # [B, L, C_in]

            delta1 = self.lora_B1(self.lora_A1(x1)) * self.scaling1   # [B, L, C_out]
            delta2 = self.lora_B2(self.lora_A2(x2)) * self.scaling2   # [B, L, C_out]

            y[:, :L] += delta1
            y[:, L:] += delta2
            return y


# class DualLoRAColumnParallelLinear(ColumnParallelLinear):

#     def __init__(self, *args, **kwargs):
#         super().__init__(*args, **kwargs)
#         lora_rank = 128
#         self.rank=lora_rank
#         self.alpha1=0
#         self.alpha2=lora_rank
#         # self.use_fp8 = True
#         self.dual_lora_linear = True
    
#     def forward(self, x):
#         y = super().forward(x)

