"""Dual-LoRA wrappers for vLLM parallel linear layers.

Provides ColumnParallelLinear (and stubs for other parallel layers) that
support two independent LoRA branches on top of the frozen base linear.

Forward computation:
    output = base(x) + (x @ A1.T @ B1.T) * (alpha1 / rank)
                      + (x @ A2.T @ B2.T) * (alpha2 / rank)

LoRA weights are NOT initialized here -- they are expected to be loaded
from an external checkpoint (e.g. ODTSR/weight.pth).
"""

from __future__ import annotations

import torch
import torch.nn as nn

from vllm.model_executor.layers.linear import (
    ColumnParallelLinear as BaseColumnParallelLinear,
)
from vllm.model_executor.layers.linear import (
    QKVParallelLinear as BaseQKVParallelLinear,
)
from vllm.model_executor.layers.linear import (
    ReplicatedLinear as BaseReplicatedLinear,
)
from vllm.model_executor.layers.linear import (
    RowParallelLinear as BaseRowParallelLinear,
)


class DualLoRAColumnParallelLinear(nn.Module):
    """ColumnParallelLinear wrapper with two independent LoRA branches.

    Wraps an existing ``BaseColumnParallelLinear`` instance and adds two
    low-rank adaptation branches.  The base linear weights are kept frozen;
    only the LoRA parameters are trainable (if training is needed).

    In tensor-parallel (TP) deployments the base layer already shards its
    output dimension across ranks.  The LoRA-B matrices follow the same
    sharding: their row count equals ``output_size_per_partition`` so the
    delta is added to the local shard directly, without extra communication.

    Args:
        base_layer: A pre-existing ``ColumnParallelLinear`` whose weights are
            already loaded / quantised.
        rank: The low-rank dimension *r* shared by both LoRA branches.
        alpha1: Scaling factor for the first LoRA branch.
        alpha2: Scaling factor for the second LoRA branch.
        lora_dtype: Data type for LoRA parameters.  Defaults to the base
            layer's ``params_dtype`` (typically float16 / bfloat16).
    """

    def __init__(
        self,
        base_layer: BaseColumnParallelLinear,
        rank: int,
        alpha1: float,
        alpha2: float,
        lora_dtype: torch.dtype | None = None,
    ):
        super().__init__()

        self.base_layer = base_layer
        self.rank = rank
        self.alpha1 = alpha1
        self.alpha2 = alpha2

        # Scaling factors following the standard LoRA convention:
        #   delta = (x @ A.T @ B.T) * (alpha / rank)
        self.scale1 = alpha1 / rank if rank > 0 else 0.0
        self.scale2 = alpha2 / rank if rank > 0 else 0.0

        # Derive dimensions from the base layer.
        # ColumnParallelLinear shards the output dim:
        #   weight shape = [output_size_per_partition, input_size]
        in_features = base_layer.input_size  # full (unsharded) input dim
        out_features_local = base_layer.output_size_per_partition  # per-TP-rank

        if lora_dtype is None:
            lora_dtype = getattr(base_layer, "params_dtype", torch.float16)

        # ---- LoRA branch 1 ----
        # A1: (rank, in_features)  -- down-projection, NOT sharded
        # B1: (out_features_local, rank) -- up-projection, sharded along output
        self.lora_a1 = nn.Parameter(torch.empty(rank, in_features, dtype=lora_dtype), requires_grad=False)
        self.lora_b1 = nn.Parameter(
            torch.empty(out_features_local, rank, dtype=lora_dtype),
            requires_grad=False,
        )

        # ---- LoRA branch 2 ----
        self.lora_a2 = nn.Parameter(torch.empty(rank, in_features, dtype=lora_dtype), requires_grad=False)
        self.lora_b2 = nn.Parameter(
            torch.empty(out_features_local, rank, dtype=lora_dtype),
            requires_grad=False,
        )

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def forward(self, input_: torch.Tensor) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor | None]:
        """Run the base linear then add dual-LoRA deltas.

        Handles both return modes of ``ColumnParallelLinear``:
        - When ``return_bias=False``: returns ``Tensor``
        - When ``return_bias=True`` (default): returns ``(Tensor, bias | None)``
        """
        base_out = self.base_layer(input_)

        # Unpack the base output -- may be a tuple (output, bias).
        if isinstance(base_out, tuple):
            output, output_bias = base_out
        else:
            output = base_out
            output_bias = None

        # Compute LoRA deltas.  We work in float32 for the low-rank matmuls
        # to preserve numerical accuracy, then cast back.
        orig_dtype = output.dtype

        if self.scale1 != 0.0:
            x_float = input_.to(torch.float32)
            # shrink: (*, in) @ (in, rank) -> (*, rank)
            lora_out1 = x_float @ self.lora_a1.to(torch.float32).t()
            # expand: (*, rank) @ (rank, out_local) -> (*, out_local)
            lora_out1 = lora_out1 @ self.lora_b1.to(torch.float32).t()
            output = output + (lora_out1 * self.scale1).to(orig_dtype)

        if self.scale2 != 0.0:
            if self.scale1 == 0.0:
                x_float = input_.to(torch.float32)
            lora_out2 = x_float @ self.lora_a2.to(torch.float32).t()
            lora_out2 = lora_out2 @ self.lora_b2.to(torch.float32).t()
            output = output + (lora_out2 * self.scale2).to(orig_dtype)

        if output_bias is not None:
            return output, output_bias
        return output

    # ------------------------------------------------------------------
    # Weight loading helpers
    # ------------------------------------------------------------------
    def set_lora_weights(
        self,
        lora_a1: torch.Tensor | None = None,
        lora_b1: torch.Tensor | None = None,
        lora_a2: torch.Tensor | None = None,
        lora_b2: torch.Tensor | None = None,
    ) -> None:
        """Load LoRA weights from tensors (e.g. from a state_dict).

        Each tensor should already be correctly shaped / sharded for the
        current TP rank.  Pass ``None`` for branches you don't want to update.
        """
        if lora_a1 is not None:
            assert lora_a1.shape == self.lora_a1.shape, (
                f"lora_a1 shape mismatch: expected {self.lora_a1.shape}, got {lora_a1.shape}"
            )
            self.lora_a1.data.copy_(lora_a1)
        if lora_b1 is not None:
            assert lora_b1.shape == self.lora_b1.shape, (
                f"lora_b1 shape mismatch: expected {self.lora_b1.shape}, got {lora_b1.shape}"
            )
            self.lora_b1.data.copy_(lora_b1)
        if lora_a2 is not None:
            assert lora_a2.shape == self.lora_a2.shape, (
                f"lora_a2 shape mismatch: expected {self.lora_a2.shape}, got {lora_a2.shape}"
            )
            self.lora_a2.data.copy_(lora_a2)
        if lora_b2 is not None:
            assert lora_b2.shape == self.lora_b2.shape, (
                f"lora_b2 shape mismatch: expected {self.lora_b2.shape}, got {lora_b2.shape}"
            )
            self.lora_b2.data.copy_(lora_b2)

    def set_scales(
        self,
        alpha1: float | None = None,
        alpha2: float | None = None,
    ) -> None:
        """Dynamically update LoRA scaling factors at runtime."""
        if alpha1 is not None:
            self.alpha1 = alpha1
            self.scale1 = alpha1 / self.rank if self.rank > 0 else 0.0
        if alpha2 is not None:
            self.alpha2 = alpha2
            self.scale2 = alpha2 / self.rank if self.rank > 0 else 0.0

    # ------------------------------------------------------------------
    # Attribute forwarding
    # ------------------------------------------------------------------
    def __getattr__(self, name: str):
        """Forward unknown attribute lookups to the wrapped base layer.

        This allows upstream code that expects a ``ColumnParallelLinear`` to
        transparently access attributes like ``output_size_per_partition``,
        ``tp_size``, ``gather_output``, etc.
        """
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.base_layer, name)

    def extra_repr(self) -> str:
        return (
            f"rank={self.rank}, "
            f"alpha1={self.alpha1}, alpha2={self.alpha2}, "
            f"scale1={self.scale1:.4f}, scale2={self.scale2:.4f}, "
            f"in_features={self.base_layer.input_size}, "
            f"out_features_local={self.base_layer.output_size_per_partition}"
        )
