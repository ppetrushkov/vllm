# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import dataclass
import math
from typing import Optional, Union

import torch

from vllm.road.config import RoadVariant
from vllm.road.layers import prepare_weights
from vllm.road.weights import RoadLayerWeights, PackedRoadLayerWeights


class DummyRoadManager:

    def __init__(self, device: torch.device = "cuda:0"):
        super().__init__()
        self._weights: dict[str, RoadLayerWeights] = {}
        self._device = device

    def init_random_lora(
        self,
        module_name: str,
        output_size: int,
        dtype,
        variant: RoadVariant,
    ):
        if variant == RoadVariant.ROAD_1:
            output_size = output_size // 2
        elif variant == RoadVariant.ROAD_2:
            output_size = output_size
        elif variant == RoadVariant.ROAD_4:
            output_size = output_size * 2
        weight = RoadLayerWeights(
            module_name,
            theta=torch.rand([output_size],
                              dtype=dtype,
                              device=self._device) * 2 * math.pi,
            alpha=torch.rand([output_size],
                              dtype=dtype,
                              device=self._device) / 2 + 0.5,
        )

        return weight

    #def init_lora(
    #    self,
    #    module_name: str,
    #    input_dim: int,
    #    output_dim: int,
    #    rank=8,
    #    noop=False,
    #    embeddings_tensor=None,
    #):
    #    lora = LoRALayerWeights(
    #        module_name,
    #        rank=rank,
    #        lora_alpha=1,
    #        lora_a=torch.rand([input_dim, rank], device="cuda"),
    #        lora_b=torch.rand([rank, output_dim], device="cuda"),
    #        embeddings_tensor=embeddings_tensor,
    #    )
    #    self.set_module_lora(module_name, lora)
    #    return lora

    #def reset_lora(self):
    #    self._loras = {}

    #def init_packed_lora(
    #    self,
    #    module_name: str,
    #    input_dim: int,
    #    output_dims: list[int],
    #    noop_lora_index: Optional[list[int]] = None,
    #    rank: int = 8,
    #):
    #    base_loras: list[LoRALayerWeights] = []
    #    noop_lora_index_set = set(noop_lora_index or [])

    #    for i, out_dim in enumerate(output_dims):
    #        base_lora = self.init_lora(
    #            module_name + "_000_" + str(i),
    #            input_dim,
    #            out_dim,
    #            rank=rank,
    #            noop=i in noop_lora_index_set,
    #        )
    #        base_loras.append(base_lora)
    #    packed_lora = PackedLoRALayerWeights.pack(base_loras)
    #    self.set_module_lora(module_name, packed_lora)
    #    return packed_lora


def assert_close(a, b):
    rtol, atol = {
        torch.float16: (6e-2, 6e-2),
        torch.bfloat16: (6e-2, 6e-2),
        torch.float32: (1e-2, 1e-2),
    }[a.dtype]
    torch.testing.assert_close(a, b, rtol=rtol, atol=atol)


def _apply_road(theta: torch.Tensor, alpha: torch.Tensor, variant: RoadVariant, group_size: int, x: torch.Tensor):

    first_col, second_col = prepare_weights(theta, alpha, variant, group_size)

    # Reference torch implementation
    x_grouped = x.reshape(-1, 2, group_size // 2)
    x1 = x_grouped[:, 0, :]
    x2 = x_grouped[:, 1, :]
    rotate_half_x = torch.stack((-x2, x1), dim=1).reshape(x.shape)
    result = x * first_col + rotate_half_x * second_col

    return result
