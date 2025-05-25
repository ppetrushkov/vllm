# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Sequence as GenericSequence
from typing import Optional

import torch
import torch.types

from vllm.road.peft_helper import PEFTRoadHelper
from vllm.utils import is_pin_memory_available


class RoadLayerWeights:
    """LoRA weights for a layer composed of two low rank matrixes."""

    def __init__(
        self,
        module_name: str,
        theta: torch.Tensor,
        alpha: torch.Tensor,
    ) -> None:
        self.module_name = module_name
        self.theta = theta
        self.alpha = alpha


    def optimize(self) -> "RoadLayerWeights":
        """Optimize the LoRA by merging the scaling into lora_b."""
        return self

    #@property
    #def input_dim(self) -> int:
    #    return self.lora_a.shape[0]

    #@property
    #def output_dim(self) -> int:
    #    return self.lora_b.shape[1]

    @property
    def is_packed(self) -> bool:
        return False

    @property
    def extra_vocab_size(self) -> int:
        return 0

    @classmethod
    def from_config(
        cls,
        module_name: str,
        peft_helper: PEFTRoadHelper,
        embeddings_tensor: Optional[torch.Tensor] = None,
    ) -> "RoadLayerWeights":
        return cls(module_name, None, None)

    @classmethod
    def create_dummy_lora_weights(
            cls,
            module_name: str,
            output_dim: int,
            dtype: torch.dtype,
            device: torch.types.Device) -> "RoadLayerWeights":
        pin_memory = str(device) == "cpu" and is_pin_memory_available()
        # Assume we always use ROAD_1 for dummy adapters
        theta = torch.zeros([output_dim//2],
                             dtype=dtype,
                             device=device,
                             pin_memory=pin_memory)
        alpha = torch.ones([output_dim//2],
                             dtype=dtype,
                             device=device,
                             pin_memory=pin_memory)

        return cls(
            module_name,
            theta=theta,
            alpha=alpha,
        )


class PackedRoadLayerWeights(RoadLayerWeights):
    """LoRA used for packed layers (eg. qkv_proj)."""

    def __init__(
        self,
        module_name: str,
        theta: list[Optional[torch.Tensor]],
        alpha: list[Optional[torch.Tensor]],
    ) -> None:
        super().__init__(
            module_name=module_name,
            theta=theta,
            alpha=alpha,
        )

    @classmethod
    def pack(
        cls, loras: GenericSequence[Optional["RoadLayerWeights"]]
    ) -> "PackedRoadLayerWeights":
        """Pack a list of LoRAs into a single LoRA.

        If LoRA is None, it signifies that the submodule does not have a LoRA.
        """
        first_lora = next(lora for lora in loras if lora is not None)
        for lora in loras:
            if lora is None:
                continue
            lora.optimize()
        module_name = first_lora.module_name
        obj = cls(
            module_name,
            [lora.theta if lora is not None else None for lora in loras],
            [lora.alpha if lora is not None else None for lora in loras],
        )
        return obj

    def optimize(self) -> "PackedRoadLayerWeights":
        return self

    @property
    def input_dim(self) -> int:
        raise NotImplementedError()

    @property
    def output_dim(self) -> int:
        raise NotImplementedError()

    @property
    def is_packed(self) -> bool:
        return True
