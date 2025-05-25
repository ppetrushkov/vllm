# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Based on:
Chen, L., Ye, Z., Wu, Y., Zhuo, D., Ceze, L., & Krishnamurthy, A. (2023). 
Punica: Multi-Tenant LoRA Serving. 
https://arxiv.org/abs/2310.18547
"""

from typing import TYPE_CHECKING, Optional, Union, final

import torch

import vllm.envs as envs
from vllm.lora.layers import LoRAMapping
from vllm.triton_utils import HAS_TRITON

if HAS_TRITON:
    from vllm.lora.ops.triton_ops import (LoRAKernelMeta, lora_expand,
                                          lora_shrink)
    from vllm.road.triton_ops import road_triton_internal_mixed

from .road_wrapper import RoadWrapperBase

if TYPE_CHECKING:
    # avoid circuit import
    from vllm.lora.models import LongContextLoRAContext


@final
class RoadWrapperGPU(RoadWrapperBase):
    """
    PunicaWrapperGPU is designed to manage and provide metadata for the punica 
    kernel. The main function is to maintain the state information for 
    Multi-LoRA, and to provide the interface for the punica triton kernel.
    """

    def __init__(self, max_num_batched_tokens: int, max_batches: int,
                 device: Union[torch.device, str], group_size: int, **kwargs):
        RoadWrapperBase.__init__(self, max_num_batched_tokens, max_batches,
                                   device, group_size)

        self.max_adapters = kwargs['max_adapters']

        self.token_mapping_meta = LoRAKernelMeta.make(self.max_adapters,
                                                      max_num_batched_tokens,
                                                      device=device)

        # When cudagraph capture size is greater than max_num_seqs (max_batches,
        # here), V0 captures the graph as if max_num_seqs is set to
        # the capture size.
        # V1 doesn't have this problem and always respects max_num_seqs.
        max_num_prompts = (max_batches
                           if envs.VLLM_USE_V1 else max_num_batched_tokens)
        self.prompt_mapping_meta = LoRAKernelMeta.make(self.max_adapters,
                                                       max_num_prompts,
                                                       device=device)

    def update_metadata(
            self,
            mapping: LoRAMapping,
            adapter_index_to_id: list[Optional[int]],
            max_adapters: int,
            vocab_size: int,
            **kwargs):

        self.is_prefill = mapping.is_prefill
        self._update_base_metadata(mapping, adapter_index_to_id, max_adapters,
                                   vocab_size)#, extra_vocab_size,
                                   #long_lora_context)

        # Prepare cuda kernel metadata tensors
        self.token_mapping_meta.prepare_tensors(self.token_lora_indices)
        self.prompt_mapping_meta.prepare_tensors(self.sampler_indices)

    def add_linear(self,
                   x: torch.Tensor,
                   first_column_stacked: tuple[torch.Tensor, ...],
                   second_column_stacked: tuple[torch.Tensor, ...],
                   output_slices: tuple[int, ...],
                   *,
                   buffer: Optional[torch.Tensor] = None,
                   **kwargs) -> None:
        """
        Applicable to linear-related lora. 

        Semantics:
            for i in range(len(lora_a_stacked)):
                y[i] += (
                    x[i].unsqueeze(0)
                    @ lora_a_stacked[indices[i], layer_idx, :, :]
                    @ lora_b_stacked[indices[i], layer_idx, :, :]
                    * scale
                    ).squeeze(0)+lora_bias_stacked[i]

        Args:
            y (torch.Tensor): Output tensor. Will be changed in-place.
            x (torch.Tensor): Input tensor
            lora_a_stacked (tuple[torch.Tensor, ...]): lora_a's weight.
            lora_b_stacked (tuple[torch.Tensor, ...]): lora_b's weight.
            lora_bias_stacked (Optional[tuple[torch.Tensor, ...]]): lora's bias.
            scale (float): Scaling factor.
            output_slices (tuple[int, ...]): Every slice's size.
            buffer (Optional[torch.Tensor]): Defaults to None.
        """

        assert len(first_column_stacked) == len(second_column_stacked) == len(output_slices)

        start_index = 0
        y = torch.empty_like(x)
        for i, output_slice in enumerate(output_slices):

            road_triton_internal_mixed(
                y[..., start_index:start_index+output_slice],
                x[..., start_index:start_index+output_slice],
                self._token_lora_indices,
                first_column_stacked[i],
                second_column_stacked[i],
                self.group_size,
            )

            start_index += output_slice

        return y

