# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# pylint: disable=unused-argument
import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional, Union, cast

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PretrainedConfig

from vllm.adapter_commons.layers import AdapterMapping
from vllm.config import LoRAConfig
from vllm.distributed import (get_tensor_model_parallel_rank,
                              get_tensor_model_parallel_world_size,
                              split_tensor_along_last_dim,
                              tensor_model_parallel_all_gather,
                              tensor_model_parallel_all_reduce)
from vllm.distributed.utils import divide
from vllm.lora.layers import _get_lora_device
# yapf: disable
from vllm.model_executor.layers.linear import (ColumnParallelLinear,
                                               LinearBase,
                                               MergedColumnParallelLinear,
                                               QKVParallelLinear,
                                               ReplicatedLinear,
                                               RowParallelLinear)
# yapf: enable
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.rotary_embedding import (
    LinearScalingRotaryEmbedding, RotaryEmbedding)
from vllm.model_executor.layers.vocab_parallel_embedding import (
    VocabParallelEmbedding)
from vllm.platforms import current_platform
from vllm.road.config import RoadVariant

if TYPE_CHECKING:
    from vllm.road.road_wrapper import RoadWrapperBase



def _not_fully_sharded_can_replace(can_replace):
    """
    decorator which adds the condition of not using fully sharded loras
    intended to wrap can_replace_layer()
    """

    def dec(*args, **kwargs):
        decorate = kwargs.pop("decorate") if "decorate" in kwargs else True
        condition = (not kwargs["lora_config"].fully_sharded_loras
                     if decorate else True)
        return can_replace(*args, **kwargs) and condition

    return dec


#@dataclass
#class RoadMapping(AdapterMapping):
#    is_prefill: bool = False


class BaseLayerWithRoad(nn.Module):

    def slice_theta(
        self, theta: Union[torch.Tensor, list[Union[torch.Tensor, None]]],
        variant: RoadVariant,
    ) -> Union[torch.Tensor, list[Union[torch.Tensor, None]]]:
        """Slice theta if splitting for tensor parallelism."""
        ...

    def slice_alpha(
        self, alpha: Union[torch.Tensor, list[Union[torch.Tensor, None]]],
        variant: RoadVariant,
    ) -> Union[torch.Tensor, list[Union[torch.Tensor, None]]]:
        """Slice alpha if splitting with tensor parallelism."""
        ...

    def create_weights(
        self,
        max_adapters: int,
        lora_config: LoRAConfig,
        model_config: Optional[PretrainedConfig] = None,
    ) -> None:
        """Initializes lora matrices."""
        ...

    def reset_weights(self, index: int):
        """Resets the lora weights at index back to 0."""
        ...

    def set_weights(
        self,
        index: int,
        theta: torch.Tensor,
        alpha: torch.Tensor,
        variant: RoadVariant,
        group_size: int,
    ):
        """Overwrites lora tensors at index."""
        ...

    def set_mapping(
        self,
        road_wrapper,
    ):
        self.road_wrapper: RoadWrapperBase = road_wrapper

    @classmethod
    def can_replace_layer(
        cls,
        source_layer: nn.Module,
        lora_config: LoRAConfig,
        packed_modules_list: list,
        model_config: Optional[PretrainedConfig],
    ) -> bool:
        """Returns True if the layer can be replaced by this LoRA layer."""
        raise NotImplementedError


class BaseLinearLayerWithRoad(BaseLayerWithRoad):

    def __init__(self, base_layer: LinearBase):
        super().__init__()
        self.base_layer = base_layer
        self.input_size = self.base_layer.input_size
        self.device = _get_lora_device(self.base_layer)

        self.output_slices: tuple[int, ...]
        self.tp_size: int
        self.output_size: int
        self.n_slices: int

    def create_weights(
        self,
        max_adapters: int,
        lora_config: LoRAConfig,
        model_config: Optional[PretrainedConfig] = None,
    ) -> None:
        self.lora_config = lora_config

        self.first_column_stacked = tuple(
            torch.zeros(
                max_adapters,
                self.output_size,
                dtype=lora_config.lora_dtype,
                device=self.device,
            ) for _ in range(self.n_slices))
        self.second_column_stacked = tuple(
            torch.zeros(
                max_adapters,
                self.output_size,
                dtype=lora_config.lora_dtype,
                device=self.device,
            ) for _ in range(self.n_slices))
        self.output_slices = (self.first_column_stacked[0].shape[1], )

    def reset_weights(self, index: int):
        for s_index in range(self.n_slices):
            self.first_column_stacked[s_index][index] = 1
            self.second_column_stacked[s_index][index] = 0

    def set_weights(
        self,
        index: int,
        theta: torch.Tensor,
        alpha: torch.Tensor,
        variant: RoadVariant,
        group_size: int,
    ):
        # Except for QKVParallelLinearWithLoRA and
        # MergedColumnParallelLinearWithLoRA, all other linear LoRA layers
        # store weights in a tuple of size 1. These two layers will
        # override this function.
        assert (len(self.first_column_stacked) == len(self.second_column_stacked) ==
                self.n_slices == 1)

        self.reset_weights(index)
        if self.tp_size > 1:
            theta = self.slice_theta(theta, variant)
            alpha = self.slice_alpha(alpha, variant)


        first_col, second_col = prepare_weights(theta, alpha, variant, group_size)

        self.first_column_stacked[0][index,
                               :first_col.shape[0]].copy_(
                                   first_col, non_blocking=True)
        self.second_column_stacked[0][index,
                               :second_col.shape[0]].copy_(
                                   second_col, non_blocking=True)

    def apply(self,
              x: torch.Tensor,
              bias: Optional[torch.Tensor] = None) -> torch.Tensor:
        output = self.base_layer.quant_method.apply(self.base_layer, x, bias)

        # In transformers backend, x and output have extra batch dimension like
        # (1, seq_len, hidden_dim), while punica expects (seq_len, hidden_dim),
        # therefore we need to flatten the batch dimensions.
        if x.ndim == 3 and output.ndim == 3:
            output = output.flatten(0, 1)
            x = x.flatten(0, 1)

        output = self.road_wrapper.add_linear(
                output, self.first_column_stacked, self.second_column_stacked,
                self.output_slices)

        return output

    @property
    def weight(self) -> torch.Tensor:

        # unquantizedLinear
        if hasattr(self.base_layer, "weight"):
            return self.base_layer.weight
        # Compressed Tensor
        elif hasattr(self.base_layer, "weight_packed"):
            return self.base_layer.weight_packed
        # GPTQ/AWQ
        elif hasattr(self.base_layer, "qweight"):
            return self.base_layer.qweight
        # marlin
        elif hasattr(self.base_layer, "B"):
            return self.base_layer.B
        # HQQ marlin
        elif hasattr(self.base_layer, "W_q"):
            return self.base_layer.W_q
        else:
            raise ValueError(f"Unsupported base layer: {self.base_layer}")

    @property
    def bias(self) -> Optional[torch.Tensor]:
        if hasattr(self.base_layer, "bias"):
            return self.base_layer.bias
        else:
            return None

class ColumnParallelLinearWithRoad(BaseLinearLayerWithRoad):
    """
    LoRA on top of ColumnParallelLinear layer.
    LoRA B is sliced for tensor parallelism.
    There are two types for the `base_layer`:
    1. ColumnParallelLinear, e.g.`dense_h_to_4h` in `FalconForCausalLM`.
    2. MergedColumnParallelLinear, e.g.`gate_up_proj` in `Phi3ForCausalLM`.
    """

    def __init__(self, base_layer: ColumnParallelLinear) -> None:
        super().__init__(base_layer)
        # The base_layer type is ColumnParallelLinear or
        # MergedColumnParallelLinear, their weight sharding logic is
        # inconsistent when TP is greater than 1.
        self.is_merged_col_linear = type(
            base_layer) is MergedColumnParallelLinear
        self.tp_size = get_tensor_model_parallel_world_size()
        self.output_size = self.base_layer.output_size_per_partition
        # There is only one LoRA layer
        self.n_slices = 1

    def slice_theta(self, theta: torch.Tensor, variant: RoadVariant) -> torch.Tensor:
        tensor_model_parallel_rank = get_tensor_model_parallel_rank()
        shard_size = get_tp_output_size_for_variant(self.output_size, variant)
        start_idx = tensor_model_parallel_rank * shard_size
        end_idx = (tensor_model_parallel_rank + 1) * shard_size
        theta = theta[start_idx:end_idx]
        return theta

    def slice_alpha(self, alpha: torch.Tensor, variant: RoadVariant) -> torch.Tensor:
        # Applicable to cases where the base_layer is
        # MergedColumnParallelLinear.
        if self.is_merged_col_linear:
            pass
            #tp_rank = get_tensor_model_parallel_rank()
            #shard_size = self.output_size // 2
            #offset = lora_b.shape[-1] // 2

            #left_weight = lora_b[:, tp_rank * shard_size:(tp_rank + 1) *
            #                     shard_size]
            #right_weight = lora_b[:, offset + tp_rank * shard_size:offset +
            #                      (tp_rank + 1) * shard_size]
            #lora_b = torch.cat([left_weight, right_weight], dim=1)
        # Applicable to cases where the base_layer is
        # ColumnParallelLinear.
        else:
            tensor_model_parallel_rank = get_tensor_model_parallel_rank()
            shard_size = get_tp_output_size_for_variant(self.output_size, variant)
            start_idx = tensor_model_parallel_rank * shard_size
            end_idx = (tensor_model_parallel_rank + 1) * shard_size
            alpha = alpha[start_idx:end_idx]
        return alpha

    def forward(
        self, input_: torch.Tensor
    ) -> Union[torch.Tensor, tuple[torch.Tensor, Optional[torch.Tensor]]]:
        """Forward of ColumnParallelLinear

        Args:
            input_: Tensor whose last dimension is `input_size`.

        Returns:
            - output
            - bias
        """
        bias = (self.base_layer.bias
                if not self.base_layer.skip_bias_add else None)

        # Matrix multiply.
        output_parallel = self.apply(input_, bias)
        if self.base_layer.gather_output:
            # All-gather across the partitions.
            output = tensor_model_parallel_all_gather(output_parallel)
        else:
            output = output_parallel

        if not self.base_layer.return_bias:
            return output

        output_bias = (self.base_layer.bias
                       if self.base_layer.skip_bias_add else None)
        return output, output_bias

    @classmethod
    @_not_fully_sharded_can_replace
    def can_replace_layer(
        cls,
        source_layer: nn.Module,
        lora_config: LoRAConfig,
        packed_modules_list: list,
        model_config: Optional[PretrainedConfig],
    ) -> bool:
        return type(source_layer) is ColumnParallelLinear or (
            type(source_layer) is MergedColumnParallelLinear
            and len(packed_modules_list) == 1)

class MergedColumnParallelLinearWithRoad(ColumnParallelLinearWithRoad):
    """ColumnParallelLinear layer that is composed of 2 sublayers (slices)
    packed together (eg. gate_proj + up_proj -> gate_up_proj).

    This means we have 2 LoRAs, each applied to one half of the layer.

    Both slices must have the same size.
    """

    def __init__(
        self, base_layer: Union[MergedColumnParallelLinear,
                                QKVParallelLinear]) -> None:
        super().__init__(base_layer)
        # There are two LoRA layers
        self.tp_size = get_tensor_model_parallel_world_size()
        self.tp_rank = get_tensor_model_parallel_rank()
        # the output_sizes in MergedColumnParallelLinear is not sharded by tp
        # we need to divide it by the tp_size to get correct slices size
        output_sizes = self.base_layer.output_sizes
        self.output_slices = tuple(
            divide(output_size, self.tp_size) for output_size in output_sizes)
        self.n_slices = len(self.output_slices)
        self.output_ids = (self.tp_rank, ) * self.n_slices

    def create_weights(
        self,
        max_adapters: int,
        lora_config: LoRAConfig,
        model_config: Optional[PretrainedConfig] = None,
    ) -> None:
        """
        The main reason for overriding this function is to enhance  code 
        maintainability.
        """
        self.lora_config = lora_config

        self.first_column_stacked = tuple(
            torch.zeros(
                max_adapters,
                output_size,
                dtype=lora_config.lora_dtype,
                device=self.device,
            ) for output_size in self.output_slices)
        self.second_column_stacked = tuple(
            torch.zeros(
                max_adapters,
                output_size,
                dtype=lora_config.lora_dtype,
                device=self.device,
            ) for output_size in self.output_slices)

    def slice_theta(
        self, theta: list[Union[torch.Tensor, None]],
        variant: RoadVariant,
    ) -> list[Union[torch.Tensor, None]]:
        for i, (shard_id, shard_size) in enumerate(
                zip(self.output_ids, self.output_slices)):
            shard_size = get_tp_output_size_for_variant(shard_size, variant)
            if (theta_i := theta[i]) is not None:
                theta[i] = theta_i[shard_size * shard_id:shard_size *
                                     (shard_id + 1)]
        return theta

    def slice_alpha(
        self, alpha: list[Union[torch.Tensor, None]],
        variant: RoadVariant,
    ) -> list[Union[torch.Tensor, None]]:
        for i, (shard_id, shard_size) in enumerate(
                zip(self.output_ids, self.output_slices)):
            shard_size = get_tp_output_size_for_variant(shard_size, variant)
            if (alpha_i := alpha[i]) is not None:
                alpha[i] = alpha_i[shard_size * shard_id:shard_size *
                                     (shard_id + 1)]
        return alpha

    def set_weights(
        self,
        index: int,
        theta: torch.Tensor,
        alpha: torch.Tensor,
        variant: RoadVariant,
        group_size: int,
    ):
        self.reset_weights(index)

        if self.tp_size > 1:
            theta = self.slice_theta(theta, variant)
            alpha = self.slice_alpha(alpha, variant)


        for i in range(self.n_slices):
            theta_i = theta[i]
            alpha_i = alpha[i]
            assert theta_i is not None
            assert alpha_i is not None

            # variant 1
            first_col, second_col = prepare_weights(theta_i, alpha_i, variant, group_size)

            self.first_column_stacked[i][
                index, :first_col.shape[0]].copy_(
                    first_col, non_blocking=True)
            self.second_column_stacked[i][
                index, :second_col.shape[0]].copy_(
                    second_col, non_blocking=True)


    @classmethod
    @_not_fully_sharded_can_replace
    def can_replace_layer(
        cls,
        source_layer: nn.Module,
        lora_config: LoRAConfig,
        packed_modules_list: list,
        model_config: Optional[PretrainedConfig],
    ) -> bool:
        return (type(source_layer) is MergedColumnParallelLinear
                and len(packed_modules_list) == 2)


class RowParallelLinearWithRoad(BaseLinearLayerWithRoad):

    def __init__(self, base_layer: RowParallelLinear) -> None:
        super().__init__(base_layer)

        self.tp_size = get_tensor_model_parallel_world_size()
        # reset input_size
        self.input_size = self.base_layer.input_size_per_partition
        self.output_size = self.base_layer.output_size

        self.tp_rank = get_tensor_model_parallel_rank()
        # There is only one LoRA layer.
        self.n_slices = 1

    def slice_theta(self, theta: torch.Tensor, variant: RoadVariant) -> torch.Tensor:
        #shard_size = self.output_size
        #start_idx = self.tp_rank * shard_size
        #end_idx = (self.tp_rank + 1) * shard_size
        #theta = theta[start_idx:end_idx]
        return theta

    def slice_alpha(self, alpha: torch.Tensor, variant: RoadVariant) -> torch.Tensor:
        #shard_size = self.output_size
        #start_idx = self.tp_rank * shard_size
        #end_idx = (self.tp_rank + 1) * shard_size
        #alpha = alpha[start_idx:end_idx]
        return alpha

    def forward(
        self, input_: torch.Tensor
    ) -> Union[torch.Tensor, tuple[torch.Tensor, Optional[torch.Tensor]]]:
        """Forward of RowParallelLinear

        Args:
            input_: tensor whose last dimension is `input_size`. If
                    `input_is_parallel` is set, then the last dimension
                    is `input_size // tp_size`.

        Returns:
            - output
            - bias
        """
        # set up backprop all-reduce.
        if self.base_layer.input_is_parallel:
            input_parallel = input_
        else:
            # TODO: simplify code below
            splitted_input = split_tensor_along_last_dim(
                input_, num_partitions=self.base_layer.tp_size)
            input_parallel = splitted_input[self.tp_rank].contiguous()

        # Matrix multiply.
        output_parallel = self.apply(input_parallel)
        if self.base_layer.reduce_results and self.base_layer.tp_size > 1:
            output_ = tensor_model_parallel_all_reduce(output_parallel)
        else:
            output_ = output_parallel

        if not self.base_layer.skip_bias_add:
            output = (output_ + self.base_layer.bias
                      if self.base_layer.bias is not None else output_)
            output_bias = None
        else:
            output = output_
            output_bias = self.base_layer.bias

        if not self.base_layer.return_bias:
            return output

        return output, output_bias

    @classmethod
    @_not_fully_sharded_can_replace
    def can_replace_layer(
        cls,
        source_layer: nn.Module,
        lora_config: LoRAConfig,
        packed_modules_list: list,
        model_config: Optional[PretrainedConfig],
    ) -> bool:
        return type(source_layer) is RowParallelLinear

class QKVParallelLinearWithRoad(ColumnParallelLinearWithRoad):
    """
    ColumnParallelLinear layer that is specifically designed for
    qkv_proj. Certain models, such as chatglm3 and baichuan-7b,
    only contains a single LoRA within their qkv_proj layer.

    During inference with Tensor Parallel, the weights of lora_b
    must be accurately partitioned according to the respective ranks.

    Q slice may have different shape than K and V slices (which both have
    the same shape).
    """

    def __init__(self, base_layer: QKVParallelLinear) -> None:
        super().__init__(base_layer)
        self.q_proj_total_size = (self.base_layer.total_num_heads *
                                  self.base_layer.head_size)
        self.q_proj_shard_size = (self.base_layer.num_heads *
                                  self.base_layer.head_size)
        self.kv_proj_shard_size = (self.base_layer.num_kv_heads *
                                   self.base_layer.head_size)
        self.kv_proj_total_size = (self.base_layer.total_num_kv_heads *
                                   self.base_layer.head_size)
        # There is only one LoRA layer
        self.n_slices = 1

    def slice_theta(self, theta: torch.Tensor, variant: RoadVariant) -> torch.Tensor:
        tp_rank = get_tensor_model_parallel_rank()
        q_shard_id = tp_rank
        kv_shard_id = tp_rank // self.base_layer.num_kv_head_replicas
        q_proj_shard_size = get_tp_output_size_for_variant(self.q_proj_shard_size, variant)
        theta_q = theta[q_proj_shard_size *
                          q_shard_id:q_proj_shard_size *
                          (q_shard_id + 1)]
        k_offset = self.q_proj_total_size // 2
        kv_proj_shard_size = get_tp_output_size_for_variant(self.kv_proj_shard_size, variant)
        theta_k = theta[:, k_offset +
                          kv_proj_shard_size * kv_shard_id:k_offset +
                          kv_proj_shard_size * (kv_shard_id + 1)]
        v_offset = k_offset + self.kv_proj_total_size // 2
        theta_v = theta[:, v_offset +
                          kv_proj_shard_size * kv_shard_id:v_offset +
                          kv_proj_shard_size * (kv_shard_id + 1)]
        theta = torch.cat([theta_q, theta_k, theta_v], dim=1)
        return theta

    def slice_alpha(self, alpha: torch.Tensor, variant: RoadVariant) -> torch.Tensor:
        tp_rank = get_tensor_model_parallel_rank()
        q_shard_id = tp_rank
        kv_shard_id = tp_rank // self.base_layer.num_kv_head_replicas
        q_proj_shard_size = get_tp_output_size_for_variant(self.q_proj_shard_size, variant)
        alpha_q = alpha[q_proj_shard_size *
                          q_shard_id:q_proj_shard_size *
                          (q_shard_id + 1)]
        k_offset = self.q_proj_total_size // 2
        kv_proj_shard_size = get_tp_output_size_for_variant(self.kv_proj_shard_size, variant)
        alpha_k = alpha[:, k_offset +
                          kv_proj_shard_size * kv_shard_id:k_offset +
                          kv_proj_shard_size * (kv_shard_id + 1)]
        v_offset = k_offset + self.kv_proj_total_size // 2
        alpha_v = alpha[:, v_offset +
                          kv_proj_shard_size * kv_shard_id:v_offset +
                          kv_proj_shard_size * (kv_shard_id + 1)]
        alpha = torch.cat([alpha_q, alpha_k, alpha_v], dim=1)
        return alpha

    @classmethod
    @_not_fully_sharded_can_replace
    def can_replace_layer(cls, source_layer: nn.Module,
                          lora_config: LoRAConfig, packed_modules_list: list,
                          model_config: Optional[PretrainedConfig]) -> bool:
        return type(source_layer) is QKVParallelLinear and len(
            packed_modules_list) == 1

class MergedQKVParallelLinearWithRoad(MergedColumnParallelLinearWithRoad):
    """MergedColumnParallelLinear layer that is composed of 3 sublayers (slices)
    packed together in qkv proj fashion
    (q_proj + k_proj + v_proj -> qkv_proj).

    This means we have 3 LoRAs, each applied to one slice of the layer.

    Q slice may have different shape than K and V slices (which both have
    the same shape).
    """

    def __init__(self, base_layer: QKVParallelLinear) -> None:
        super().__init__(base_layer)
        # There are three LoRA layer.
        self.n_slices = len(self.base_layer.output_sizes)
        self.tp_size = get_tensor_model_parallel_world_size()
        self.tp_rank = get_tensor_model_parallel_rank()

        self.q_proj_shard_size = (self.base_layer.num_heads *
                                  self.base_layer.head_size)
        self.kv_proj_shard_size = (self.base_layer.num_kv_heads *
                                   self.base_layer.head_size)
        self.q_shard_id = self.tp_rank
        self.kv_shard_id = self.tp_rank // self.base_layer.num_kv_head_replicas

        self.output_slices = (
            self.q_proj_shard_size,
            self.kv_proj_shard_size,
            self.kv_proj_shard_size,
        )
        self.output_ids = (
            self.q_shard_id,
            self.kv_shard_id,
            self.kv_shard_id,
        )

    def create_weights(
        self,
        max_adapters: int,
        lora_config: LoRAConfig,
        model_config: Optional[PretrainedConfig] = None,
    ) -> None:
        """
        The main reason for overloading this function is to handle inconsistent 
        weight dimensions in qkv lora.
        """
        super().create_weights(max_adapters, lora_config, model_config)

    @classmethod
    @_not_fully_sharded_can_replace
    def can_replace_layer(
        cls,
        source_layer: nn.Module,
        lora_config: LoRAConfig,
        packed_modules_list: list,
        model_config: Optional[PretrainedConfig],
    ) -> bool:
        return (type(source_layer) is QKVParallelLinear
                and len(packed_modules_list) == 3)

def prepare_weights(theta: torch.Tensor, alpha: torch.Tensor, variant: RoadVariant, group_size: int) -> tuple[torch.Tensor, torch.Tensor]:
    if variant == RoadVariant.ROAD_1:
        # In each group there are only group_size // 2 parameters that are reused
        theta = theta.reshape(-1, group_size // 2).repeat_interleave(2, dim=0).flatten()
        alpha = alpha.reshape(-1, group_size // 2).repeat_interleave(2, dim=0).flatten()

        theta_cos = theta.cos()
        theta_sin = theta.sin()

        first_col = alpha * theta_cos
        second_col = alpha * theta_sin
    elif variant == RoadVariant.ROAD_2:
        # Each group has exactly group_size parameters
        theta_cos = theta.cos()
        theta_sin = theta.sin()

        first_col = alpha * theta_cos
        second_col = alpha * theta_sin
    elif variant == RoadVariant.ROAD_4:
        # Each group has 2*group_size parameters, first half used for first column, second half for second column
        road_theta = theta.reshape(-1, 2, group_size)
        theta_cos = road_theta[:, 0, :].cos().flatten()
        theta_sin = road_theta[:, 1, :].sin().flatten()
        road_alpha = alpha.reshape(-1, 2, group_size)
        alpha_1 = road_alpha[:, 0, :].flatten()
        alpha_2 = road_alpha[:, 1, :].flatten()

        first_col = alpha_1 * theta_cos
        second_col = alpha_2 * theta_sin

    return first_col, second_col


def get_tp_output_size_for_variant(shard_size: int, variant: RoadVariant) -> int:
    if variant == RoadVariant.ROAD_1:
        return shard_size // 2
    elif variant == RoadVariant.ROAD_2:
        return shard_size
    elif variant == RoadVariant.ROAD_4:
        return shard_size * 2
