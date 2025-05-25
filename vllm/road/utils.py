# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os
from typing import Optional, Union

import huggingface_hub
import regex as re
from huggingface_hub.utils import (EntryNotFoundError, HfHubHTTPError,
                                   HFValidationError, RepositoryNotFoundError)
from torch import nn
from transformers import PretrainedConfig

from vllm.config import LoRAConfig
from vllm.logger import init_logger
from vllm.lora.fully_sharded_layers import (
    ColumnParallelLinearWithShardedLoRA,
    MergedColumnParallelLinearWithShardedLoRA,
    MergedQKVParallelLinearWithShardedLoRA, QKVParallelLinearWithShardedLoRA,
    RowParallelLinearWithShardedLoRA)
# being imported for _all_lora_classes below
# yapf conflicts with isort for this block
# yapf: disable
from vllm.road.layers import (BaseLayerWithRoad, ColumnParallelLinearWithRoad,
                              MergedColumnParallelLinearWithRoad,
                              RowParallelLinearWithRoad,
                              QKVParallelLinearWithRoad,
                              MergedQKVParallelLinearWithRoad)
from vllm.model_executor.layers.linear import LinearBase
# yapf: enable
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead
from vllm.model_executor.models.utils import WeightsMapper

logger = init_logger(__name__)

_all_lora_classes: set[type[BaseLayerWithRoad]] = {
    #VocabParallelEmbeddingWithLoRA,
    ColumnParallelLinearWithRoad,
    MergedColumnParallelLinearWithRoad,
    QKVParallelLinearWithRoad,
    MergedQKVParallelLinearWithRoad,
    RowParallelLinearWithRoad,
    #ReplicatedLinearWithLoRA,
    #LogitsProcessorWithLoRA,
    #ColumnParallelLinearWithShardedLoRA,
    #QKVParallelLinearWithShardedLoRA,
    #MergedColumnParallelLinearWithShardedLoRA,
    #MergedQKVParallelLinearWithShardedLoRA,
    #RowParallelLinearWithShardedLoRA,
    #LinearScalingRotaryEmbeddingWithLoRA,
}


def from_layer(layer: nn.Module,
               max_adapters: int,
               lora_config: LoRAConfig,
               packed_modules_list: list,
               model_config: Optional[PretrainedConfig] = None) -> nn.Module:
    for lora_cls in _all_lora_classes:
        # specifying kwargs so they can be easily accessed in decorator
        if lora_cls.can_replace_layer(source_layer=layer,
                                      lora_config=lora_config,
                                      packed_modules_list=packed_modules_list,
                                      model_config=model_config):
            instance_layer = lora_cls(layer)
            instance_layer.create_weights(max_adapters, lora_config,
                                          model_config)
            return instance_layer
    return layer
