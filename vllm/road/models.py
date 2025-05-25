# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import math
import os
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Callable, Optional, Union

import regex as re
import safetensors.torch
import torch
from torch import nn

from vllm.adapter_commons.models import (AdapterLRUCache, AdapterModel,
                                         AdapterModelManager)
from vllm.adapter_commons.utils import (add_adapter, deactivate_adapter,
                                        get_adapter, list_adapters,
                                        remove_adapter, set_adapter_mapping)
from vllm.config import LoRAConfig
from vllm.logger import init_logger
from vllm.road.config import RoadVariant
from vllm.road.layers import BaseLayerWithRoad
from vllm.lora.layers import LoRAMapping
from vllm.road.weights import RoadLayerWeights, PackedRoadLayerWeights
from vllm.road.road_wrapper import get_road_wrapper
from vllm.road.utils import from_layer
from vllm.road.peft_helper import PEFTRoadHelper
from vllm.lora.utils import (get_supported_lora_modules,
                             is_regex_target_modules,
                             parse_fine_tuned_lora_name, replace_submodule)
from vllm.model_executor.model_loader.tensorizer import TensorizerConfig
from vllm.model_executor.models import SupportsLoRA, supports_multimodal
from vllm.model_executor.models.interfaces import is_pooling_model
from vllm.model_executor.models.module_mapping import MultiModelKeys
from vllm.model_executor.models.utils import PPMissingLayer, WeightsMapper
from vllm.model_executor.utils import get_packed_modules_mapping
from vllm.model_executor.layers.linear import LinearBase
from vllm.utils import is_pin_memory_available

logger = init_logger(__name__)

_GLOBAL_ADAPTER_ID = 0



def get_adapter_id():
    global _GLOBAL_ADAPTER_ID
    _GLOBAL_ADAPTER_ID += 1
    return _GLOBAL_ADAPTER_ID


class RoadModel(AdapterModel):
    """A LoRA fine-tuned model."""

    def __init__(
        self,
        adapter_id: int,
        variant: RoadVariant,
        group_size: int,
        weights: dict[str, RoadLayerWeights],
    ) -> None:
        """
        Args:
            lora_model_id: The integer id for the lora model.
            rank: lora rank.
            loras: module name -> weights for lora-replaced layers.
            scaling_factor: Scaling factor to support long context lora model.
                None if the lora is not tuned for long context support.
        """
        self.id = adapter_id
        # Scaling factor for long context lora model. None if it is not
        # fine tuned for the long context.
        assert (
            adapter_id
            > 0), f"a valid lora id should be greater than 0, got {self.id}"
        self.variant = variant
        self.group_size = group_size
        self.weights: dict[str, RoadLayerWeights] = weights

    def clone(self, road_model_id: int) -> "RoadModel":
        """Return a copy of the object with different ids.

        Will share the underlying tensors."""
        return self.__class__(
            road_model_id,
            variant=self.variant,
            group_size=self.group_size,
            weights=self.weights.copy(),
        )

    #@property
    #def extra_vocab_size(self) -> int:
    #    return max(lora.extra_vocab_size
    #               for lora in self.loras.values()) if self.loras else 0

    def get_weight(self, module_name: str) -> Optional[RoadLayerWeights]:
        """Get weights for a given module by name"""
        return self.weights.get(module_name, None)

    def check_adapter_name(self, adapter_name: str) -> bool:
        return adapter_name in self.weights

    # (yard1): TODO see if we can derive target_embedding_padding automatically
    @classmethod
    def from_tensors(
        cls,
        road_model_id: int,
        tensors: dict[str, torch.Tensor],
        peft_helper: PEFTRoadHelper,
        device: str = "cuda",
        dtype: Optional[torch.dtype] = None,
        #embeddings: Optional[dict[str, torch.Tensor]] = None,
        #target_embedding_padding: Optional[int] = None,
        #embedding_modules: Optional[dict[str, str]] = None,
        #embedding_padding_modules: Optional[list[str]] = None,
        weights_mapper: Optional[WeightsMapper] = None,
    ) -> "RoadModel":
        """Create a RoadModel from a dictionary of tensors."""
        pin_memory = str(device) == "cpu" and is_pin_memory_available()
        weights: dict[str, RoadLayerWeights] = {}
        for tensor_name, tensor in tensors.items():
            module_name, is_theta = parse_fine_tuned_road_name(
                tensor_name, weights_mapper)
            if module_name not in weights:
                weights[module_name] = RoadLayerWeights.from_config(
                    module_name, peft_helper)

            if is_theta:
                weights[module_name].theta = tensor.to(device=device,
                                                      dtype=dtype).t()
                if pin_memory:
                    weights[module_name].theta = weights[
                        module_name].theta.pin_memory()
            else:
                weights[module_name].alpha = tensor.to(device=device,
                                                      dtype=dtype).t()
                if pin_memory:
                    weights[module_name].alpha = weights[
                        module_name].alpha.pin_memory()

        for weight in weights.values():
            weight.optimize()

        return cls(road_model_id,
                   peft_helper.variant,
                   peft_helper.group_size,
                   weights)

    @classmethod
    def from_local_checkpoint(
            cls,
            road_dir: str,
            expected_road_modules: list[str],
            peft_helper: PEFTRoadHelper,
            *,
            road_model_id: Optional[int] = None,
            device: str = "cuda",
            dtype: Optional[torch.dtype] = None,
            #target_embedding_padding: Optional[int] = None,
            #embedding_modules: Optional[dict[str, str]] = None,
            #embedding_padding_modules: Optional[list[str]] = None,
            weights_mapper: Optional[WeightsMapper] = None,
            tensorizer_config_dict: Optional[dict] = None) -> "RoadModel":
        """Create a RoadModel from a local checkpoint.

        Args:
            lora_dir: The local path that has lora data.
            expected_lora_modules: Name of modules that are expected to be
                replaced by lora.
            peft_helper: Loaded lora configuration information.
            lora_model_id: LoRA model id. If not given, automatically set by
                a global counter.
            device: Device where the lora model is loaded.
            dtype: dtype of the lora model weights.

        Returns:
            Loaded LoRA Model.
        """
        lora_tensor_path = os.path.join(road_dir, "adapter_model.safetensors")
        lora_bin_file_path = os.path.join(road_dir, "adapter_model.bin")
        tensors: dict[str, torch.Tensor] = {}
        unexpected_modules: list[Union[list[str], str]] = []

        def check_unexpected_modules(modules: dict):
            for road_module in modules.keys():  # noqa
                module_name, _ = parse_fine_tuned_road_name(
                    road_module, weights_mapper)
                part_name = module_name.split(".")[-1]
                if part_name not in expected_road_modules:
                    unexpected_modules.append(module_name)
            if unexpected_modules:
                raise ValueError(
                    f"While loading {road_dir}, expected"
                    f" target modules in {expected_road_modules}"
                    f" but received {unexpected_modules}."
                    f" Please verify that the loaded LoRA module is correct")

        if tensorizer_config_dict:
            from tensorizer import TensorDeserializer

            tensorizer_config = TensorizerConfig(**tensorizer_config_dict)
            lora_tensor_path = os.path.join(tensorizer_config.tensorizer_dir,
                                            "adapter_model.tensors")
            tensorizer_args = tensorizer_config._construct_tensorizer_args()
            tensors = TensorDeserializer(lora_tensor_path,
                                         dtype=tensorizer_config.dtype,
                                         **tensorizer_args.deserializer_params)
            check_unexpected_modules(tensors)

        elif os.path.isfile(lora_tensor_path):
            # Find unexpected modules.
            # Use safetensor key as a source of truth to find expected modules.
            # in peft if you have target_modules A, B, C and C does not exist
            # in the model it won’t error and model will be trained with A, B
            # loraified. C won’t exist in the safetensor but it will exist in
            # the target_modules of the adapter_config.json.
            unexpected_modules = []
            with safetensors.safe_open(lora_tensor_path,
                                       framework="pt") as f:  # type: ignore
                # Load tensors if there are only expected modules.
                check_unexpected_modules(f)
                for module in f.keys():  # noqa
                    tensors[module] = f.get_tensor(module)
        elif os.path.isfile(lora_bin_file_path):
            # When a bin file is provided, we rely on config to find unexpected
            # modules.
            unexpected_modules = []
            target_modules = peft_helper.target_modules
            if not isinstance(target_modules, list):
                target_modules = [target_modules]
            for module in target_modules:
                # Compatible with more modules,
                # such as:layers.11.self_attn.k_proj
                part_name = module.split(".")[-1]
                if part_name not in expected_road_modules:
                    unexpected_modules.append(module)
            # loaded lora's target modules must be a subset of
            # expected_lora_modules. It is not reliable. See
            # https://github.com/vllm-project/vllm/pull/5909. But there's no
            # other better mechanism.
            if unexpected_modules and not is_regex_target_modules(
                    peft_helper.target_modules, expected_road_modules):
                raise ValueError(
                    f"While loading {road_dir}, expected"
                    f" target modules in {expected_road_modules}"
                    f" but received {unexpected_modules}."
                    f" Please verify that the loaded LoRA module is correct")
            tensors = torch.load(lora_bin_file_path,
                                 map_location=device,
                                 weights_only=True)
        else:
            raise ValueError(f"{road_dir} doesn't contain tensors")

        return cls.from_tensors(
            road_model_id=get_adapter_id()
            if road_model_id is None else road_model_id,
            tensors=tensors,
            peft_helper=peft_helper,
            device=device,
            dtype=dtype,
            weights_mapper=weights_mapper)


class RoadModelManager(AdapterModelManager):
    """A manager that manages multiple Road-fine-tuned models."""

    def __init__(
        self,
        model: SupportsLoRA,
        max_num_seqs: int,
        max_num_batched_tokens: int,
        vocab_size: int,
        lora_config: LoRAConfig,
        device: torch.device,
    ):
        """Create a LoRAModelManager and adapter for a given model.

        Args:
            model: the model to be adapted.
            max_num_seqs: the maximum number of sequences model can run in a
                single batch.
            max_num_batched_tokens: the maximum number of tokens model can run
                in a single batch.
            vocab_size: the vocab size of the model.
            lora_config: the LoRA configuration.
        """
        self.lora_config = lora_config
        self.device = device
        self.max_num_seqs = max_num_seqs
        assert self.capacity >= self.adapter_slots
        self.max_num_batched_tokens = math.ceil(max_num_batched_tokens / 8) * 8
        self.road_index_to_id: list[Optional[int]] = [None] * self.adapter_slots
        self.vocab_size = vocab_size
        #self.long_lora_context: Optional[LongContextLoRAContext] = None
        self.road_wrapper = get_road_wrapper(
            max_num_batched_tokens,
            max_batches=self.max_num_seqs,
            device=self.device,
            max_adapters=self.lora_config.max_loras,
            group_size=self.lora_config.road_group_size
        )
        # Scaling factor -> offset to the sin_cos_cache to it.
        # Used for long context lora.
        #self.scaling_factor_to_offset: dict[float, int] = {}
        super().__init__(model)

        self.supported_adapter_modules = get_supported_road_modules(self.model)
        assert self.supported_adapter_modules, "No supported Road modules found in"
        f" {self.model.__class__.__name__}."

        self.packed_modules_mapping = get_packed_modules_mapping(self.model)
        # Used to indicate whether the model is a multimodal model
        self.supports_mm: bool = (
            supports_multimodal(self.model)
            # In case the model only supports LoRA for
            # text modules (e.g. ChatGLM)
            and hasattr(self.model, "get_mm_mapping"))
        self.is_pooling_model = is_pooling_model(self.model)
        self.packed_modules: dict[str, list[str]] = {}
        self.modules: dict[str, BaseLayerWithRoad] = {}
        # Dict instead of a set for compatibility with LRUCache.
        self._last_mapping: Optional[RoadMapping] = None
        self._create_adapter_modules()
        self.model.road_manager = self
        self.adapter_type = 'Road'

    @property
    def capacity(self) -> int:
        return self.lora_config.max_cpu_loras

    @property
    def adapter_slots(self) -> int:
        return self.lora_config.max_loras

    def activate_adapter(
        self,
        road_id: int,
    ) -> bool:
        """Move LoRA into a GPU buffer to be used in the forward pass."""
        if road_id in self._active_adapters:
            return False
        first_free_slot = next(
            ((i, road_id) for i, road_id in enumerate(self.road_index_to_id)
             if road_id is None), None)
        if first_free_slot is None:
            raise ValueError("No free lora slots")
        index, _ = first_free_slot
        self._active_adapters[road_id] = None
        road_model = self._registered_adapters[road_id]
        logger.debug("Activating Road. int id: %d, slot index: %d",
                     road_model.id, index)
        self.road_index_to_id[index] = road_model.id
        for module_name, module in self.modules.items():
            module_road = self._get_road_layer_weights(road_model, module_name)
            if module_road:
                module_road.optimize()
                module.set_weights(index, module_road.theta, module_road.alpha,
                                   road_model.variant, road_model.group_size)
            else:
                module.reset_weights(index)
        return True

    def _deactivate_adapter(self, road_id: int):
        try:
            index = self.road_index_to_id.index(road_id)
            self.road_index_to_id[index] = None
        except ValueError:
            pass

    def _add_adapter(self, road: RoadModel):
        if road.group_size != self.lora_config.road_group_size:
            raise ValueError(
                f"Road group size {road.group_size} does not match "
                f"the expected group size {self.lora_config.road_group_size}."
            )
        self._create_merged_loras_inplace(road)
        self._registered_adapters[road.id] = road

    def pin_adapter(self, lora_id: int) -> bool:
        """Pin a LoRAModel in the manager cache."""
        raise NotImplementedError(
            "Pinning is not supported in LoRAModelManager. "
            "Use LRUCacheLoRAModelManager for pinning")  # type: ignore

    def _set_adapter_mapping(self, mapping: LoRAMapping) -> None:
        # update lora states
        self.road_wrapper.update_metadata(
            mapping,
            self.road_index_to_id,
            self.adapter_slots + 1,
            self.vocab_size,
            #self.lora_config.lora_extra_vocab_size,
            #self.long_lora_context,
        )

    def remove_all_adapters(self):
        """Remove all LoRAModels from the manager."""
        self._registered_adapters.clear()
        self.road_index_to_id = [None] * self.adapter_slots
        self._active_adapters.clear()

    def _create_adapter_modules(self):
        for module_name, module in self.model.named_modules(
                remove_duplicate=False):
            if isinstance(module, PPMissingLayer):
                continue
            if not self._match_target_modules(module_name):
                continue
            # A temporary approach for multimodal models to support LoRA
            # TODO: Remove this restriction
            if self._filter_unsupported_mm_module(module_name):
                logger.warning(
                    "Regarding multimodal models, vLLM currently only supports "
                    "adding LoRA to language model, %s will be ignored.",
                    module_name,
                )
                continue
            parts = module_name.split(".")[-1]
            packed_moduled_lst = self.packed_modules_mapping.get(parts, [])
            new_module = replace_submodule(
                self.model, module_name,
                from_layer(module, self.adapter_slots, self.lora_config,
                           packed_moduled_lst, self.model.config))

            # LinearScalingRotaryEmbeddingWithLoRA is used to handle
            # long context lora. Register relevant metadata.
            #if isinstance(new_module, LinearScalingRotaryEmbeddingWithLoRA):
            #    self.long_lora_context = LongContextLoRAContext(
            #        new_module.scaling_factors, new_module.rotary_dim)
            #    self.scaling_factor_to_offset = \
            #        new_module.scaling_factor_to_offset
            # (yard1): TODO make this more robust
            #if "lm_head" in module_name:
            #    logits_processor_module = self.model.get_submodule(
            #        "logits_processor")
            #    new_module = replace_submodule(
            #        self.model, "logits_processor",
            #        from_layer_logits_processor(logits_processor_module,
            #                                    module, self.lora_slots,
            #                                    self.lora_config,
            #                                    self.model.config))

            # In some models, especially multimodal ones, layers with the same
            # name may have different types, such as nn.Linear and
            # ReplicatedLinear. The nn.Linear layers cannot be replaced with
            # LoRA layers, leading to assertion error. The following check
            # aims to prevent this error
            if self.supports_mm and not isinstance(new_module,
                                                   BaseLayerWithRoad):
                continue
            self.register_module(module_name, new_module)
            self._register_packed_modules(module_name)
            # All lora layers share the same punica_wrapper based on reference.
            new_module.set_mapping(self.road_wrapper)

    def register_module(self, module_name: str, module: "BaseLayerWithRoad"):
        assert isinstance(module, BaseLayerWithRoad)
        self.modules[module_name] = module

    def create_dummy_adapter(
            self,
            lora_id: int) -> RoadModel:
        """Create zero-initialized LoRAModel for warmup."""
        # In a dummy model, it can have any variant, it doesn't matter
        model = RoadModel(lora_id, RoadVariant.ROAD_1, self.lora_config.road_group_size, {})
        for module_name, module in self.model.named_modules():
            if (not self._match_target_modules(module_name)
                    or not isinstance(module, BaseLayerWithRoad)
                    or self._filter_unsupported_mm_module(module_name)):
                continue
            parts = module_name.split(".")
            if module_name not in self.packed_modules:
                lora = RoadLayerWeights.create_dummy_lora_weights(
                    module_name,
                    module.first_column_stacked[0].shape[1],
                    module.first_column_stacked[0].dtype,
                    "cpu",
                )
                lora.optimize()
            else:
                parts = module_name.split(".")
                replacements = self.packed_modules_mapping[parts[-1]]
                subloras: list[Optional[RoadLayerWeights]] = []
                for i, r in enumerate(replacements):
                    lora = RoadLayerWeights.create_dummy_lora_weights(
                        module_name + "." + r,
                        module.first_column_stacked[i].shape[1],
                        module.first_column_stacked[i].dtype,
                        "cpu",
                    )
                    lora.optimize()
                    subloras.append(lora)
                lora = PackedRoadLayerWeights.pack(subloras)
            model.weights[module_name] = lora
        return model

    def _match_target_modules(self, module_name: str):
        return any(
            re.match(
                r".*\.{target_module}$".format(target_module=target_module),
                module_name) or target_module == module_name
            for target_module in self.supported_adapter_modules)

    def _filter_unsupported_mm_module(self, module_name: str) -> bool:
        """
        Regarding multimodal models, vLLM currently only supports adding LoRA to
        language model. LoRA for other modules, such as the vision tower, will
        be filtered out.
        """
        if self.supports_mm:
            module_mapping: MultiModelKeys = self.model.get_mm_mapping()
            prefix_lst = module_mapping.connector + module_mapping.tower_model
            return any(
                [module_name.startswith(prefix) for prefix in prefix_lst])
        return False

    def _register_packed_modules(self, module_full_name: str) -> None:
        parts = module_full_name.split(".")
        module_name = parts[-1]
        replacements = self.packed_modules_mapping.get(module_name, [])
        # When replacements is less than or equal to 1, it indicates that this
        # module is not a packed module.
        if len(replacements) <= 1:
            return
        prefix = ".".join(parts[:-1])
        self.packed_modules[module_full_name] = [
            prefix + "." + r if prefix else r for r in replacements
        ]

    def _create_merged_loras_inplace(self, road_model: RoadModel) -> None:
        for module_name, new_module_names in self.packed_modules.items():
            replacement_weights: list[Optional[RoadLayerWeights]] = []
            replaced_module: set[str] = set()
            has_replacement = False
            for r in new_module_names:
                weights = self._get_road_layer_weights(road_model, r)
                replacement_weights.append(weights)
                if weights:
                    has_replacement = True
                    replaced_module.add(r)
            if not has_replacement:
                continue
            for i in range(len(replacement_weights)):
                if replacement_weights[i]:
                    continue
                replacement_weights[i] = None
            # HACK Temporary solution for the pool model.
            if self.is_pooling_model and not road_model.check_adapter_name(
                    module_name):
                replaced_module_name = module_name.replace("model.", "")
                if road_model.check_adapter_name(module_name):
                    module_name = replaced_module_name
            road_model.weights[module_name] = PackedRoadLayerWeights.pack(
                replacement_weights)
            # Remove the modules that have been replaced.
            for module in replaced_module:
                road_model.weights.pop(module, None)

    def _get_road_layer_weights(
            self, road_model: RoadModel,
            module_name: str) -> Optional[RoadLayerWeights]:
        org_module_name = module_name
        if self.is_pooling_model and not road_model.check_adapter_name(
                module_name):
            # If it's a pool model, and the layer name is not found,
            # remove the prefix 'model.' and search again.
            module_name = module_name.replace("model.", "")
            if road_model.check_adapter_name(module_name):
                org_module_name = module_name
                logger.info_once(
                    "For the pool model, successfully loaded the LoRA weights "
                    "after removing the prefix 'model.'.")
        return road_model.get_weight(org_module_name)

    def deactivate_adapter(self, adapter_id: int) -> bool:
        return deactivate_adapter(adapter_id, self._active_adapters,
                                  self._deactivate_adapter)

    def add_adapter(self, adapter: RoadModel) -> bool:
        logger.debug(
            "Adding lora. Model id: %d, "
            "int id: %d, ", adapter.id, adapter.id)
        return add_adapter(adapter, self._registered_adapters, self.capacity,
                           self._add_adapter)

    def set_adapter_mapping(self, mapping: LoRAMapping) -> None:
        self._last_mapping = set_adapter_mapping(mapping, self._last_mapping,
                                                 self._set_adapter_mapping)

    def remove_adapter(self, adapter_id: int) -> bool:
        return remove_adapter(adapter_id, self._registered_adapters,
                              self.deactivate_adapter)

    def list_adapters(self) -> dict[int, Any]:
        return list_adapters(self._registered_adapters)

    def get_adapter(self, adapter_id: int) -> Optional[Any]:
        return get_adapter(adapter_id, self._registered_adapters)


class RoadLRUCache(AdapterLRUCache[RoadModel]):

    def __init__(self, capacity: int, deactivate_lora_fn: Callable[[int],
                                                                   bool]):
        super().__init__(capacity, deactivate_lora_fn)


class LRUCacheRoadModelManager(RoadModelManager):
    """A model manager that manages multiple LoRAs with LRU cache."""

    def __init__(self, model: nn.Module, max_num_seqs: int,
                 max_num_batched_tokens: int, vocab_size: int,
                 lora_config: LoRAConfig, device: torch.device):
        super().__init__(model, max_num_seqs, max_num_batched_tokens,
                         vocab_size, lora_config, device)
        self._registered_adapters: RoadLRUCache = RoadLRUCache(
            self.capacity, self.deactivate_adapter)
        self._active_adapters: RoadLRUCache = RoadLRUCache(
            self.adapter_slots, self._deactivate_adapter)

    def list_adapters(self) -> dict[int, RoadModel]:
        """List all registered LoRAModels."""
        return dict(self._registered_adapters.cache)

    def add_adapter(self, adapter: RoadModel) -> bool:
        """Add a LoRAModel to the manager."""
        logger.debug(
            "Adding lora. Model id: %d, "
            "int id: %d, ", adapter.id, adapter.id)
        if adapter.id not in self._registered_adapters:
            self._add_adapter(adapter)
            was_added = True
        else:
            # We always touch to update the LRU cache order
            self._registered_adapters.touch(adapter.id)
            was_added = False
        return was_added

    def activate_adapter(
        self,
        adapter_id: int,
    ) -> bool:
        if adapter_id not in self._active_adapters and len(
                self._active_adapters) >= self.adapter_slots:
            self._active_adapters.remove_oldest()
        result = super().activate_adapter(adapter_id)
        # We always touch to update the LRU cache order
        self._active_adapters.touch(adapter_id)
        return result

    def remove_oldest_adapter(self) -> bool:
        if len(self._registered_adapters) > 0:
            self._registered_adapters.remove_oldest()
            return True
        return False

    def pin_adapter(self, adapter_id: int) -> bool:
        """Pin a LoRAModel in the manager cache."""
        self._pin_lora_in_cpu_cache(adapter_id)
        self._pin_lora_in_gpu_cache(adapter_id)
        return True

    def _pin_adapter_in_cpu_cache(self, adapter_id: int):
        try:
            self._registered_adapters.pin(adapter_id)
        except ValueError as err:
            raise ValueError("Pinning failed. "
                             f"Adapter {adapter_id} is not registered.") from err

    def _pin_lora_in_gpu_cache(self, lora_id: int):
        if lora_id not in self._active_adapters:
            # move lora to gpu if not already active
            self.activate_adapter(lora_id)

        self._active_adapters.pin(lora_id)


def create_adapter_manager(
        model: nn.Module,
        max_num_seqs: int,
        max_num_batched_tokens: int,
        vocab_size: int,
        lora_config: LoRAConfig,
        device: torch.device,
        adapter_manager_cls: type[RoadModelManager] = RoadModelManager,
        **kwargs) -> RoadModelManager:
    """Create a LoRA adapter for a given model."""
    if not hasattr(model, "packed_modules_mapping"):
        raise ValueError(f"Model {type(model)} is not supported for LoRA.")
    adapter_manager = adapter_manager_cls(
        model=model,
        max_num_seqs=max_num_seqs,
        max_num_batched_tokens=max_num_batched_tokens,
        vocab_size=vocab_size,
        lora_config=lora_config,
        device=device,
        **kwargs)
    return adapter_manager

def parse_fine_tuned_road_name(
        name: str,
        weights_mapper: Optional[WeightsMapper] = None
) -> tuple[str, bool]:
    """Parse the name of lora weights.

    args:
        name: the name of the fine-tuned LoRA, e.g.
            base_model.model.dense1.weight
        weights_mapper: maps the name of weight, e.g.
            `model.` -> `language_model.model.`,
    return:
        tuple(module_name, is_lora_a):
            module_name: the name of the module, e.g. model.dense1,
            is_lora_a whether the tensor is lora_a or lora_b.
            is_bias whether the tensor is lora bias.
    """

    # LoRA weight qualified name usually starts with `base_model.model.`,
    # so we remove the prefix `base_model.model.` to make the following
    # mapping correctly.
    if name.startswith("base_model.model."):
        name = name.replace("base_model.model.", "")
        name = weights_mapper._map_name(name) if weights_mapper else name
        # recover the prefix `base_model.model.`
        name = "base_model.model." + name
    else:
        name = weights_mapper._map_name(name) if weights_mapper else name

    # In some situations, we may not start with `base_model.model.`.
    # If we don't (e.g., ibm-granite/granite-speech-3.3-8b),
    # we should keep the prefix intact.
    start_index = 2 if name.startswith("base_model.model.") else 0

    parts = name.split(".")
    if (parts[-1] == "road_theta" or parts[-1] == "road_alpha"):
        new_name = ".".join(parts[start_index:-1])
        return new_name, parts[-1] == "road_theta"

    raise ValueError(f"{name} is unsupported Road weight")

def get_supported_road_modules(model: nn.Module) -> list[str]:
    """
    In vLLM, all linear layers support LoRA.
    """
    supported_road_modules: set[str] = set()
    # step1: traverse the model to get all the linear subfixes.
    for name, module in model.named_modules():
        if isinstance(module, (LinearBase, )):
            supported_road_modules.add(name.split(".")[-1])
    return list(supported_road_modules)
