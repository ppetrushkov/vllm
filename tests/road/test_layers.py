# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import random
from copy import deepcopy
from dataclasses import dataclass
from typing import Optional
from unittest.mock import patch

import pytest
import torch
import torch.nn.functional as F

from vllm.config import LoRAConfig
# yapf conflicts with isort for this block
# yapf: disable
from vllm.lora.layers import LoRAMapping
from vllm.road.config import RoadVariant
from vllm.road.layers import (BaseLayerWithRoad, ColumnParallelLinearWithRoad,
                              MergedColumnParallelLinearWithRoad,
                              MergedQKVParallelLinearWithRoad,
                              QKVParallelLinearWithRoad,
                              RowParallelLinearWithRoad,
                              )
# yapf: enable
from vllm.road.models import RoadLayerWeights, PackedRoadLayerWeights
from vllm.road.road_wrapper import get_road_wrapper
from vllm.model_executor.layers.linear import (ColumnParallelLinear,
                                               MergedColumnParallelLinear,
                                               QKVParallelLinear,
                                               RowParallelLinear)
from vllm.model_executor.utils import set_random_seed
from vllm.platforms import current_platform

from .utils import DummyRoadManager, _apply_road

TOLERANCES = {
    torch.float16: (5e-3, 2.2),
    torch.float32: (5e-3, 5e-3),
    torch.bfloat16: (3e-2, 2e-2),
}

pytestmark = pytest.mark.skipif(
    not (current_platform.is_cuda_alike() or current_platform.is_cpu()),
    reason="Backend not supported")

DEVICES = ([
    f"cuda:{i}" for i in range(1 if torch.cuda.device_count() == 1 else 2)
] if current_platform.is_cuda_alike() else ["cpu"])

# prefill stage(True) or decode stage(False)
STAGES = [True, False]

NUM_RANDOM_SEEDS = 6

VOCAB_PARALLEL_EMBEDDING_TEST_NUM_RANDOM_SEEDS = 128


#@pytest.fixture(autouse=True)
#def clean_cache_reset_device(reset_default_device):
#    # Release any memory we might be holding on to. CI runs OOMs otherwise.
#    from vllm.lora.ops.triton_ops.utils import (_LORA_A_PTR_DICT,
#                                                _LORA_B_PTR_DICT)
#    _LORA_B_PTR_DICT.clear()
#    _LORA_A_PTR_DICT.clear()
#
#    yield


@pytest.fixture(autouse=True)
def skip_cuda_with_stage_false(request):
    """
    On cuda-like platforms, we use the same kernels for prefill and decode 
    stage, and 'stage' is generally ignored, so we only need to test once.
    """
    if current_platform.is_cuda_alike():
        try:
            if hasattr(request.node, "callspec") and hasattr(
                    request.node.callspec, "params"):
                params = request.node.callspec.params
                if "stage" in params and params["stage"] is False:
                    pytest.skip("Skip test when stage=False")
        except Exception:
            pass
    yield


def get_random_id_to_index(num_loras: int,
                           num_slots: int,
                           log: bool = True) -> list[Optional[int]]:
    """Creates a random lora_id_to_index mapping.

    Args:
        num_loras: The number of active loras in the mapping.
        num_slots: The number of slots in the mapping. Must be larger
            than num_loras.
        log: Whether to log the output.
    """

    if num_loras > num_slots:
        raise ValueError(
            f"num_loras is higher than num_slots: {num_loras} > {num_slots}. "
            "num_loras must be less than or equal to num_slots.")

    slots: list[Optional[int]] = [None] * num_slots
    random_slot_selections = (torch.randperm(num_slots)[:num_loras]).tolist()
    for lora_id, slot_idx in enumerate(random_slot_selections, start=1):
        slots[slot_idx] = lora_id

    if log:
        print(f"Created lora_id_to_index mapping: {slots}.")

    return slots


def populate_adapters(
    id_to_index: list[Optional[int]],
    layer: BaseLayerWithRoad,
    layer_weights: torch.Tensor,
    repeats: int = 1,
    variant: RoadVariant = RoadVariant.ROAD_1,
    group_size: int = 2,
) -> tuple[dict[int, RoadLayerWeights], dict[int, list[RoadLayerWeights]]]:
    """This method populates the lora layers with lora weights.

    Args:
        id_to_index: a list of lora ids. The index of the lora id
            represents which memory slot the lora matrices are
            stored in. A None value indicates a free slot.
        layer: the LoRAlayer to populate.
        layer_weights: the PyTorch tensor containing the layer's
            weights.
        generate_embeddings_tensor: whether to generate an
            embeddings tensor for each LoRA.
        repeats: must only be set for column parallel packed
            layers. Indicates the number of loras to compose
            together to create a single lora layer.
    """

    # Dictionary that maps the lora ID to the
    # corresponding lora weights.
    lora_dict: dict[int, RoadLayerWeights] = dict()

    # Dictionary that maps the lora ID to the
    # corresponding subloras.
    sublora_dict: dict[int, list[RoadLayerWeights]] = dict()

    for slot_idx, lora_id in enumerate(id_to_index):
        if lora_id is not None:
            subloras: list[RoadLayerWeights] = []
            for i in range(repeats):
                sublora = DummyRoadManager(
                    layer_weights.device).init_random_lora(
                        module_name=f"fake_{i}",
                        output_size=layer_weights.shape[0] // repeats,
                        dtype=layer_weights.dtype,
                        variant=variant,
                    )
                sublora.optimize()
                subloras.append(sublora)

            lora = PackedRoadLayerWeights.pack(
                subloras) if repeats > 1 else subloras[0]

            layer.set_weights(
                slot_idx,
                theta=lora.theta,
                alpha=lora.alpha,
                variant=variant,
                group_size=group_size,
            )

            lora_dict[lora_id] = lora
            sublora_dict[lora_id] = subloras

    return lora_dict, sublora_dict


def create_random_inputs(
    active_lora_ids: list[int],
    num_inputs: int,
    input_size: tuple[int, ...],
    input_range: tuple[float, float],
    input_type: torch.dtype = torch.int,
    device: torch.device = "cuda"
) -> tuple[list[torch.Tensor], list[int], list[int]]:
    """Creates random inputs.

    Args:
        active_lora_ids: lora IDs of active lora weights.
        num_inputs: the number of inputs to create.
        input_size: the size of each individual input.
        input_range: the range of values to include in the input.
            input_range[0] <= possible input values < input_range[1]
        input_type: the type of values in the input.
    """

    low, high = input_range

    inputs: list[torch.Tensor] = []
    index_mapping: list[int] = []
    prompt_mapping: list[int] = []

    for _ in range(num_inputs):
        if input_type == torch.int:
            inputs.append(
                torch.randint(low=int(low),
                              high=int(high),
                              size=input_size,
                              device=device))
        else:
            inputs.append(
                torch.rand(size=input_size, dtype=input_type, device=device) *
                high + low)

        lora_id = random.choice(active_lora_ids)
        index_mapping += [lora_id] * input_size[0]
        prompt_mapping += [lora_id]

    return inputs, index_mapping, prompt_mapping


def check_road_wrapper(road_wrapper) -> bool:
    if current_platform.is_cuda_alike():
        from vllm.road.road_gpu import RoadWrapperGPU

        return type(road_wrapper) is RoadWrapperGPU
    #elif current_platform.is_cpu():
    #    from vllm.lora.punica_wrapper.punica_cpu import PunicaWrapperCPU

    #    return type(punica_wrapper) is PunicaWrapperCPU
    else:
        return False


@torch.inference_mode()
@pytest.mark.parametrize("num_loras", [1, 2, 4, 8])
@pytest.mark.parametrize("orientation", ["row", "column"])
@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("stage", STAGES)
@pytest.mark.parametrize("variant", [RoadVariant.ROAD_1, RoadVariant.ROAD_2, RoadVariant.ROAD_4])
@pytest.mark.parametrize("group_size", [32, 64])
def test_linear_parallel(dist_init, num_loras, orientation,
                         device, stage, variant, group_size) -> None:

    if current_platform.is_cuda_alike():
        torch.cuda.set_device(device)

    max_loras = 8
    torch.set_default_device(device)
    road_wrapper = get_road_wrapper(8192, 256, device, max_adapters=max_loras, group_size=group_size)
    assert check_road_wrapper(road_wrapper)
    lora_config = LoRAConfig(max_loras=max_loras,
                             lora_dtype=torch.float16,
                             road_group_size=group_size)

    def create_random_linear_parallel_layer():
        if orientation == "row":
            linear = RowParallelLinear(4096,
                                       4096,
                                       bias=False,
                                       params_dtype=torch.float16)
            linear.weight.data = torch.rand_like(linear.weight.data)
            lora_linear = RowParallelLinearWithRoad(linear) 
        else:
            linear = ColumnParallelLinear(4096,
                                          4096,
                                          bias=False,
                                          params_dtype=torch.float16)
            linear.weight.data = torch.rand_like(linear.weight.data)
            lora_linear = ColumnParallelLinearWithRoad(linear)
        lora_linear.create_weights(max_loras, lora_config)
        assert (lora_linear.n_slices == len(lora_linear.first_column_stacked) == len(
            lora_linear.second_column_stacked) == 1)
        return linear, lora_linear

    for i in range(NUM_RANDOM_SEEDS):
        set_random_seed(i)

        id_to_index = get_random_id_to_index(num_loras, max_loras)
        linear, lora_linear = create_random_linear_parallel_layer()
        assert torch.equal(linear.weight, lora_linear.weight)
        lora_linear.set_mapping(road_wrapper)
        lora_dict, _ = populate_adapters(
            id_to_index,
            layer=lora_linear,
            layer_weights=linear.weight,
            variant=variant,
            group_size=group_size,
        )

        inputs, index_mapping, prompt_mapping = create_random_inputs(
            active_lora_ids=list(lora_dict.keys()),
            num_inputs=32 * num_loras,
            input_size=(1, 4096),
            input_range=(0, 1),
            input_type=torch.float16,
            device=device)
        lora_mapping = LoRAMapping(index_mapping,
                                   prompt_mapping,
                                   is_prefill=stage)
        road_wrapper.update_metadata(
            lora_mapping,
            id_to_index,
            max_loras,
            512,
        )

        lora_result = lora_linear(torch.cat(inputs))[0]

        expected_results: list[torch.Tensor] = []
        for input_, lora_id in zip(inputs, prompt_mapping):
            lora = lora_dict[lora_id]
            result = linear(input_)[0]
            result = _apply_road(lora.theta, lora.alpha, variant, group_size, result)
            expected_results.append(result)
        expected_result = torch.cat(expected_results)

        rtol, atol = TOLERANCES[lora_result.dtype]
        torch.testing.assert_close(lora_result,
                                   expected_result,
                                   rtol=rtol,
                                   atol=atol)

        # Check that resetting the lora weights succeeds

        for slot_idx in range(max_loras):
            lora_linear.reset_weights(slot_idx)

        inputs, index_mapping, prompt_mapping = create_random_inputs(
            active_lora_ids=[0],
            num_inputs=32 * num_loras,
            input_size=(1, 4096),
            input_range=(0, 1),
            input_type=torch.float16,
            device=device)
        lora_mapping = LoRAMapping(index_mapping,
                                   prompt_mapping,
                                   is_prefill=stage)

        road_wrapper.update_metadata(lora_mapping, id_to_index, max_loras,
                                       512)

        lora_result = lora_linear(torch.cat(inputs))[0]
        expected_result = linear(torch.cat(inputs))[0]

        rtol, atol = TOLERANCES[lora_result.dtype]
        torch.testing.assert_close(lora_result,
                                   expected_result,
                                   rtol=rtol,
                                   atol=atol)


@torch.inference_mode()
@pytest.mark.parametrize("num_loras", [1, 2, 4, 8])
@pytest.mark.parametrize("repeats", [1, 2, 3])
@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("stage", STAGES)
@pytest.mark.parametrize("variant", [RoadVariant.ROAD_1, RoadVariant.ROAD_2, RoadVariant.ROAD_4])
@pytest.mark.parametrize("group_size", [32, 64])
def test_column_parallel_packed(dist_init, num_loras, repeats,
                                device, stage, variant, group_size) -> None:

    if current_platform.is_cuda_alike():
        torch.cuda.set_device(device)

    max_loras = 8
    torch.set_default_device(device)
    road_wrapper = get_road_wrapper(8192, 256, device, max_adapters=max_loras, group_size=group_size)
    assert check_road_wrapper(road_wrapper)
    lora_config = LoRAConfig(max_loras=max_loras,
                             max_lora_rank=8,
                             lora_dtype=torch.float16,
                             road_group_size=group_size)

    def create_column_parallel_packed_layer():
        if repeats == 2:
            linear = MergedColumnParallelLinear(4096, [4096] * repeats,
                                                bias=False,
                                                params_dtype=torch.float16)
            linear.weight.data = torch.rand_like(linear.weight.data)
            lora_linear = MergedColumnParallelLinearWithRoad(linear)
        elif repeats == 3:
            linear = QKVParallelLinear(4096,
                                       64,
                                       32,
                                       bias=False,
                                       params_dtype=torch.float16)
            linear.weight.data = torch.rand_like(linear.weight.data)
            lora_linear = MergedQKVParallelLinearWithRoad(linear)
        else:
            linear = QKVParallelLinear(4096,
                                       64,
                                       32,
                                       bias=False,
                                       params_dtype=torch.float16)
            linear.weight.data = torch.rand_like(linear.weight.data)
            lora_linear = QKVParallelLinearWithRoad(linear)

        @dataclass
        class FakeConfig:
            hidden_size = 4096
            num_key_value_heads = 32
            num_attention_heads = 32

        n_slices = repeats
        lora_linear.create_weights(max_loras,
                                   lora_config,
                                   model_config=FakeConfig())
        assert (lora_linear.n_slices == len(lora_linear.first_column_stacked) == len(
            lora_linear.second_column_stacked) == n_slices)
        return linear, lora_linear

    for i in range(NUM_RANDOM_SEEDS):
        set_random_seed(i)

        id_to_index = get_random_id_to_index(num_loras, max_loras)

        linear, lora_linear = create_column_parallel_packed_layer()
        assert torch.equal(linear.weight, lora_linear.weight)
        lora_linear.set_mapping(road_wrapper)
        lora_dict, sublora_dict = populate_adapters(
            id_to_index,
            layer=lora_linear,
            layer_weights=linear.weight,
            repeats=repeats,
            variant=variant,
            group_size=group_size,
        )

        inputs, index_mapping, prompt_mapping = create_random_inputs(
            active_lora_ids=list(lora_dict.keys()),
            num_inputs=32 * num_loras,
            input_size=(1, 4096),
            input_range=(0, 1),
            input_type=torch.float16,
            device=device)
        lora_mapping = LoRAMapping(index_mapping,
                                   prompt_mapping,
                                   is_prefill=stage)

        road_wrapper.update_metadata(
            lora_mapping,
            id_to_index,
            max_loras,
            512,
        )

        lora_result = lora_linear(torch.cat(inputs))[0]

        expected_results: list[torch.Tensor] = []
        for input_, lora_id in zip(inputs, prompt_mapping):
            result = linear(input_)[0]
            subloras = sublora_dict[lora_id]
            for i, sublora in enumerate(subloras):
                sublora_shape = result.shape[1] // repeats
                result[:, i*sublora_shape:(i+1)*sublora_shape] = _apply_road(
                    sublora.theta,
                    sublora.alpha,
                    variant,
                    group_size,
                    result[:, i*sublora_shape:(i+1)*sublora_shape]
                )
            expected_results.append(result)
        expected_result = torch.cat(expected_results)

        rtol, atol = TOLERANCES[lora_result.dtype]
        torch.testing.assert_close(lora_result,
                                   expected_result,
                                   rtol=rtol,
                                   atol=atol)

        for slot_idx in range(max_loras):
            lora_linear.reset_weights(slot_idx)

        inputs, index_mapping, prompt_mapping = create_random_inputs(
            active_lora_ids=[0],
            num_inputs=32 * num_loras,
            input_size=(1, 4096),
            input_range=(0, 1),
            input_type=torch.float16,
            device=device)
        lora_mapping = LoRAMapping(index_mapping,
                                   prompt_mapping,
                                   is_prefill=stage)

        road_wrapper.update_metadata(
            lora_mapping,
            id_to_index,
            max_loras,
            512,
        )

        lora_result = lora_linear(torch.cat(inputs))[0]
        expected_result = linear(torch.cat(inputs))[0]

        rtol, atol = TOLERANCES[lora_result.dtype]
        torch.testing.assert_close(lora_result,
                                   expected_result,
                                   rtol=rtol,
                                   atol=atol)


