# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

import vllm.road.triton_ops as triton_ops
from vllm.platforms import current_platform

from .utils import assert_close


@pytest.fixture(autouse=True)
def reset_device(reset_default_device):
    pass


def check_road(batches: int, num_adapters: int,
                             hidden_size: int,
                             dtype: torch.dtype,
                             device: str,
                             group_size: int):
    """
    Compare outputs of torch_ops.sgmv_expand and triton_ops.lora_expand
    kernels.
    """

    inputs_tensor = torch.rand(
        (batches, hidden_size),
        dtype=dtype,
    ).to(device)

    first_column = torch.rand(
        (num_adapters, hidden_size),
        dtype=dtype,
    ).to(device)
    second_column = torch.rand(
        (num_adapters, hidden_size),
        dtype=dtype,
    ).to(device)
    indices = torch.randint(0, num_adapters, (batches,)).to(device)

    # Setup output tensors
    out_tensor = inputs_tensor.clone()
    ref_out_tensor = inputs_tensor.clone()

    triton_ops.road_triton_internal_mixed(
        out_tensor,
        inputs_tensor,
        indices,
        first_column,
        second_column,
        group_size,
    )

    # Reference
    for i in range(batches):
        index = indices[i].item()

        x = inputs_tensor[i]
        x_grouped = x.reshape(-1, 2, group_size // 2)
        x1 = x_grouped[:, 0, :]
        x2 = x_grouped[:, 1, :]
        rotate_half_x = torch.stack((-x2, x1), dim=1).reshape(x.shape)
        ref_out_tensor[i, :] = x * first_column[index] + rotate_half_x * second_column[index]

    assert_close(out_tensor, ref_out_tensor)


# Tests
# We test the punica kernels along 2 verticals mainly.
# 1. Variations in hidden_dim size
# 2. Variations in all other parameters like (batch_size, max_rank, num_loras
#  etc.)

# We have collected the hidden_sizes included in the LoRA models
# currently supported by vLLM. It tests whether the corresponding Triton
# kernel can run normally when tensor parallelism is set to
# [1, 2, 4, 8, 16, 32, 64].
HIDDEN_SIZES = [
    128,
    256,
    512,
    896,
    1024,
    1152,
    1216,
    1280,
    1536,
    1664,
    2048,
    2240,
    2304,
    2368,
    2432,
    2560,
    2752,
    3072,
    3328,
    3456,
    3584,
    3712,
    4096,
    4480,
    4608,
    4736,
    4864,
    5120,
    5504,
    5632,
    5888,
    6144,
    6400,
    6848,
    6912,
    7168,
    7424,
    8192,
    8960,
    9216,
    9472,
    10240,
    11008,
    11264,
    13824,
    14336,
    14784,
    14848,
    15360,
    18944,
    22016,
    22528,
    24576,
    27392,
    27648,
    29568,
    29696,
    32000,
    32256,
    32512,
    32768,
    33024,
    36864,
    43264,
    49152,
    49408,
    60544,
    60672,
    64000,
    64256,
    102400,
    102656,
    128000,
    128256,
]
#The size of TP
divisibility = [1, 2, 8, 16, 64]

all_hidden_size = []
for div in divisibility:
    for hidden_size in HIDDEN_SIZES:
        if (hidden_size // div) % 32 == 0:
            all_hidden_size.append(hidden_size // div)

HIDDEN_SIZES = list(set(all_hidden_size))

# Test params that focuses on hidden_size variation.
hs_test_params = {
    "hidden_sizes": HIDDEN_SIZES,
    "batches": [4],
    "num_adapters": [4],
}

# General tests params that tests for variations in all dimensions
# except hidden_size.
test_params = {
    "hidden_sizes": [2048],
    "batches": [1, 4, 16, 32],
    "num_adapters": [1, 8, 32, 128],
    "group_size": [32, 64],
}

DTYPES = [torch.float16, torch.bfloat16]
DEVICES = [f"cuda:{0}"]
SEED = [0]


@pytest.mark.parametrize("batches", test_params['batches'])
@pytest.mark.parametrize("num_adapters", test_params['num_adapters'])
@pytest.mark.parametrize("hidden_size", test_params['hidden_sizes'])
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("seed", SEED)
@pytest.mark.parametrize("group_size", test_params['group_size'])
def test_kernels(
    batches: int,
    num_adapters: int,
    hidden_size: int,
    dtype: torch.dtype,
    device: str,
    seed: int,
    group_size: int,
):
    """
    Tests LoRA kernels.
    """
    torch.set_default_device(device)
    current_platform.seed_everything(seed)

    check_road(batches=batches,
               num_adapters=num_adapters,
               hidden_size=hidden_size,
               dtype=dtype,
               device=device,
               group_size=group_size)


@pytest.mark.parametrize("batches", hs_test_params['batches'])
@pytest.mark.parametrize("num_adapters", hs_test_params['num_adapters'])
@pytest.mark.parametrize("hidden_size", hs_test_params['hidden_sizes'])
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("seed", SEED)
def test_kernels_hidden_size(
    batches: int,
    num_adapters: int,
    hidden_size: int,
    dtype: torch.dtype,
    device: str,
    seed: int,
):
    """
    Tests SGMV and LoRA kernels.
    """
    torch.set_default_device(device)
    current_platform.seed_everything(seed)

    check_road(batches=batches,
               num_adapters=num_adapters,
               hidden_size=hidden_size,
               dtype=dtype,
               device=device,
               group_size=32)
