# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch
import triton
import triton.language as tl

from vllm.lora.ops.triton_ops.kernel_utils import do_shrink_kernel
from vllm.lora.ops.triton_ops.utils import _get_lora_a_ptr
from vllm.utils import direct_register_custom_op

@triton.jit
def road_kernel_mixed(y_ptr, x_ptr, first_col_ptr, second_col_ptr, indices_ptr,
                      input_row_stride, output_row_stride, col_stride,
                      n_rows, n_cols,
                      BLOCK_SIZE: tl.constexpr):
    """
    Generally the idea of RoAD is to split the input vector into many 2D vectors
    and rotate each 2D vector with its own 2D rotation matrix. For additional
    flexibility, each rotation matrix is multiplied by a trainable scale.

    It can be written with a single matrix R like this:

    ( α₀cosθ₀   0         ...      0       -α₀sinθ₀  0         0         0      )
    ( 0         α₁cosθ₁   0        ...     0         -α₁sinθ₁  ...       0      )
    ( 0         0         α₂cosθ₂  0       ...       0         -α₂sinθ₂  ...    )
    ( ...       ...       ...      ...     ...       ...       ...       ...    )
    ( α₀sinθ₀   0         ...      0       α₀cosθ₀   0         0         0      )
    ( 0         α₁sinθ₁   0        ...     0         α₁cosθ₁   ...       0      )
    ( 0         0         α₂sinθ₂  0       ...       0         α₂cosθ₂   ...    )
    ( ...       ...       ...      ...     ...       ...       ...       ...    )

    when applied to vector R @ x each pair of elements of x is transformed like this:
    ( α₀cosθ₀   -α₀sinθ₀ )      ( x₀ )
    ( α₀sinθ₀   α₀cosθ₀  )      ( xₙ )

    The scales and angles inside each rotation matrix may actually be different
    (when using variant 2 or 4).

    Note that instead of using two consecutive elements x₀ x₁ we pair elements from
    the first half and second half, which allows for more efficient inference implementation.

    This kernels expects scales and cosine/sine multiplication to be already precomputed
    and stored in two separate tensors, first_col and second_col. It will then compute
    equivalent to:

    ```python
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    rotate_half_x = torch.cat((-x2, x1), dim=-1)
    result = x * first_col + rotate_half_x * second_col
    ```

    It also supports multi-adapter capability, where each row of the input
    might require different rotation matrices, based on the indices provided.

    """

    # x: [n_rows, n_cols]
    # y: [n_rows, n_cols]
    # first_col: [max_cols, n_cols]
    # second_col: [max_cols, n_cols]
    # indices: [n_rows]

    row_start = tl.program_id(0)
    row_step = tl.num_programs(0)

    block_idx = tl.program_id(1)
    col_idx = block_idx * BLOCK_SIZE

    for row_idx in tl.range(row_start, n_rows, row_step):
        # Index of the correct column for this row (token)
        global_col_idx = tl.load(indices_ptr + row_idx)
        if global_col_idx < 0:
            # For rows that don't require adapter simply copy the values
            row_start_ptr = x_ptr + row_idx * input_row_stride
            y_row_start_ptr = y_ptr + row_idx * output_row_stride
            offsets = tl.arange(0, BLOCK_SIZE//2)

            col_idx_low = col_idx
            col_idx_high = BLOCK_SIZE//2 + col_idx

            col_offsets_low = col_idx_low + offsets
            col_offsets_high = col_idx_high + offsets
            x_ptrs_low = row_start_ptr + col_offsets_low
            x_ptrs_high = row_start_ptr + col_offsets_high
            row_low = tl.load(x_ptrs_low)
            row_high = tl.load(x_ptrs_high)

            y_ptrs_low = y_row_start_ptr + col_offsets_low
            y_ptrs_high = y_row_start_ptr + col_offsets_high
            tl.store(y_ptrs_low, row_low)
            tl.store(y_ptrs_high, row_high)
        else:
            global_col_offset = global_col_idx * col_stride

            row_start_ptr = x_ptr + row_idx * input_row_stride
            y_row_start_ptr = y_ptr + row_idx * output_row_stride
            offsets = tl.arange(0, BLOCK_SIZE//2)

            col_idx_low = col_idx
            col_idx_high = BLOCK_SIZE//2 + col_idx

            col_offsets_low = col_idx_low + offsets
            col_offsets_high = col_idx_high + offsets
            x_ptrs_low = row_start_ptr + col_offsets_low
            x_ptrs_high = row_start_ptr + col_offsets_high
            row_low = tl.load(x_ptrs_low)
            row_high = tl.load(x_ptrs_high)

            first_col_low = tl.load(first_col_ptr + global_col_offset + col_offsets_low)
            first_col_high = tl.load(first_col_ptr + global_col_offset + col_offsets_high)
            second_col_low = tl.load(second_col_ptr + global_col_offset + col_offsets_low)
            second_col_high = tl.load(second_col_ptr + global_col_offset + col_offsets_high)

            acc_low = first_col_low * row_low
            acc_low -= second_col_low * row_high
            acc_high = first_col_high * row_high
            acc_high += second_col_high * row_low

            y_ptrs_low = y_row_start_ptr + col_offsets_low
            y_ptrs_high = y_row_start_ptr + col_offsets_high
            tl.store(y_ptrs_low, acc_low)
            tl.store(y_ptrs_high, acc_high)

@torch.inference_mode()
def _road_triton_internal_mixed(
    y: torch.Tensor,
    x: torch.Tensor,
    indices: torch.Tensor,
    first_col: torch.Tensor,
    second_col: torch.Tensor,
    group_size: int,
) -> None:
    # Actual kernel
    n_rows, n_cols = x.shape
    BLOCK_SIZE = group_size
    assert (n_cols) % BLOCK_SIZE == 0, f"n_cols must be divisible by {BLOCK_SIZE}"
    n_blocks = n_cols // BLOCK_SIZE

    road_kernel_mixed[(n_rows, n_blocks, 1)](y, x, first_col, second_col, indices,
                                            x.stride(0), y.stride(0), first_col.stride(0),
                                            n_rows, n_cols,
                                            BLOCK_SIZE)

def _road_triton_internal_mixed_fake(
    y: torch.Tensor,
    x: torch.Tensor,
    indices: torch.Tensor,
    first_col: torch.Tensor,
    second_col: torch.Tensor,
    group_size: int,
) -> None:
    return

try:
    direct_register_custom_op(
        op_name="road_triton_internal_mixed",
        op_func=_road_triton_internal_mixed,
        mutates_args=["y"],
        fake_impl=_road_triton_internal_mixed_fake,
    )
    road_triton_internal_mixed = torch.ops.vllm.road_triton_internal_mixed

except AttributeError:
    road_triton_internal_mixed = _road_triton_internal_mixed

