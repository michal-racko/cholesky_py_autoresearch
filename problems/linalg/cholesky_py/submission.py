import torch
import triton
import triton.language as tl

from task import input_t, output_t


@triton.jit
def _cholesky_left_kernel(
    input_ptr,
    output_ptr,
    matrix_stride: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """One program per matrix: an unblocked left-looking Cholesky that keeps the
    whole lower triangle resident and rebuilds one column per step.

    At step k, ``values`` already holds L[:, :k] in columns 0..k-1 and the
    untouched lower triangle of A elsewhere. We read row k and column k out of
    the tile with masked reductions, apply the standard scalar recurrences

        L[k, k] = sqrt(A[k, k] - sum_{j<k} L[k, j]^2)
        L[i, k] = (A[i, k] - sum_{j<k} L[i, j] * L[k, j]) / L[k, k]

    and write the freshly computed column back into the tile.

    This is only competitive when the whole tile is tiny: every one of the
    BLOCK_N sequential steps reduces over the full BLOCK_N x BLOCK_N tile, so
    the wasted (masked-off) work grows with n while parallelism stays flat.
    Benchmarked on a B200, it beats cuSOLVER's batched potrf at n=32 (~1.5x)
    but loses badly from n=64 up (1.7x at 64, 40x at 256), so dispatch is
    restricted to n=32 below.
    """
    matrix = tl.program_id(0)
    row_ids = tl.arange(0, BLOCK_N)
    col_ids = tl.arange(0, BLOCK_N)
    rows = row_ids[:, None]
    cols = col_ids[None, :]
    offsets = matrix * matrix_stride + rows * BLOCK_N + cols
    values = tl.where(rows >= cols, tl.load(input_ptr + offsets), 0.0)

    for k in range(BLOCK_N):
        row = tl.sum(tl.where(rows == k, values, 0.0), axis=0)
        diagonal = tl.sum(tl.where(col_ids == k, row, 0.0), axis=0)
        diagonal -= tl.sum(tl.where(col_ids < k, row * row, 0.0), axis=0)
        diagonal = tl.sqrt(tl.maximum(diagonal, 0.0))

        column = tl.sum(tl.where(cols == k, values, 0.0), axis=1)
        products = tl.where(cols < k, values * row[None, :], 0.0)
        column = (column - tl.sum(products, axis=1)) / diagonal
        values = tl.where((rows == k) & (cols == k), diagonal, values)
        values = tl.where((rows > k) & (cols == k), column[:, None], values)

    tl.store(output_ptr + offsets, values)


# Matrix sizes served by the fused per-matrix kernel, mapped to a warp count
# that keeps the tile reduction cheap. Only n=32 is a measured win over
# cuSOLVER's batched potrf on a B200; larger n regresses (see kernel docstring),
# so everything else falls back to torch.linalg.cholesky_ex.
_NUM_WARPS = {32: 1}


def custom_kernel(data: input_t) -> output_t:
    batch, n, _ = data.shape
    num_warps = _NUM_WARPS.get(n)
    if num_warps is None or not data.is_cuda or data.dtype != torch.float32:
        return torch.linalg.cholesky_ex(data, check_errors=False).L

    data = data.contiguous()
    output = torch.empty_like(data)
    _cholesky_left_kernel[(batch,)](
        data,
        output,
        n * n,
        BLOCK_N=n,
        num_warps=num_warps,
    )
    return output
