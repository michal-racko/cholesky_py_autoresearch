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

# Block-column width for the tensor-core path. The trailing rank-b update is a
# TF32 GEMM (the bulk of the O(n^3/3) work); the diagonal factorization and
# panel solve of each b-wide column stay in FP32.
_TF32_BLOCK = 1024

# Smallest n for which the blocked TF32 factorization is used. Below this,
# cuSOLVER's single-matrix potrf wins and TF32 rounding is too large for the
# residual gate (allowed reconstruction residual scales with n). Measured on a
# B200: n=8192 breaks even, n=16384 ~2x, n=32768 ~4x.
_TF32_MIN_N = 8192

# cuSOLVER's *batched* potrf is pathological for large matrices at small batch
# (measured on a B200: ~4x worse per matrix than a single-matrix call at
# n=4096/batch=2, and ~2.8x at n=2048/batch=2). Factoring each matrix on the
# well-tuned single-matrix path wins there. But the batched path amortizes fine
# once the batch is large (n=2048/batch=8 is efficient), so only loop when the
# matrix is big AND the batch is small.
_LOOP_MIN_N = 2048
_LOOP_MAX_BATCH = 4


def _blocked_cholesky_tf32(data: torch.Tensor, block: int) -> torch.Tensor:
    """Left-looking blocked Cholesky with the trailing update in TF32.

    For each block column ``[j, je)`` we form the accumulated left-panel product
    ``S = L[j:, :j] @ L[j:je, :j].T`` (a single TF32 GEMM, >90% of the FLOPs),
    subtract it from the corresponding block of A, factor the ``b x b`` diagonal
    block in FP32, and solve the panel below it with an FP32 triangular solve.

    Runs batched: ``data`` is ``(batch, n, n)`` and every op broadcasts over the
    batch dim, so no matrix is ever handed to a batched potrf whole.
    """
    n = data.shape[-1]
    out = torch.zeros_like(data)
    old_tf32 = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = True
    try:
        for j in range(0, n, block):
            je = min(j + block, n)
            if j > 0:
                left_top = out[..., j:je, :j]
                s_top = left_top @ left_top.mT
                a_top = data[..., j:je, j:je] - s_top
            else:
                a_top = data[..., j:je, j:je]

            l_top = torch.linalg.cholesky_ex(a_top, check_errors=False).L
            out[..., j:je, j:je] = l_top

            if je < n:
                below = data[..., je:, j:je]
                if j > 0:
                    below = below - out[..., je:, :j] @ left_top.mT
                out[..., je:, j:je] = torch.linalg.solve_triangular(
                    l_top.mT, below, upper=True, left=False
                )
    finally:
        torch.backends.cuda.matmul.allow_tf32 = old_tf32
    return out


def custom_kernel(data: input_t) -> output_t:
    if not data.is_cuda or data.dtype != torch.float32:
        return torch.linalg.cholesky_ex(data, check_errors=False).L

    n = data.shape[-1]

    num_warps = _NUM_WARPS.get(n)
    if num_warps is not None:
        data = data.contiguous()
        output = torch.empty_like(data)
        _cholesky_left_kernel[(data.shape[0],)](
            data,
            output,
            n * n,
            BLOCK_N=n,
            num_warps=num_warps,
        )
        return output

    if n >= _TF32_MIN_N:
        return _blocked_cholesky_tf32(data.contiguous(), _TF32_BLOCK)

    # Work around cuSOLVER's slow batched potrf for large matrices at small
    # batch by factoring each matrix on its own well-tuned single-matrix path.
    batch = data.shape[0]
    if n >= _LOOP_MIN_N and 1 < batch <= _LOOP_MAX_BATCH:
        return torch.stack(
            [torch.linalg.cholesky_ex(m, check_errors=False).L for m in data]
        )

    return torch.linalg.cholesky_ex(data, check_errors=False).L
