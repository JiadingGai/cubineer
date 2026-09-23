"""
VeOmni Mixture of Experts (MoE) Grouped GEMM - KernelBench Triton Variant

This is the Triton baseline implementation that serves as the reference for optimization.
Your task is to write CUDA/cuBLAS code that beats this Triton baseline.

Baseline Performance:
    - H100: 575.5 TFLOPS (58% efficiency)
    - Configuration: 128 experts, 524K tokens, 2880×2880 dimensions

Optimization Target:
    - H100: 650+ TFLOPS (66%+ efficiency)
    - Improvement: +13% over Triton baseline

Task Type: KernelBench Triton (Triton → CUDA optimization)
"""

import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def get_pid_mn(pid, M, N, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, GROUP_SIZE: tl.constexpr):
    """Get 2D thread block ID with swizzling for cache locality."""
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_in_group = GROUP_SIZE * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m
    return pid_m, pid_n


@triton.jit
def load_with_pred_1d(ptr, skip_boundary_check: tl.constexpr, mask: tl.tensor, other=0):
    """Load 1D tensor with optional boundary checking."""
    if not skip_boundary_check:
        return tl.load(ptr, mask, other=other)
    else:
        return tl.load(ptr)


@triton.jit
def store_with_pred_2d(
    ptr, value,
    skip_boundary_check_0: tl.constexpr,
    skip_boundary_check_1: tl.constexpr,
    mask_0: tl.tensor,
    mask_1: tl.tensor,
):
    """Store 2D tensor with optional boundary checking."""
    if not skip_boundary_check_0 and not skip_boundary_check_1:
        tl.store(ptr, value, mask_0 & mask_1)
    elif not skip_boundary_check_0 and skip_boundary_check_1:
        tl.store(ptr, value, mask_0)
    elif skip_boundary_check_0 and not skip_boundary_check_1:
        tl.store(ptr, value, mask_1)
    else:
        tl.store(ptr, value)


@triton.heuristics(
    values={
        "N_ALIGNED": lambda args: args["N"] % args["BLOCK_N"] == 0,
        "K_ALIGNED": lambda args: args["K"] % args["BLOCK_K"] == 0,
    }
)
@triton.jit
def group_gemm_kernel(
    a_ptr, b_ptr, c_ptr,
    cumsum_M, max_M, total_M,
    G: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    TRANSPOSE_A: tl.constexpr, TRANSPOSE_B: tl.constexpr,
    GROUP: tl.constexpr,
    N_ALIGNED: tl.constexpr, K_ALIGNED: tl.constexpr,
):
    """
    Triton Grouped GEMM kernel for MoE workloads.

    Performance: 575.5 TFLOPS on H100 (58% efficiency)

    Computes: C[m,n] = A[m,k] @ B[k,n] for multiple experts
    where each expert processes a subset of tokens.
    """
    # Get 2D block ID with swizzling
    m, n = get_pid_mn(tl.program_id(axis=0), max_M, N, BLOCK_M, BLOCK_N, GROUP)
    gid = tl.program_id(1).to(tl.uint64)

    # Get this expert's token range
    gtid_start = tl.load(cumsum_M + gid - 1, mask=gid > 0, other=0)
    gtid_end = tl.load(cumsum_M + gid)
    m_size = (gtid_end - gtid_start).to(tl.uint64)

    # Early exit if block out of range
    if m * BLOCK_M >= m_size:
        return

    # Offset pointers to expert's data
    a_ptr += gtid_start * K
    b_ptr += gid * K * N
    c_ptr += gtid_start * N

    # Compute block offsets
    offs_m = m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_am = offs_m % m_size.to(tl.int64)
    offs_bn = offs_n % N
    blk_k = tl.arange(0, BLOCK_K)

    # Compute strides
    stride_am, stride_ak = (K, 1) if not TRANSPOSE_A else (1, m_size)
    stride_bk, stride_bn = (N, 1) if not TRANSPOSE_B else (1, K)

    # Initialize pointers
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + blk_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (blk_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)
    c_ptrs = c_ptr + N * offs_m[:, None] + offs_n[None, :]

    # Initialize accumulator
    c = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Main GEMM loop
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        # Load tiles
        a = load_with_pred_1d(a_ptrs, K_ALIGNED, blk_k[None, :] < K - k * BLOCK_K, other=0)
        b = load_with_pred_1d(b_ptrs, K_ALIGNED, blk_k[:, None] < K - k * BLOCK_K, other=0)

        # Tensor Core matmul
        c = tl.dot(a, b, c)

        # Advance pointers
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    # Store result
    store_with_pred_2d(c_ptrs, c, False, N_ALIGNED, offs_m[:, None] < m_size, offs_n[None, :] < N)


def group_gemm_triton(a, b, cumsum_M, max_M, transpose_a=False, transpose_b=False):
    """
    Triton Grouped GEMM wrapper.

    Baseline Performance: 575.5 TFLOPS on H100

    Args:
        a: Input activations [total_tokens, K]
        b: Expert weights [num_experts, K, N]
        cumsum_M: Cumulative token counts [num_experts]
        max_M: Maximum tokens per expert
        transpose_a: Transpose A (not supported)
        transpose_b: Transpose B

    Returns:
        Output tensor [total_tokens, N]
    """
    if transpose_b:
        G, N, K = b.shape
    else:
        G, K, N = b.shape

    assert not transpose_a, "Transpose A not supported"
    assert a.dtype in [torch.bfloat16, torch.float16]
    assert a.is_contiguous() and b.is_contiguous()

    c = torch.empty((a.shape[0], N), dtype=a.dtype, device=a.device)

    # Baseline configuration: 575.5 TFLOPS
    BLOCK_M, BLOCK_N, BLOCK_K, GROUP = 128, 256, 64, 8

    grid = lambda meta: (
        triton.cdiv(max_M, meta["BLOCK_M"]) * triton.cdiv(N, meta["BLOCK_N"]),
        G,
    )

    group_gemm_kernel[grid](
        a, b, c, cumsum_M, max_M, a.shape[0],
        G, N, K, BLOCK_M, BLOCK_N, BLOCK_K,
        transpose_a, transpose_b, GROUP,
        num_warps=8, num_stages=3,
    )

    return c


class Model(nn.Module):
    """
    VeOmni MoE Grouped GEMM - Triton Baseline

    Performance: 575.5 TFLOPS on H100 (58% efficiency)

    This is the reference Triton implementation. Your task is to write
    CUDA/cuBLAS code that beats this baseline performance.

    Mathematical Operation:
        For each expert i:
            C[tokens_i] = A[tokens_i] @ B[i]
        where tokens_i are the tokens assigned to expert i

    Optimization Target:
        - Beat 575.5 TFLOPS baseline
        - Target: 650+ TFLOPS (66%+ H100 efficiency)
        - Use CUDA, cuBLAS, or other optimized implementations
    """
    def __init__(self, num_experts: int, hidden_dim: int, intermediate_dim: int):
        super().__init__()
        self.num_experts = num_experts
        self.hidden_dim = hidden_dim
        self.intermediate_dim = intermediate_dim
        self.expert_weights = nn.Parameter(
            torch.randn(num_experts, hidden_dim, intermediate_dim, dtype=torch.float16)
        )

    def forward(self, tokens: torch.Tensor, cumsum_M: torch.Tensor) -> torch.Tensor:
        """
        Triton baseline implementation (575.5 TFLOPS on H100).

        Args:
            tokens: Input activations [total_tokens, K]
            cumsum_M: Cumulative token counts [num_experts]

        Returns:
            Output tensor [total_tokens, N]
        """
        max_M = tokens.shape[0] // self.num_experts
        return group_gemm_triton(tokens, self.expert_weights, cumsum_M, max_M, transpose_b=False)


# ============================================================================
# Problem Configuration
# ============================================================================

NUM_EXPERTS = 128
TOKENS_PER_EXPERT = 4096
HIDDEN_DIM = 2880
INTERMEDIATE_DIM = 2880
TOTAL_TOKENS = NUM_EXPERTS * TOKENS_PER_EXPERT  # 524,288 tokens


def get_inputs():
    """
    Generate test inputs for the MoE kernel.

    Returns:
        List of [tokens, cumsum_M]
    """
    tokens = torch.randn(TOTAL_TOKENS, HIDDEN_DIM, dtype=torch.float16, device='cuda')
    cumsum_M = torch.arange(1, NUM_EXPERTS + 1, dtype=torch.int64, device='cuda') * TOKENS_PER_EXPERT
    return [tokens, cumsum_M]


def get_init_inputs():
    """
    Initialization inputs for the model.

    Returns:
        List of [num_experts, hidden_dim, intermediate_dim]
    """
    return [NUM_EXPERTS, HIDDEN_DIM, INTERMEDIATE_DIM]
