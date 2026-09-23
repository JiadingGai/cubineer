"""
VeOmni Mixture of Experts (MoE) Grouped GEMM - KernelBench Triton Variant

This is the Triton baseline implementation that serves as the reference for optimization.
Your task is to write CUDA/cuBLAS code that beats this Triton baseline.

Baseline Performance:
    - gpt-oss-120B FFN gate_up_proj (fused SwiGLU): K=hidden=2880 -> N=2*intermediate=5760
    - Configuration: 128 experts, top-4 routing, hidden=2880, intermediate=2880
    - Realistic top-4 routed token imbalance (NOT uniform 4096/expert)

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

    Baseline throughput is measured at runtime (gate_up shape: K=2880, N=5760,
    128 experts, imbalanced rows); it is NOT the square-task 575.5 TFLOPS figure.

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
    Triton Grouped GEMM wrapper (VeOmni group_gemm_same_nk).

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

    # Baseline tiling configuration (tuned for the square task; runtime-measured here)
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
    VeOmni MoE Grouped GEMM (gate_up_proj) - Triton Baseline

    This is the reference Triton implementation (VeOmni group_gemm_same_nk).
    Your task is to write CUDA/CuTe code that beats this baseline; the baseline
    throughput is measured at runtime for the gate_up shape (K=2880, N=5760).

    Mathematical Operation:
        For each expert i:
            C[tokens_i] = A[tokens_i] @ B[i]
        where tokens_i are the tokens assigned to expert i
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
        Triton baseline implementation (VeOmni group_gemm_same_nk).

        Args:
            tokens: Input activations [total_tokens, K]
            cumsum_M: Cumulative token counts [num_experts]

        Returns:
            Output tensor [total_tokens, N]
        """
        # Imbalanced (top-k routed) token counts: max_M is the TRUE largest per-expert
        # count, derived from cumsum deltas — NOT tokens//num_experts (uniform assumption).
        per_expert = torch.diff(cumsum_M, prepend=cumsum_M.new_zeros(1))
        max_M = int(per_expert.max().item())
        return group_gemm_triton(tokens, self.expert_weights, cumsum_M, max_M, transpose_b=False)


# ============================================================================
# Problem Configuration
# ============================================================================

NUM_EXPERTS = 128
HIDDEN_DIM = 2880            # K for gate_up_proj
INTERMEDIATE_DIM = 2880
GATEUP_N = 2 * INTERMEDIATE_DIM   # N = 5760 (fused gate+up for SwiGLU), gpt-oss-120B shape
TOP_K = 4                    # gpt-oss-120B routes top-4 experts/token
NUM_TOKENS = 16384           # routed tokens (batch*seq); total expert-rows = NUM_TOKENS*TOP_K
TOTAL_TOKENS = NUM_TOKENS * TOP_K  # 65,536 expert-token rows dispatched across 128 experts


def get_inputs():
    """
    Generate test inputs for the gpt-oss-120B MoE gate_up_proj group-GEMM.

    How cumsum_M (per-expert token counts) is simulated, and why
    -----------------------------------------------------------------
    A forward pass routes a FIXED number of token-rows N = NUM_TOKENS * TOP_K
    across E experts; each row's expert is drawn from per-expert routing
    probabilities p. The per-expert COUNT vector is therefore exactly
    Multinomial(N, p) -- the definition of counting N i.i.d. categorical draws,
    and (unlike a Poisson model) it conserves the token budget (sum == N).
    cumsum_M is its cumulative sum.

    p is drawn from a Dirichlet(alpha=0.24) prior, fitted to MEASURED gpt-oss-120B
    routing (see below): p ~ Dir(alpha * 1_E), then counts ~ Multinomial(N, p).
    The small alpha=0.24 reproduces the real heavy tail -- top expert ~8-11% of
    tokens, a long warm shoulder, a few cold/empty experts -- giving a per-layer
    coefficient of variation CV (std/mean of counts) ~1.8-2.1, matching the
    measured per-(sample,layer) CV. A uniform/Gaussian router (alpha -> inf) would
    give CV ~0.01 (far too flat); the pooled/long-run regime (CV ~0.28) corresponds
    to a large alpha (~tens). Drawing p fresh per layer matches the observation
    that the hot-expert SET shifts across layers.

    Measured provenance (gpt-oss-120B, 30 LongBench-v2 samples x 36 MoE layers,
    recomputing each layer's top-K from router weights, counting tokens/expert):
      - per-(sample,layer) CV = 2.13   <- what the grouped GEMM actually sees
      - pooled CV             = 0.277  (balance only emerges after pooling layers)
      - uniform baseline CV   = 0.011  (= sqrt((E-1)/R))
      - max/mean (pooled)     = 1.82 ; top-1 / top-10 share = 1.43% / 12.5%
    Generator fit by L1 to the empirical sorted-share curve: Dirichlet-Multinomial
    (alpha~0.24, L1=0.129) beats logistic-normal softmax(sigma*z) (sigma~1.4,
    L1=0.147, over-peaks the hottest expert). This is inference-time routing; the
    per-input-skewed vs pooled-balanced split and the Dirichlet shape are the
    transferable characterization for training-shaped workloads too.

    Returns:
        List of [tokens, cumsum_M]
    """
    torch.manual_seed(0)
    # p ~ Dirichlet(0.24 * 1_E) via Gamma normalization (fitted to measured routing)
    g = torch.distributions.Gamma(torch.full((NUM_EXPERTS,), 0.24), 1.0).sample()
    p = g / g.sum()
    # Per-expert counts ~ Multinomial(N, p): each of N rows picks an expert ~ p.
    assign = torch.multinomial(p, TOTAL_TOKENS, replacement=True)
    counts = torch.bincount(assign, minlength=NUM_EXPERTS).to(torch.int64).cuda()
    cumsum_M = torch.cumsum(counts, dim=0)            # [E], last == TOTAL_TOKENS
    tokens = torch.randn(int(cumsum_M[-1].item()), HIDDEN_DIM, dtype=torch.float16, device='cuda')
    return [tokens, cumsum_M]


def get_init_inputs():
    """
    Initialization inputs for the model.

    Returns:
        List of [num_experts, hidden_dim, intermediate_dim]
    """
    return [NUM_EXPERTS, HIDDEN_DIM, GATEUP_N]
