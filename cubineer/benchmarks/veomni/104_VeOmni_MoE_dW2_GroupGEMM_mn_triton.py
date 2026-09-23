"""
VeOmni MoE Weight-Gradient Grouped GEMM (dW2, FC2/down_proj) - KernelBench Triton Variant

This is the Triton baseline (VeOmni group_gemm_same_mn) that serves as the
reference for optimization. Your task is to write CUDA/CuTe code that beats it.

What this kernel computes (gpt-oss-120B MoE training backward, FC2 weight grad):
    For each expert e:
        dW2[e] = Hmid[e]^T @ GY[e]
    where Hmid[e] = packed GLU output rows for expert e ([K_e, D]),
          GY[e]   = grad of FC2 output for expert e ([K_e, H]).

Key structural difference vs the FORWARD group_gemm_same_nk (tasks 101/102):
    - same_nk: ragged dimension is M (output rows); output is [R, N].
    - same_mn: ragged dimension is K (the REDUCTION / contraction length per
      expert, segmented by cumsum_K); output is a FIXED [E, M, N] per expert.
    So token imbalance lands on the dot-product accumulation length, not the
    output size -- a different optimization problem (variable-length reduction,
    fixed-size output tiles).

Real gpt-oss-120B shapes (confirmed from production trace 16k/bs2/packed):
    - E = 128 experts, top_k = 4
    - R = total packed rows = T * top_k = 14442 * 4 = 57768
    - M = D = 2880 (intermediate), N = H = 2880 (hidden)
    - a = Hmid   [R, D]   = [57768, 2880]
    - b = GY     [R, H]   = [57768, 2880]
    - c = dW2    [E, D, H] = [128, 2880, 2880]

==========================================================================
Where this task derives from: the real VeOmni same_mn call sites
==========================================================================
group_gemm_same_mn is called in TWO places, both backward weight-gradient
GEMMs in verl/models/transformers/gpt_oss_moe.py. Signature:
    group_gemm_same_mn(a, b, c, cumsum_K, max_K, transpose_a=True, transpose_b=False)
The output c is PASSED IN (its [E, M, N] shape defines the problem); the
ragged axis is cumsum_K (the reduction length per expert).

dW2 (FC2 / down_proj weight gradient) -- THIS TASK:
    group_gemm_same_mn(a=Hmid, b=grad_fc2_output, c=dW2,
                       cumsum_K=cumsum_t, max_K=R, transpose_a=True)
    # math: dW2[e] = Hmid[e]^T @ GY[e]
      a = Hmid (packed GLU output)   [R, D]   = [57768, 2880]
      b = GY   (grad of FC2 output)  [R, H]   = [57768, 2880]
      c = dW2  (per-expert wgrad)    [E, D, H]= [128, 2880, 2880]
      cumsum_K (ragged K segments)   [E]=[128], cumsum_K[-1] = R = 57768
      max_K = R = 57768
    Per expert e: [D, K_e] @ [K_e, H] -> [D, H] = [2880, K_e] @ [K_e, 2880] -> [2880, 2880].

(The sibling call dW1 -- a=scatter_output(X), b=GZ, c=dW1[E,H,2D] -- is task 103.)

Provenance of the numbers (all confirmed from the production trace):
  - E=128, H=2880, 2D=5760, D=2880: trace weight tensors [128,2880,5760]
    (gate_up) and [128,2880,2880] (down), matching the code's [E,H,2D]/[E,D,H].
  - R = T * top_k = 14442 * 4 = 57768: trace shows tokens [14442,2880], routing
    [14442,4], and the packed GEMM tensor appearing as a [57768,*] row dim;
    in code R = scatter_index.numel() = T * K.
  - cumsum_t is the SAME imbalanced cumsum the forward uses
    (cumsum(expert_histogram(expert_index))) -- here consumed as cumsum_K.

Task Type: KernelBench Triton (Triton -> CUDA optimization)
"""

import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def group_gemm_same_mn_kernel(
    a_ptr, b_ptr, c_ptr,
    cumsum_K,
    G: tl.constexpr, M: tl.constexpr, N: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    GROUP: tl.constexpr,
):
    """
    VeOmni same_mn grouped GEMM: c[e] = a[seg_e]^T @ b[seg_e] per expert.

    Ragged reduction: expert e owns rows [cumsum_K[e-1] : cumsum_K[e]] of a/b,
    so the K (contraction) length k = gtid_end - gtid_start varies per expert.
    Output c[e] is a fixed [M, N] tile regardless of k.
    """
    pid = tl.program_id(axis=0)
    gid = tl.program_id(axis=1).to(tl.uint64)

    # 2D block id with swizzling for L2 locality
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_in_group = GROUP * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP
    group_size_m = min(num_pid_m - first_pid_m, GROUP)
    m = first_pid_m + (pid % group_size_m)
    n = (pid % num_pid_in_group) // group_size_m

    # This expert's row segment defines the reduction length k
    gtid_start = tl.load(cumsum_K + gid - 1, mask=gid > 0, other=0)
    gtid_end = tl.load(cumsum_K + gid)
    k = (gtid_end - gtid_start).to(tl.int64)

    # a is [R, M] row-major; for a^T we read with strides (1, M) -> [M, k] view
    a_block_ptr = tl.make_block_ptr(
        base=a_ptr + gtid_start * M, shape=(M, k), strides=(1, M),
        offsets=(m * BLOCK_M, 0), block_shape=(BLOCK_M, BLOCK_K), order=(0, 1),
    )
    # b is [R, N] row-major -> [k, N] view
    b_block_ptr = tl.make_block_ptr(
        base=b_ptr + gtid_start * N, shape=(k, N), strides=(N, 1),
        offsets=(0, n * BLOCK_N), block_shape=(BLOCK_K, BLOCK_N), order=(1, 0),
    )
    # c[e] is a fixed [M, N] tile at expert offset gid * M * N
    c_block_ptr = tl.make_block_ptr(
        base=c_ptr + gid * M * N, shape=(M, N), strides=(N, 1),
        offsets=(m * BLOCK_M, n * BLOCK_N), block_shape=(BLOCK_M, BLOCK_N), order=(1, 0),
    )

    out = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for _ in range(0, tl.cdiv(k, BLOCK_K)):
        a = tl.load(a_block_ptr, boundary_check=(0, 1), padding_option="zero")
        b = tl.load(b_block_ptr, boundary_check=(0, 1), padding_option="zero")
        out += tl.dot(a, b)
        a_block_ptr = tl.advance(a_block_ptr, (0, BLOCK_K))
        b_block_ptr = tl.advance(b_block_ptr, (BLOCK_K, 0))

    # Empty expert (k == 0) writes zeros, matching VeOmni semantics
    tl.store(c_block_ptr, out.to(c_ptr.dtype.element_ty), boundary_check=(0, 1))


def group_gemm_same_mn(a, b, cumsum_K, num_experts, M, N):
    """
    VeOmni weight-gradient grouped GEMM wrapper.

    Args:
        a: [R, M] packed activations (e.g. Hmid)
        b: [R, N] packed grads        (e.g. GY = grad_fc2_output)
        cumsum_K: [E] cumulative per-expert row counts (ragged K), last == R
        num_experts, M, N: output shape [E, M, N]

    Returns:
        c: [E, M, N] per-expert weight gradient
    """
    assert a.dtype in [torch.bfloat16, torch.float16]
    assert a.is_contiguous() and b.is_contiguous()
    assert len(cumsum_K) == num_experts

    c = torch.empty((num_experts, M, N), dtype=a.dtype, device=a.device)

    BLOCK_M, BLOCK_N, BLOCK_K, GROUP = 128, 128, 32, 8

    grid = lambda meta: (
        triton.cdiv(M, meta["BLOCK_M"]) * triton.cdiv(N, meta["BLOCK_N"]),
        num_experts,
    )
    group_gemm_same_mn_kernel[grid](
        a, b, c, cumsum_K,
        num_experts, M, N,
        BLOCK_M, BLOCK_N, BLOCK_K, GROUP,
        num_warps=8, num_stages=3,
    )
    return c


class Model(nn.Module):
    """
    VeOmni MoE FC2 weight-gradient grouped GEMM (dW2) - Triton baseline.

    This is the reference Triton implementation (VeOmni group_gemm_same_mn).
    Your task is to write CUDA/CuTe code that beats it; the baseline throughput
    is measured at runtime for the dW2 shape (M=2880, N=2880, ragged-K).

    Mathematical Operation:
        For each expert e:
            dW2[e] = Hmid[e]^T @ GY[e]
        where Hmid[e]/GY[e] are the rows of a/b owned by expert e (segmented by cumsum_K).
    """
    def __init__(self, num_experts: int, m_dim: int, n_dim: int):
        super().__init__()
        self.num_experts = num_experts
        self.m_dim = m_dim   # M = D (rows of dW2)
        self.n_dim = n_dim   # N = H (cols of dW2)

    def forward(self, a: torch.Tensor, b: torch.Tensor, cumsum_K: torch.Tensor) -> torch.Tensor:
        return group_gemm_same_mn(a, b, cumsum_K, self.num_experts, self.m_dim, self.n_dim)


# ============================================================================
# Problem Configuration  (real gpt-oss-120B FC2 backward, 16k/bs2/packed step)
# ============================================================================

NUM_EXPERTS = 128
HIDDEN_DIM = 2880               # N = H (cols of dW2)
INTERMEDIATE_DIM = 2880         # M = D (rows of dW2)
TOP_K = 4
NUM_TOKENS = 14442              # real packed token count (16k buffer, bs2)
TOTAL_ROWS = NUM_TOKENS * TOP_K  # R = 57768 packed expert-rows


def get_inputs():
    """
    Inputs for the dW2 weight-gradient grouped GEMM.

    cumsum_K simulation: R = NUM_TOKENS * TOP_K rows are routed across E experts,
    so per-expert row counts ~ Multinomial(R, p) -- the exact distribution for a
    fixed token budget (sum == R; Poisson would not conserve it). Here the
    imbalance lands on the K (reduction) axis, so hot experts get a longer
    accumulation while every expert writes a fixed [M, N] output tile.

    p ~ Dirichlet(alpha=0.24) fitted to MEASURED gpt-oss-120B routing (30
    LongBench-v2 samples x 36 MoE layers): per-(sample,layer) CV (std/mean of
    counts) = 2.13 -- heavy-tailed, the regime the grouped GEMM actually sees;
    pooled CV = 0.277; uniform baseline CV = 0.011 (=sqrt((E-1)/R)); top-1/top-10
    share = 1.43%/12.5%. Generator fit by L1 to the empirical sorted-share curve:
    Dirichlet-Multinomial (alpha~0.24, L1=0.129) beats logistic-normal
    softmax(sigma*z) (sigma~1.4, L1=0.147). alpha=0.24 makes the top expert ~8-11%
    of rows with a few cold/empty experts (the k==0 path fires); uniform/Gaussian
    (alpha -> inf) is far too flat. Inference-time routing, but the per-input-skewed
    shape is the transferable characterization for training-shaped workloads too.

    Returns:
        List of [a (Hmid), b (GY), cumsum_K]
    """
    torch.manual_seed(0)
    g = torch.distributions.Gamma(torch.full((NUM_EXPERTS,), 0.24), 1.0).sample()
    p = g / g.sum()                                  # p ~ Dirichlet(0.24 * 1_E)
    assign = torch.multinomial(p, TOTAL_ROWS, replacement=True)
    counts = torch.bincount(assign, minlength=NUM_EXPERTS).to(torch.int64).cuda()
    cumsum_K = torch.cumsum(counts, dim=0)            # [E], last == TOTAL_ROWS
    R = int(cumsum_K[-1].item())
    a = torch.randn(R, INTERMEDIATE_DIM, dtype=torch.float16, device='cuda')  # Hmid [R, D]
    b = torch.randn(R, HIDDEN_DIM, dtype=torch.float16, device='cuda')        # GY   [R, H]
    return [a, b, cumsum_K]


def get_init_inputs():
    """Returns [num_experts, M=D, N=H] -> output dW2 is [E, D, H]."""
    return [NUM_EXPERTS, INTERMEDIATE_DIM, HIDDEN_DIM]
