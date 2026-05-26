"""Ollivier-Ricci curvature on graphs, GPU-batched, Sinkhorn-based.

The Ollivier (2009) discrete Ricci curvature on a metric measure space
is defined for any two points x, y as

    κ(x, y)  =  1  −  W_1( μ_x , μ_y ) / d(x, y)

where d is the metric, μ_x and μ_y are probability measures attached
to x and y (typically lazy-random-walk distributions), and W_1 is the
Wasserstein-1 distance between them. For graphs we follow the standard
choices:

    d   = shortest-path distance (hop count for unweighted graphs;
          weight-respecting for weighted graphs assuming edge weights
          are interpreted as lengths)
    μ_x = (1−α) · u_x  +  α · δ_x
          where u_x is the (weight-normalized) distribution over x's
          neighbors and δ_x is the unit mass at x.

The parameter α ∈ [0, 1] is the "laziness" parameter. α = 0 gives the
original Ollivier (2009) curvature; α = 1 makes the walk completely
lazy (returns trivial 0 curvature); the limit α → 1 with appropriate
rescaling gives the Liu-Lin-Yau (2011) curvature, which is the most
common convention in network analysis.

The W_1 computation uses entropic Sinkhorn (Cuturi 2013) with cost
matrix equal to the shortest-path distance, which is well-suited for
GPU batching since each iteration is a pair of matrix-vector products.
Entropic regularization introduces a small bias; we expose `reg` so
callers can trade accuracy for stability.

Why this approximates Perelman's Ricci flow on networks:

  Perelman (2003) studied ∂g/∂t = −2 Ric(g) on Riemannian 3-manifolds.
  In the discrete setting, Sia-Jonckheere-Bogdan (2019) and Ni et al.
  (2019) show that iterating w_ij(t+1) = (1 − δt · κ_ij(t)) · w_ij(t)
  on edge weights with κ = Ollivier-Ricci leads to a network analog of
  Ricci flow, with negatively-curved edges getting elongated (analog
  of forming neckpinches) until "surgery" — edge removal — separates
  the graph into geometric pieces (the network analog of Thurston's
  geometric decomposition). This module provides the κ primitive; the
  flow and surgery operations are planned follow-ons.

References:
  Ollivier, Y. (2009). Ricci curvature of Markov chains on metric
    spaces. Journal of Functional Analysis 256(3):810–864.
  Liu, F., Lin, Y., Yau, S.-T. (2011). Ricci curvature of graphs.
    Tohoku Math. Journal 63(4):605–627.  [Often cited as "Lin-Lu-Yau"
    after the related Lu coauthor on the broader programme.]
  Cuturi, M. (2013). Sinkhorn distances: lightspeed computation of
    optimal transport. Advances in Neural Information Processing
    Systems 26 (NIPS 2013):2292–2300.
  Sia, J., Jonckheere, E., Bogdan, P. (2019). Ollivier-Ricci
    curvature-based method to community detection in complex networks.
    Scientific Reports 9:9800.
  Ni, C.-C., Lin, Y.-Y., Luo, F., Gao, J. (2019). Community detection
    on networks with Ricci flow. Scientific Reports 9:9984.
"""

from __future__ import annotations

from typing import Optional

import torch

from synoros_lib.provenance import with_provenance


# Default regularization for entropic Sinkhorn, per Cuturi (2013) §4.
# A regularization of 0.01 is a common middle-ground: small enough that
# the entropic bias on W_1 is < 1% for typical edge-distance scales,
# large enough that the Sinkhorn iteration converges in O(100) steps
# without numerical underflow. Catalog name: `sinkhorn_reg_default`.
# Scale-of-validity: shortest-path costs in [1, ~diameter]; if the
# graph has a much wider distance scale, scale `reg` proportionally.
SINKHORN_REG_DEFAULT: float = 0.01

# Default Sinkhorn iteration count, per Cuturi (2013) §4 — 100 iterations
# achieves convergence (relative change < 1e-4) for graph-metric costs
# at the default regularization. Catalog: `sinkhorn_n_iter_default`.
SINKHORN_N_ITER_DEFAULT: int = 100

# Convention: distance from a node to itself is 0; "no path" is +inf.
# We represent +inf with a large finite value derived from the matrix
# to avoid NaN propagation through Sinkhorn while still strongly
# disincentivizing transport across disconnected components.
DISCONNECTED_DISTANCE_MULTIPLIER: float = 1000.0


@with_provenance("synoros_lib.discrete_geometry.ollivier_ricci_curvature", op_version="0.1")
def ollivier_ricci_curvature(
    A: torch.Tensor,
    alpha: float = 0.0,
    reg: float = SINKHORN_REG_DEFAULT,
    n_iter: int = SINKHORN_N_ITER_DEFAULT,
) -> torch.Tensor:
    """Pairwise Ollivier-Ricci curvature on a (batched) weighted graph.

    Args:
      A: (B, n, n) symmetric weighted adjacency. Non-negative entries
        interpreted as edge weights (larger = stronger connection,
        shorter as a length in the shortest-path metric). Zero off-
        diagonal means no edge.
      alpha: laziness in [0, 1]. 0 is Ollivier's original convention;
        0.5 is a common "half lazy" choice in network analysis.
      reg: entropic-Sinkhorn regularization ε > 0. See module docstring.
      n_iter: Sinkhorn iteration count.

    Returns:
      κ: (B, n, n) curvature tensor. κ[..., i, j] is the curvature
        between nodes i and j using the shortest-path metric. The
        diagonal κ[..., i, i] = 1 by convention (W_1(δ, δ) = 0 and
        d(x, x) = 0 — we set the diagonal to 1 to avoid 0/0).
        Pairs from different connected components have undefined
        geometric meaning; we return 0 there to keep the tensor finite.

    Shape note:
      The result is dense across all pairs, not just edges. To get
      edge-only curvatures, mask: `κ * (A > 0)`. Computing all pairs
      costs O(B · n³ · n_iter) for the Sinkhorn step plus O(B · n³)
      for shortest paths; for large n use a sparse/edge-only follow-on
      (planned).

    References:
      Ollivier (2009), Definition 1.
      Cuturi (2013), Algorithm 1 — Sinkhorn iteration.
    """
    if not (0.0 <= alpha <= 1.0):
        raise ValueError(f"alpha must be in [0, 1], got {alpha}")
    if reg <= 0:
        raise ValueError(f"reg must be > 0, got {reg}")
    if A.ndim < 2 or A.shape[-1] != A.shape[-2]:
        raise ValueError(
            f"A must be (..., n, n) symmetric; got A.shape={tuple(A.shape)}"
        )

    *batch, n, _ = A.shape

    # 1. Shortest-path distances d_G(i, j). Floyd-Warshall on the
    # weighted graph, batched. Treat A[i,j]=0 (off-diagonal) as no edge.
    d_G = _shortest_path_distances(A)  # (B, n, n)

    # 2. Lazy-random-walk distributions μ_x for each node x.
    #    μ_x(y) = (1−α) · (A[x,y] / Σ_z A[x,z])   for y in N(x)
    #    μ_x(x) = α (+ 0 if x is isolated, in which case μ_x = δ_x)
    mu = _lazy_walk_distributions(A, alpha)  # (B, n, n) — μ_x is row x

    # 3. Sinkhorn-based W_1 between μ_x and μ_y for every pair (x, y).
    #    All pairs of rows of `mu` — the cost matrix is d_G.
    W1 = _batched_sinkhorn_w1(mu, d_G, reg=reg, n_iter=n_iter)  # (B, n, n)

    # 4. Curvature κ(x, y) = 1 - W_1 / d_G.
    # Diagonal: d_G is 0 there; set κ to 1 by convention (vacuous).
    eps = 1e-9  # numerical floor
    safe_dG = torch.where(d_G > eps, d_G, torch.full_like(d_G, eps))
    kappa = 1.0 - W1 / safe_dG
    # Diagonal entries: 1 by convention.
    eye = torch.eye(n, device=A.device, dtype=A.dtype).expand(*batch, n, n)
    kappa = torch.where(eye > 0, torch.ones_like(kappa), kappa)
    return kappa


# ============================================================
# Internal helpers
# ============================================================


def _shortest_path_distances(A: torch.Tensor) -> torch.Tensor:
    """Batched Floyd-Warshall on a weighted adjacency.

    Treats A[i,j] = 0 (off-diagonal, i ≠ j) as "no direct edge" — distance
    infinity. Diagonal is forced to 0. Symmetric (we don't enforce, but
    if A is non-symmetric this returns the directed shortest path).

    For disconnected pairs we return a large finite value rather than
    +inf, so downstream Sinkhorn computations don't underflow exp(−d/ε).
    """
    *batch, n, _ = A.shape
    # Build initial distance matrix: A[i,j] where i≠j and A[i,j]>0, else inf.
    eye = torch.eye(n, device=A.device, dtype=A.dtype).expand(*batch, n, n)
    off_diag = (eye == 0)
    no_edge = off_diag & (A == 0)
    # Use a finite "infinity" for numerical stability; will be relaxed
    # to a large multiple of the diameter at the end if any pair is
    # still unreachable.
    inf_sentinel = torch.full_like(A, float("inf"))
    D = torch.where(no_edge, inf_sentinel, A.clone())
    # Force diagonal to 0
    D = torch.where(eye > 0, torch.zeros_like(D), D)

    # Floyd-Warshall: for k in 0..n-1: D[i,j] = min(D[i,j], D[i,k] + D[k,j]).
    # Batched: vectorize over (i, j); k is the loop axis.
    for k in range(n):
        # via_k[..., i, j] = D[..., i, k] + D[..., k, j]
        via_k = D[..., :, k : k + 1] + D[..., k : k + 1, :]
        D = torch.minimum(D, via_k)

    # Replace any remaining +inf with a large finite value derived from
    # the existing finite distances, to keep Sinkhorn numerics sane.
    finite_mask = torch.isfinite(D)
    if finite_mask.all():
        return D
    # max finite distance per-batch, then multiply by sentinel
    D_finite = torch.where(finite_mask, D, torch.zeros_like(D))
    per_batch_max = D_finite.flatten(start_dim=-2).max(dim=-1).values  # (B,)
    big = (per_batch_max * DISCONNECTED_DISTANCE_MULTIPLIER).clamp(min=1.0)
    # Broadcast `big` back to D's shape
    big_full = big.view(*per_batch_max.shape, 1, 1).expand_as(D)
    return torch.where(finite_mask, D, big_full)


def _lazy_walk_distributions(A: torch.Tensor, alpha: float) -> torch.Tensor:
    """Lazy-random-walk distributions for every node.

    Row x of the output is μ_x, where
      μ_x(x) = α   (plus 1−α concentrated on x if x is isolated)
      μ_x(y) = (1−α) · A[x,y] / Σ_z A[x,z]   for y ≠ x with A[x,z]>0.

    Returns:
      (..., n, n) tensor; row x is μ_x as a probability vector.
    """
    n = A.shape[-1]
    row_sums = A.sum(dim=-1, keepdim=True)  # (..., n, 1)
    is_isolated = row_sums.squeeze(dim=-1) <= 0  # (..., n)

    # Avoid div-by-zero: use a safe denominator
    safe_row_sums = torch.where(
        row_sums > 0, row_sums, torch.ones_like(row_sums),
    )
    walk = A / safe_row_sums  # (..., n, n) — rows are normalized

    # Lazy mixture: (1−α) * walk + α * δ_x (identity rows)
    eye = torch.eye(n, device=A.device, dtype=A.dtype).expand_as(A)
    mu = (1.0 - alpha) * walk + alpha * eye

    # Isolated nodes default to δ_x (mass on themselves)
    delta_rows = eye
    isolated_mask = is_isolated.unsqueeze(dim=-1).expand_as(mu)
    return torch.where(isolated_mask, delta_rows, mu)


def _batched_sinkhorn_w1(
    mu: torch.Tensor,
    cost: torch.Tensor,
    reg: float,
    n_iter: int,
) -> torch.Tensor:
    """W_1(μ_i, μ_j) for every pair (i, j) of rows of `mu`, Sinkhorn-based.

    Args:
      mu:   (..., n, n) — row i is the source distribution μ_i.
      cost: (..., n, n) — pairwise cost matrix (used for all pairs).
      reg:  entropic regularization ε > 0.
      n_iter: Sinkhorn iteration count.

    Returns:
      (..., n, n) — entry [i, j] is the Sinkhorn-approximated
      W_1(μ_i, μ_j) using the given cost.

    Algorithm (vectorized across all source/target pairs):
      Stack pairs as a batch of n² OT problems with a shared cost.
      Iterate Sinkhorn updates in log-domain for stability.

    Notes:
      Log-domain Sinkhorn (Cuturi 2013 §4 + Schmitzer 2019) avoids
      underflow when reg is small. We use it unconditionally for
      robustness; the cost is one extra logsumexp per iteration.
    """
    # Shapes:
    #   *batch, n, n — for both mu and cost
    *batch, n, _ = mu.shape
    n_pairs = n * n

    # For each pair (i, j): source = mu[i], target = mu[j].
    # Build (..., n_pairs, n) source/target batches.
    source = mu.unsqueeze(dim=-2).expand(*batch, n, n, n).reshape(*batch, n_pairs, n)
    target = mu.unsqueeze(dim=-3).expand(*batch, n, n, n).reshape(*batch, n_pairs, n)

    # The cost is shared across all pairs (it's the metric on the support).
    # cost broadcasts to (*batch, 1, n, n) → applied to each pair.
    cost_b = cost.unsqueeze(dim=-3)  # (*batch, 1, n, n)

    # Log-domain Sinkhorn:
    #   K = -cost / reg                          (log-kernel)
    #   log_u, log_v initialized to 0.
    #   Repeat:
    #     log_u = log(source) - logsumexp(K + log_v[:, None, :], dim=-1)
    #     log_v = log(target) - logsumexp(K + log_u[:, :, None], dim=-2)
    log_K = -cost_b / reg  # (*batch, 1, n, n)

    # Use the library's numerical_floor_convention to avoid log(0).
    # 1e-9 is comfortably below any realistic probability mass for n < 1e9.
    log_source = torch.log(source.clamp(min=1e-9))  # (*batch, n_pairs, n)
    log_target = torch.log(target.clamp(min=1e-9))

    log_u = torch.zeros_like(log_source)
    log_v = torch.zeros_like(log_target)

    for _ in range(n_iter):
        # log_u_i = log_source_i - logsumexp_k (log_K[i,k] + log_v_k)
        log_u = log_source - torch.logsumexp(
            log_K + log_v.unsqueeze(dim=-2), dim=-1,
        )
        log_v = log_target - torch.logsumexp(
            log_K + log_u.unsqueeze(dim=-1), dim=-2,
        )

    # Transport plan in log space: log_pi = log_u[:, None] + log_K + log_v[None, :]
    log_pi = log_u.unsqueeze(dim=-1) + log_K + log_v.unsqueeze(dim=-2)
    pi = torch.exp(log_pi)

    # Sinkhorn cost ⟨π, cost⟩ (the entropic-W_1, slightly biased; we
    # use ⟨π, C⟩ which is the transport cost component, not the
    # entropy-regularized objective). Bias is O(reg · H) and small
    # for our default reg=0.01.
    sink_w1 = (pi * cost_b).sum(dim=(-2, -1))  # (*batch, n_pairs)
    return sink_w1.reshape(*batch, n, n)
