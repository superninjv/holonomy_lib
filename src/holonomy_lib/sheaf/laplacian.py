"""Sheaf coboundary, sheaf Laplacian, and Dirichlet energy on graphs.

The coboundary operator `δ: C^0 → C^1` sends a node-stalk vector
`x ∈ R^{n_v · d_v}` to an edge-stalk vector whose value at each edge
`e = (u, v)` is `F_left[e] · x_u − F_right[e] · x_v`. Stacked across
all edges, `δ` is an `(n_e · d_e) × (n_v · d_v)` matrix.

The sheaf Laplacian is `L_F = δ^T δ`, which is symmetric PSD by
construction. Its kernel is the space of **global sections**: tuples
of node-stalk values that all incident edges agree on under the
restriction maps. For the trivial sheaf this kernel is the constant
vectors (one per connected component) — exactly the kernel of the
combinatorial Laplacian.

Implementation: builds `δ` densely in v1. For a graph with `n_e`
edges, node-stalk dim `d_v`, edge-stalk dim `d_e`, this is a `(n_e
· d_e) × (n_v · d_v)` matrix — fine up to thousands of nodes; a
sparse path is a natural extension for large graphs.

References:
  Hansen-Ghrist (2019), §3-4 — sheaf Laplacian + spectral theory.
  Bodnar et al. (2022), eq. 6 — the cellular-sheaf Laplacian as a
    block matrix.
"""

from __future__ import annotations

import torch

from holonomy_lib.provenance import with_provenance
from holonomy_lib.sheaf.graph_sheaf import GraphSheaf


@with_provenance(
    "holonomy_lib.sheaf.sheaf_coboundary", op_version="0.1",
)
def sheaf_coboundary(sheaf: GraphSheaf) -> torch.Tensor:
    """Build the sheaf coboundary `δ` as a dense `(n_e·d_e, n_v·d_v)`
    tensor.

    For edge index `e = (u, v)`, the corresponding `d_e`-row block of
    `δ` has `+F_left[e]` at columns `u·d_v : (u+1)·d_v` and
    `−F_right[e]` at columns `v·d_v : (v+1)·d_v`.

    Args:
      sheaf: a `GraphSheaf`.

    Returns:
      `(n_e · d_e, n_v · d_v)` coboundary on the sheaf's device/dtype.

    References:
      Hansen-Ghrist (2019), eq. 2.
    """
    n_e = sheaf.n_edges
    n_v = sheaf.n_nodes
    d_e = sheaf.edge_stalk_dim
    d_v = sheaf.node_stalk_dim
    device, dtype = sheaf.device, sheaf.dtype

    delta = torch.zeros(n_e * d_e, n_v * d_v, device=device, dtype=dtype)

    if n_e == 0:
        return delta

    # Vectorized assembly: build a (n_e, d_e, n_v * d_v) staging tensor
    # then reshape to (n_e * d_e, n_v * d_v). Allocating one big
    # intermediate beats a Python loop over edges for any non-trivial n_e.
    block = torch.zeros(n_e, d_e, n_v * d_v, device=device, dtype=dtype)
    edge_idx = torch.arange(n_e, device=device)
    u_idx = sheaf.edges[:, 0]                                     # (n_e,)
    v_idx = sheaf.edges[:, 1]                                     # (n_e,)
    # For each edge e, place F_left[e] into columns [u·d_v : (u+1)·d_v].
    # Use advanced indexing with an explicit column-range tensor.
    col_off = torch.arange(d_v, device=device)
    u_cols = u_idx.unsqueeze(dim=-1) * d_v + col_off              # (n_e, d_v)
    v_cols = v_idx.unsqueeze(dim=-1) * d_v + col_off              # (n_e, d_v)
    # Broadcast assignment: block[e, :, u_cols[e, k]] = +F_left[e, :, k].
    # scatter is cleanest with the (n_e, d_e, d_v) source.
    block.scatter_(
        dim=-1,
        index=u_cols.unsqueeze(dim=-2).expand(n_e, d_e, d_v),
        src=sheaf.F_left,
    )
    # `scatter_` overwrites; for v_cols we want to subtract F_right.
    # Pull current values at v_cols (which are 0, since we haven't
    # touched those entries — but defensively use `scatter_add` with
    # -F_right against zeros to compose with any prior content).
    block.scatter_add_(
        dim=-1,
        index=v_cols.unsqueeze(dim=-2).expand(n_e, d_e, d_v),
        src=-sheaf.F_right,
    )

    return block.reshape(n_e * d_e, n_v * d_v)


@with_provenance(
    "holonomy_lib.sheaf.sheaf_laplacian", op_version="0.1",
)
def sheaf_laplacian(sheaf: GraphSheaf) -> torch.Tensor:
    """Sheaf Laplacian `L_F = δ^T δ`, a symmetric PSD operator of
    shape `(n_v · d_v, n_v · d_v)`.

    For the *trivial* sheaf (every restriction map is identity,
    uniform stalk dim `d`):
      d = 1   → L_F equals the combinatorial graph Laplacian.
      d > 1   → L_F is the block-diagonal `d × L` (each channel is a
                copy of the scalar Laplacian, no cross-channel mixing).

    Args:
      sheaf: a `GraphSheaf`.

    Returns:
      `(n_v · d_v, n_v · d_v)` symmetric PSD Laplacian.

    References:
      Hansen-Ghrist (2019), §3.1.
      Bodnar et al. (2022), eq. 6.
    """
    delta = sheaf_coboundary(sheaf)
    # `δ^T @ δ` is the canonical form; matmul of (n_v·d_v, n_e·d_e) by
    # its transpose. Symmetrize against any drift introduced by the
    # blocked BLAS path (rounding can leave |L − L^T|_∞ ~ 1e-15 even
    # though mathematically `δ^T δ` is exactly symmetric — symmetrize
    # so callers can chain straight into `torch.linalg.eigh`).
    L = delta.mT @ delta
    return 0.5 * (L + L.mT)


@with_provenance(
    "holonomy_lib.sheaf.sheaf_dirichlet_energy", op_version="0.1",
)
def sheaf_dirichlet_energy(
    sheaf: GraphSheaf, x: torch.Tensor,
) -> torch.Tensor:
    """Per-batch Dirichlet energy `E(x) = x^T L_F x` for sheaf-valued
    signals.

    The Dirichlet energy is the squared coboundary norm — it measures
    how much each edge's two restriction-mapped values disagree.
    Minimized at the global sections (kernel of `L_F`).

    Args:
      sheaf: a `GraphSheaf`.
      x: `(B, n_v · d_v)` or `(n_v · d_v,)` node-stalk signals.

    Returns:
      Scalar `()` for unbatched input or `(B,)` for batched input.

    References:
      Hansen-Ghrist (2019), §3.2.
    """
    n_v, d_v = sheaf.n_nodes, sheaf.node_stalk_dim
    if x.shape[-1] != n_v * d_v:
        raise ValueError(
            f"x last dim must equal n_nodes·node_stalk_dim = {n_v * d_v}; "
            f"got x.shape={tuple(x.shape)}"
        )
    delta = sheaf_coboundary(sheaf)                  # (n_e·d_e, n_v·d_v)
    # x: (..., n_v·d_v); δ x: (..., n_e·d_e); energy: (..., )
    delta_x = torch.einsum("ij,...j->...i", delta, x)
    return (delta_x * delta_x).sum(dim=-1)
