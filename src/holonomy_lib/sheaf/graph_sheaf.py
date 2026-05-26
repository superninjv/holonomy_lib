"""GraphSheaf dataclass — the data of a cellular sheaf on a graph.

The sheaf attaches a `d_v`-dim stalk to each node and a `d_e`-dim
stalk to each edge. For every edge `e = (u, v)` it carries two
restriction maps:

  F_left[e]  : R^{d_v} → R^{d_e}   (the map from node u's stalk)
  F_right[e] : R^{d_v} → R^{d_e}   (the map from node v's stalk)

The coboundary on a node-stalk vector `x ∈ R^{n_v · d_v}` reads each
edge's contribution as `F_left[e] · x_u − F_right[e] · x_v`. The
sheaf Laplacian is `δ^T δ`.

v1 restrictions:
  - Uniform stalk dims: every node has the same `d_v`, every edge the
    same `d_e`. (Heterogeneous stalks are mathematically fine and a
    natural v2 extension; v1 keeps the storage as dense tensors.)
  - Undirected graphs: edges are unordered, but our (u, v) tuple
    convention encodes the orientation that defines `δ`'s sign
    convention. Two sheaves with the same edge set but swapped (u, v)
    orientations represent the same underlying sheaf with reversed
    coboundary signs; the Laplacian `L_F = δ^T δ` is identical in
    both since it is quadratic in δ.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class GraphSheaf:
    """A cellular sheaf on a graph.

    Attributes:
      n_nodes:      number of nodes (each carries a d_v-dim stalk).
      edges:        `(n_edges, 2)` int64 tensor of `(u, v)` pairs.
      node_stalk_dim:  `d_v`.
      edge_stalk_dim:  `d_e`.
      F_left:       `(n_edges, d_e, d_v)` restriction maps for the
                    `u`-side of each edge.
      F_right:      `(n_edges, d_e, d_v)` restriction maps for the
                    `v`-side of each edge.

    Validation:
      - `edges` must be `(n_edges, 2)` int64.
      - `F_left`, `F_right` shapes match `(n_edges, d_e, d_v)`.
      - All tensors must share `device` and a single float dtype.
    """

    n_nodes: int
    edges: torch.Tensor
    node_stalk_dim: int
    edge_stalk_dim: int
    F_left: torch.Tensor
    F_right: torch.Tensor

    def __post_init__(self) -> None:
        if self.n_nodes <= 0:
            raise ValueError(f"n_nodes must be > 0; got {self.n_nodes}")
        if self.node_stalk_dim <= 0:
            raise ValueError(
                f"node_stalk_dim must be > 0; got {self.node_stalk_dim}"
            )
        if self.edge_stalk_dim <= 0:
            raise ValueError(
                f"edge_stalk_dim must be > 0; got {self.edge_stalk_dim}"
            )
        if self.edges.ndim != 2 or self.edges.shape[-1] != 2:
            raise ValueError(
                f"edges must be (n_edges, 2); got "
                f"shape={tuple(self.edges.shape)}"
            )
        n_e = self.edges.shape[0]
        d_e, d_v = self.edge_stalk_dim, self.node_stalk_dim
        for name, t in (("F_left", self.F_left), ("F_right", self.F_right)):
            if t.shape != (n_e, d_e, d_v):
                raise ValueError(
                    f"{name} must have shape (n_edges={n_e}, "
                    f"edge_stalk_dim={d_e}, node_stalk_dim={d_v}); got "
                    f"shape={tuple(t.shape)}"
                )
        if self.F_left.device != self.F_right.device:
            raise ValueError(
                f"F_left and F_right must be on the same device; "
                f"got {self.F_left.device} vs {self.F_right.device}"
            )
        if self.F_left.dtype != self.F_right.dtype:
            raise ValueError(
                f"F_left and F_right must share dtype; got "
                f"{self.F_left.dtype} vs {self.F_right.dtype}"
            )
        # Edge indices must lie in [0, n_nodes).
        if (self.edges < 0).any() or (self.edges >= self.n_nodes).any():
            raise ValueError(
                f"edges entries must be in [0, n_nodes={self.n_nodes})"
            )

    @property
    def n_edges(self) -> int:
        return int(self.edges.shape[0])

    @property
    def device(self) -> torch.device:
        return self.F_left.device

    @property
    def dtype(self) -> torch.dtype:
        return self.F_left.dtype

    def _provenance_signature(self) -> dict:
        """Deterministic canonical form for `@with_provenance` hashing.

        Sheaf identity = topology (n_nodes + edges) + stalk dims +
        device + dtype. The restriction maps are themselves tensors,
        so they pass through `_resolve_tensor_hex` separately and
        carry their content into the input_hexes of any decorated
        sheaf op; we do NOT put them in the signature dict.
        """
        edges_tuple = tuple(map(tuple, self.edges.cpu().tolist()))
        return {
            "class": "GraphSheaf",
            "n_nodes": self.n_nodes,
            "node_stalk_dim": self.node_stalk_dim,
            "edge_stalk_dim": self.edge_stalk_dim,
            "edges": edges_tuple,
            "device": str(self.device),
            "dtype": str(self.dtype),
        }

    @staticmethod
    def trivial(
        n_nodes: int,
        edges: torch.Tensor,
        stalk_dim: int = 1,
        device: str | torch.device = "cpu",
        dtype: torch.dtype = torch.float64,
    ) -> "GraphSheaf":
        """Construct the *trivial* sheaf: identity restriction maps on
        every edge, uniform stalk dimension on nodes and edges.

        With `stalk_dim = 1` this is the sheaf whose Laplacian equals
        the standard combinatorial graph Laplacian; with `stalk_dim
        = d > 1` it's the block-diagonal `d × L` "d copies of the
        scalar Laplacian" sheaf. Useful baseline + smoke test.
        """
        if edges.ndim != 2 or edges.shape[-1] != 2:
            raise ValueError(
                f"edges must be (n_edges, 2); got shape={tuple(edges.shape)}"
            )
        n_e = edges.shape[0]
        eye = torch.eye(stalk_dim, dtype=dtype, device=device)
        F = eye.unsqueeze(dim=0).expand(n_e, stalk_dim, stalk_dim).contiguous()
        edges = edges.to(device=device, dtype=torch.int64)
        return GraphSheaf(
            n_nodes=n_nodes,
            edges=edges,
            node_stalk_dim=stalk_dim,
            edge_stalk_dim=stalk_dim,
            F_left=F,
            F_right=F.clone(),
        )
