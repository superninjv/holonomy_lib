"""Tests for synoros_lib.discrete_geometry.ricci.ollivier_ricci_curvature.

Sinkhorn introduces a small entropic bias on W_1, so closed-form
expectations are checked with a tolerance commensurate with the default
regularization (~1%). Tighter checks use `reg=0.005` to push the bias
down further.
"""

from __future__ import annotations

import pytest
import torch

from synoros_lib.discrete_geometry import ollivier_ricci_curvature


def _complete_graph(n: int, batch: int = 1, dtype=torch.float64) -> torch.Tensor:
    """K_n complete graph adjacency, batched."""
    A = torch.ones(batch, n, n, dtype=dtype) - torch.eye(n, dtype=dtype).unsqueeze(dim=0)
    return A


def _path_graph(n: int, batch: int = 1, dtype=torch.float64) -> torch.Tensor:
    """P_n path graph: 0 — 1 — 2 — ... — n−1."""
    A = torch.zeros(batch, n, n, dtype=dtype)
    for i in range(n - 1):
        A[:, i, i + 1] = 1.0
        A[:, i + 1, i] = 1.0
    return A


def _cycle_graph(n: int, batch: int = 1, dtype=torch.float64) -> torch.Tensor:
    """C_n cycle graph."""
    A = _path_graph(n, batch=batch, dtype=dtype)
    A[:, 0, n - 1] = 1.0
    A[:, n - 1, 0] = 1.0
    return A


# --------------------------------------------------------------------
# Input validation
# --------------------------------------------------------------------


class TestValidation:
    def test_rejects_alpha_out_of_range(self):
        with pytest.raises(ValueError, match="alpha"):
            ollivier_ricci_curvature(_complete_graph(3), alpha=-0.1)
        with pytest.raises(ValueError, match="alpha"):
            ollivier_ricci_curvature(_complete_graph(3), alpha=1.5)

    def test_rejects_nonpositive_reg(self):
        with pytest.raises(ValueError, match="reg"):
            ollivier_ricci_curvature(_complete_graph(3), reg=0.0)

    def test_rejects_non_square(self):
        with pytest.raises(ValueError, match="must be"):
            ollivier_ricci_curvature(torch.zeros(1, 4, 5))


# --------------------------------------------------------------------
# Shapes
# --------------------------------------------------------------------


@pytest.mark.parametrize("batch", [1, 2])
class TestShapes:
    def test_output_shape(self, batch):
        A = _complete_graph(4, batch=batch)
        kappa = ollivier_ricci_curvature(A, alpha=0.0)
        assert kappa.shape == (batch, 4, 4)


# --------------------------------------------------------------------
# Closed-form properties
# --------------------------------------------------------------------


class TestClosedForms:
    def test_complete_graph_K3(self):
        """On K_3, κ(edge) = 1/2 for α=0.

        Proof: μ_x = (0, 1/2, 1/2) on (x, y, z); μ_y = (1/2, 0, 1/2);
        optimal transport moves 1/2 from y to x at cost 1; d(x, y) = 1;
        κ = 1 − 1/2 = 1/2.
        """
        A = _complete_graph(3)
        kappa = ollivier_ricci_curvature(A, alpha=0.0, reg=0.005, n_iter=300)
        # Off-diagonal entries (i, j with i≠j) should all equal 1/2
        i, j = torch.triu_indices(3, 3, offset=1)
        off_diag = kappa[0, i, j]
        torch.testing.assert_close(
            off_diag, torch.full_like(off_diag, 0.5), atol=0.02, rtol=0,
        )

    def test_complete_graph_K_n_formula(self):
        """On K_n, κ(any edge) = (n−2)/(n−1).

        Derivation: μ_x, μ_y are uniform on each other's neighborhoods,
        differing only by swapping mass at x and y, each of weight 1/(n−1).
        Optimal transport moves 1/(n−1) from y → x at unit cost.
        """
        for n in [3, 4, 5, 6]:
            A = _complete_graph(n)
            kappa = ollivier_ricci_curvature(A, alpha=0.0, reg=0.005, n_iter=300)
            expected = (n - 2) / (n - 1)
            i, j = torch.triu_indices(n, n, offset=1)
            off_diag = kappa[0, i, j]
            torch.testing.assert_close(
                off_diag, torch.full_like(off_diag, expected),
                atol=0.02, rtol=0,
            ), f"K_{n}: expected κ={expected}, got {off_diag}"

    def test_path_interior_edge_is_flat(self):
        """On a path P_n, the curvature of an interior edge is ≈ 0
        (paths are "flat" — neither positively nor negatively curved).
        """
        A = _path_graph(7)  # 0—1—2—3—4—5—6; edge (3, 4) is interior
        kappa = ollivier_ricci_curvature(A, alpha=0.0, reg=0.005, n_iter=300)
        # Edge (3, 4) is the most-interior, fully symmetric
        assert abs(kappa[0, 3, 4].item()) < 0.05

    def test_diagonal_is_one(self):
        """κ(x, x) = 1 by convention (W_1(δ_x, δ_x) = 0, d=0 vacuous)."""
        A = _complete_graph(5)
        kappa = ollivier_ricci_curvature(A, alpha=0.0)
        diag = torch.diagonal(kappa, dim1=-2, dim2=-1)  # (B, n)
        torch.testing.assert_close(
            diag, torch.ones_like(diag), atol=0, rtol=0,
        )

    def test_symmetric_unweighted(self):
        """κ(x, y) = κ(y, x) for unweighted graphs (where Sinkhorn
        converges tightly).

        Sinkhorn alternates u-then-v updates each iteration, so the
        approximate W_1 is symmetric only to Sinkhorn convergence
        tolerance. On unweighted K_n the cost matrix has small dynamic
        range so convergence is fast and symmetry is tight.
        """
        A = _complete_graph(5)
        kappa = ollivier_ricci_curvature(A, alpha=0.0, reg=0.005, n_iter=300)
        torch.testing.assert_close(kappa, kappa.mT, atol=1e-6, rtol=0)

    def test_approximately_symmetric_weighted(self):
        """On weighted graphs, κ(x, y) ≈ κ(y, x) up to Sinkhorn
        convergence tolerance (a few percent).
        """
        n = 5
        g = torch.Generator()
        g.manual_seed(7)
        U = torch.rand(1, n, n, generator=g, dtype=torch.float64)
        A = torch.triu(U, diagonal=1)
        A = A + A.mT
        kappa = ollivier_ricci_curvature(A, alpha=0.0, reg=0.01, n_iter=200)
        # Allow ~5% (the entropic-Sinkhorn convergence floor)
        torch.testing.assert_close(kappa, kappa.mT, atol=0.05, rtol=0)


# --------------------------------------------------------------------
# Bottleneck / community structure: surgery-relevant properties
# --------------------------------------------------------------------


class TestBottleneckCurvature:
    def test_bridge_edge_more_negative_than_intra_cluster(self):
        """For a two-clique graph K_4 ⊔_e K_4 (joined by a single bridge
        edge), the bridge edge has curvature much lower than intra-clique
        edges. This is the signature surgery exploits (Sia 2019, Ni 2019).
        """
        n = 8  # nodes 0..3 in clique 0, 4..7 in clique 1; bridge 3-4
        A = torch.zeros(1, n, n, dtype=torch.float64)
        # Intra-clique 0..3 (K_4)
        for i in range(4):
            for j in range(4):
                if i != j:
                    A[0, i, j] = 1.0
        # Intra-clique 4..7 (K_4)
        for i in range(4, 8):
            for j in range(4, 8):
                if i != j:
                    A[0, i, j] = 1.0
        # Single bridge
        A[0, 3, 4] = A[0, 4, 3] = 1.0
        kappa = ollivier_ricci_curvature(A, alpha=0.0, reg=0.005, n_iter=300)
        # Bridge edge curvature
        bridge = kappa[0, 3, 4].item()
        # An intra-clique edge (say 0–1) curvature
        intra = kappa[0, 0, 1].item()
        # We expect bridge ≪ intra (much more negative)
        assert bridge < intra - 0.3, (
            f"bridge κ={bridge:.3f} should be much less than intra κ={intra:.3f}"
        )

    def test_complete_clique_has_positive_curvature(self):
        """All edges in a complete graph have positive curvature."""
        for n in [4, 5, 6]:
            A = _complete_graph(n)
            kappa = ollivier_ricci_curvature(A, alpha=0.0, reg=0.005, n_iter=300)
            i, j = torch.triu_indices(n, n, offset=1)
            assert (kappa[0, i, j] > 0).all(), (
                f"K_{n}: all edges should have positive κ"
            )


# --------------------------------------------------------------------
# Comparison against GraphRicciCurvature library
# --------------------------------------------------------------------


try:
    import GraphRicciCurvature  # noqa: F401
    import networkx  # noqa: F401
    _HAVE_GRC = True
except ImportError:
    _HAVE_GRC = False


@pytest.mark.skipif(not _HAVE_GRC, reason="GraphRicciCurvature not installed")
class TestAgainstGraphRicciCurvature:
    """Cross-check against the canonical CPU NetworkX implementation.

    GraphRicciCurvature uses POT (exact LP-based W_1) under the hood
    while we use entropic Sinkhorn, so we allow ~2% tolerance per edge.
    """

    def test_K4_matches_grc(self):
        import networkx as nx
        from GraphRicciCurvature.OllivierRicci import OllivierRicci

        # K_4
        A = _complete_graph(4)
        kappa_ours = ollivier_ricci_curvature(
            A, alpha=0.0, reg=0.005, n_iter=400,
        )

        # Reference — note that `compute_ricci_curvature` stores results
        # on `orc.G`, not the input graph.
        G = nx.complete_graph(4)
        orc = OllivierRicci(G, alpha=0.0, verbose="ERROR")
        orc.compute_ricci_curvature()
        for u, v, data in orc.G.edges(data=True):
            ref = data["ricciCurvature"]
            ours = kappa_ours[0, u, v].item()
            assert abs(ours - ref) < 0.03, (
                f"K_4 edge ({u},{v}): ours={ours:.4f}, ref={ref:.4f}"
            )
