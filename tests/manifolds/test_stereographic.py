"""Tests for holonomy_lib.manifolds.stereographic.KappaStereographicManifold.

Six layers, parametrized across the three κ branches (spherical,
Euclidean, hyperbolic):
  1. Construction validation.
  2. Shapes across B ∈ {0, 1, several}.
  3. Property tests — exp/log inverse, distance symmetry, Möbius
     additive inverse, Euclidean recovery at κ = 0.
  4. Autograd-finite tests at boundary inputs.
  5. Provenance signature roundtrip.
  6. Cross-comparison against geoopt's Stereographic (when installed).
"""

from __future__ import annotations

import math

import pytest
import torch

from holonomy_lib.manifolds import KappaStereographicManifold


# Test the three κ branches plus a couple of magnitudes
KAPPA_VALUES = [-1.0, -0.5, 0.0, 0.5, 1.0]


def _make_manifold(n=3, kappa=-1.0, dtype=torch.float64):
    return KappaStereographicManifold(n=n, kappa=kappa, dtype=dtype)


def _seed(s: int) -> torch.Generator:
    g = torch.Generator()
    g.manual_seed(s)
    return g


# --------------------------------------------------------------------
# Construction
# --------------------------------------------------------------------


class TestConstruction:
    def test_rejects_n_zero(self):
        with pytest.raises(ValueError, match="n"):
            KappaStereographicManifold(n=0)

    def test_rejects_n_negative(self):
        with pytest.raises(ValueError, match="n"):
            KappaStereographicManifold(n=-2)

    def test_rejects_non_float_kappa(self):
        with pytest.raises(TypeError, match="kappa"):
            KappaStereographicManifold(n=3, kappa=torch.tensor(0.5))

    @pytest.mark.parametrize("k", KAPPA_VALUES)
    def test_branch_dispatch(self, k):
        mfd = _make_manifold(kappa=k)
        if k > 0:
            assert mfd._branch == "spherical"
        elif k < 0:
            assert mfd._branch == "hyperbolic"
        else:
            assert mfd._branch == "euclidean"

    def test_dim_and_ambient_dim_equal(self):
        mfd = _make_manifold(n=5)
        assert mfd.dim == 5
        assert mfd.ambient_dim == 5


# --------------------------------------------------------------------
# Shapes across batch sizes
# --------------------------------------------------------------------


@pytest.mark.parametrize("batch", [0, 1, 4])
@pytest.mark.parametrize("k", KAPPA_VALUES)
class TestShapes:
    def test_random_point(self, batch, k):
        mfd = _make_manifold(kappa=k)
        x = mfd.random_point(batch_size=batch, generator=_seed(0))
        assert x.shape == (batch, mfd.n)

    def test_origin(self, batch, k):
        mfd = _make_manifold(kappa=k)
        o = mfd.origin(batch_size=batch)
        assert o.shape == (batch, mfd.n)

    def test_exp_log(self, batch, k):
        mfd = _make_manifold(kappa=k)
        x = mfd.random_point(batch_size=batch, generator=_seed(1))
        v = torch.randn(batch, mfd.n, dtype=mfd.dtype,
                        generator=_seed(2)) * 0.05
        y = mfd.exp(x, v)
        out = mfd.log(x, y)
        assert y.shape == (batch, mfd.n)
        assert out.shape == (batch, mfd.n)

    def test_distance(self, batch, k):
        mfd = _make_manifold(kappa=k)
        x = mfd.random_point(batch_size=batch, generator=_seed(3))
        y = mfd.random_point(batch_size=batch, generator=_seed(4))
        d = mfd.distance(x, y)
        assert d.shape == (batch,)


# --------------------------------------------------------------------
# Properties — per-κ-branch invariants
# --------------------------------------------------------------------


@pytest.mark.parametrize("k", KAPPA_VALUES)
class TestProperties:
    def test_random_point_in_domain(self, k):
        mfd = _make_manifold(kappa=k)
        x = mfd.random_point(batch_size=8, generator=_seed(10))
        assert mfd.is_on_manifold(x).all()

    def test_origin_on_manifold(self, k):
        mfd = _make_manifold(kappa=k)
        o = mfd.origin(batch_size=3)
        assert mfd.is_on_manifold(o).all()

    def test_mobius_add_neutral_at_origin(self, k):
        """0 ⊕_κ y = y and x ⊕_κ 0 = x."""
        mfd = _make_manifold(kappa=k)
        o = mfd.origin(batch_size=3)
        y = mfd.random_point(batch_size=3, generator=_seed(11))
        torch.testing.assert_close(
            mfd.mobius_add(o, y), y, atol=1e-12, rtol=1e-12,
        )
        torch.testing.assert_close(
            mfd.mobius_add(y, o), y, atol=1e-12, rtol=1e-12,
        )

    def test_mobius_add_inverse(self, k):
        """(-x) ⊕_κ x = 0 (Möbius additive inverse)."""
        mfd = _make_manifold(kappa=k)
        x = mfd.random_point(batch_size=3, generator=_seed(12))
        result = mfd.mobius_add(-x, x)
        torch.testing.assert_close(
            result, torch.zeros_like(result), atol=1e-10, rtol=0,
        )

    def test_exp_log_inverse(self, k):
        """log_x(exp_x(v)) = v for small v."""
        mfd = _make_manifold(kappa=k)
        x = mfd.random_point(batch_size=3, generator=_seed(13))
        v = torch.randn(3, mfd.n, dtype=mfd.dtype,
                        generator=_seed(14)) * 0.05
        v_back = mfd.log(x, mfd.exp(x, v))
        torch.testing.assert_close(v_back, v, atol=1e-9, rtol=1e-9)

    def test_log_exp_inverse(self, k):
        """exp_x(log_x(y)) = y."""
        mfd = _make_manifold(kappa=k)
        x = mfd.random_point(batch_size=3, generator=_seed(15))
        y = mfd.random_point(batch_size=3, generator=_seed(16))
        y_back = mfd.exp(x, mfd.log(x, y))
        torch.testing.assert_close(y_back, y, atol=1e-9, rtol=1e-9)

    def test_exp_0_log_0_inverse(self, k):
        """log_0(exp_0(v)) = v."""
        mfd = _make_manifold(kappa=k)
        v = torch.randn(4, mfd.n, dtype=mfd.dtype,
                        generator=_seed(17)) * 0.1
        v_back = mfd.log_0(mfd.exp_0(v))
        torch.testing.assert_close(v_back, v, atol=1e-9, rtol=1e-9)

    def test_distance_symmetric(self, k):
        mfd = _make_manifold(kappa=k)
        x = mfd.random_point(batch_size=4, generator=_seed(18))
        y = mfd.random_point(batch_size=4, generator=_seed(19))
        torch.testing.assert_close(
            mfd.distance(x, y), mfd.distance(y, x),
            atol=1e-10, rtol=0,
        )

    def test_distance_at_same_point(self, k):
        """d(x, x) = 0 to machine precision."""
        mfd = _make_manifold(kappa=k)
        x = mfd.random_point(batch_size=3, generator=_seed(20))
        d = mfd.distance(x, x)
        assert d.abs().max().item() < 1e-9


# --------------------------------------------------------------------
# κ = 0 Euclidean recovery (separately verified — the limit is the
# Bachmann-normalization Euclidean form, where d_κ=0 = 2·‖y - x‖).
# --------------------------------------------------------------------


class TestEuclideanRecovery:
    """At κ = 0 the manifold IS Euclidean R^n — but with the conformal-
    factor-2 normalization Bachmann uses, distances are 2× the standard."""

    def test_exp_is_identity(self):
        mfd = _make_manifold(kappa=0.0)
        x = mfd.random_point(batch_size=3, generator=_seed(30))
        v = torch.randn(3, mfd.n, dtype=mfd.dtype,
                        generator=_seed(31)) * 0.5
        out = mfd.exp(x, v)
        torch.testing.assert_close(out, x + v, atol=1e-12, rtol=0)

    def test_log_is_difference_over_lambda(self):
        """log_x(y) = (2/λ_κ(x)) · (y - x). At κ=0 λ_κ ≡ 2, so log_x(y)
        = y - x."""
        mfd = _make_manifold(kappa=0.0)
        x = mfd.random_point(batch_size=3, generator=_seed(32))
        y = mfd.random_point(batch_size=3, generator=_seed(33))
        out = mfd.log(x, y)
        torch.testing.assert_close(out, y - x, atol=1e-12, rtol=0)

    def test_distance_is_twice_euclidean(self):
        """d_{κ=0}(x, y) = 2 · ‖y - x‖_2 (Bachmann normalization)."""
        mfd = _make_manifold(kappa=0.0)
        x = mfd.random_point(batch_size=4, generator=_seed(34))
        y = mfd.random_point(batch_size=4, generator=_seed(35))
        d = mfd.distance(x, y)
        expected = 2.0 * torch.linalg.vector_norm(y - x, dim=-1)
        torch.testing.assert_close(d, expected, atol=1e-10, rtol=1e-10)


# --------------------------------------------------------------------
# Autograd-finite at boundary inputs
# --------------------------------------------------------------------


class TestAutogradFinite:
    """Same boundary cases as Lorentz: x = y, v = 0, y = origin must
    all produce finite gradients in backward."""

    @pytest.mark.parametrize("k", KAPPA_VALUES)
    def test_distance_at_same_point_backward(self, k):
        mfd = _make_manifold(kappa=k)
        v = (torch.randn(5, mfd.n, dtype=torch.float64,
                          generator=_seed(40)) * 0.1)
        v.requires_grad_(True)
        x = mfd.exp_0(v)
        d = mfd.distance(x, x)
        d.sum().backward()
        assert torch.isfinite(v.grad).all(), (
            f"κ={k}: v.grad NaN: {torch.isnan(v.grad).sum().item()}/"
            f"{v.grad.numel()}"
        )

    @pytest.mark.parametrize("k", KAPPA_VALUES)
    def test_exp_0_backward_at_zero(self, k):
        mfd = _make_manifold(kappa=k)
        v = torch.zeros(3, mfd.n, dtype=torch.float64, requires_grad=True)
        x = mfd.exp_0(v)
        x.sum().backward()
        assert torch.isfinite(v.grad).all()

    @pytest.mark.parametrize("k", KAPPA_VALUES)
    def test_log_0_backward_at_origin(self, k):
        mfd = _make_manifold(kappa=k)
        v = torch.zeros(3, mfd.n, dtype=torch.float64, requires_grad=True)
        y = mfd.exp_0(v)  # all rows = origin
        out = mfd.log_0(y)
        out.sum().backward()
        assert torch.isfinite(v.grad).all()

    @pytest.mark.parametrize("k", KAPPA_VALUES)
    def test_full_nll_chain_backward(self, k):
        """Substrate-style chain: exp_0 → all-pairs distance → NLL.
        Includes d(x_i, x_i) = 0 self-pairs in the partition function."""
        mfd = _make_manifold(kappa=k, n=4)
        v = (torch.randn(8, 4, dtype=torch.float64,
                          generator=_seed(41)) * 0.2)
        v.requires_grad_(True)
        x = mfd.exp_0(v)
        N = x.shape[0]
        Xi = x.unsqueeze(1).expand(N, N, mfd.n).reshape(-1, mfd.n)
        Xj = x.unsqueeze(0).expand(N, N, mfd.n).reshape(-1, mfd.n)
        d = mfd.distance(Xi, Xj).reshape(N, N)
        log_partition = torch.logsumexp(-d, dim=-1)
        loss = log_partition.sum()
        loss.backward()
        assert torch.isfinite(v.grad).all(), (
            f"κ={k}: NaN {torch.isnan(v.grad).sum().item()}"
        )


# --------------------------------------------------------------------
# Provenance roundtrip
# --------------------------------------------------------------------


class TestProvenance:
    def test_signature_roundtrip(self):
        mfd = KappaStereographicManifold(n=5, kappa=-0.7,
                                          dtype=torch.float32)
        sig = mfd._provenance_signature()
        mfd2 = KappaStereographicManifold._from_signature(sig)
        assert mfd2.n == mfd.n
        assert mfd2.kappa == mfd.kappa
        assert mfd2._branch == mfd._branch
        assert mfd2.dtype == mfd.dtype

    def test_record_in_context(self):
        from holonomy_lib import provenance

        mfd = _make_manifold(kappa=-1.0)
        with provenance.record() as reg:
            x = mfd.random_point(batch_size=2, generator=_seed(60))
            y = mfd.random_point(batch_size=2, generator=_seed(61))
            _ = mfd.distance(x, y)
        ops = {n.op_id for n in reg}
        assert (
            "holonomy_lib.manifolds.KappaStereographicManifold.distance"
            in ops
        )


# --------------------------------------------------------------------
# Comparison against geoopt's Stereographic (when installed)
# --------------------------------------------------------------------


try:
    import geoopt as _geoopt  # noqa: F401
    _HAVE_GEOOPT = True
except ImportError:
    _HAVE_GEOOPT = False


@pytest.mark.skipif(
    not _HAVE_GEOOPT,
    reason="geoopt not installed; install with `uv pip install geoopt`",
)
class TestAgainstGeoopt:
    """Cross-check distance / exp_0 / log_0 against geoopt's
    `Stereographic`. Convention mapping is direct: both libraries
    use the same κ-stereographic formulas (Bachmann et al. 2020) and
    the same parameter sign (κ > 0 spherical, κ < 0 hyperbolic).
    """

    @staticmethod
    def _geoopt_mfd(k: float):
        from geoopt import Stereographic
        return Stereographic(k=k)

    @pytest.mark.parametrize("k", [-1.0, -0.5, 0.5, 1.0])
    def test_distance_matches_geoopt(self, k):
        mfd = _make_manifold(kappa=k)
        gmfd = self._geoopt_mfd(k)
        x = mfd.random_point(batch_size=4, generator=_seed(70))
        y = mfd.random_point(batch_size=4, generator=_seed(71))
        d_ours = mfd.distance(x, y)
        d_geoopt = gmfd.dist(x, y)
        torch.testing.assert_close(d_ours, d_geoopt, atol=1e-7, rtol=1e-7)

    @pytest.mark.parametrize("k", [-1.0, -0.5, 0.5, 1.0])
    def test_exp_0_matches_geoopt(self, k):
        mfd = _make_manifold(kappa=k)
        gmfd = self._geoopt_mfd(k)
        v = torch.randn(3, mfd.n, dtype=mfd.dtype,
                        generator=_seed(72)) * 0.1
        out_ours = mfd.exp_0(v)
        out_geoopt = gmfd.expmap0(v)
        torch.testing.assert_close(out_ours, out_geoopt,
                                    atol=1e-7, rtol=1e-7)

    @pytest.mark.parametrize("k", [-1.0, -0.5, 0.5, 1.0])
    def test_log_0_matches_geoopt(self, k):
        mfd = _make_manifold(kappa=k)
        gmfd = self._geoopt_mfd(k)
        # Generate y inside the domain via geoopt's own exp
        v = torch.randn(3, mfd.n, dtype=mfd.dtype,
                        generator=_seed(73)) * 0.1
        y = gmfd.expmap0(v)
        out_ours = mfd.log_0(y)
        out_geoopt = gmfd.logmap0(y)
        torch.testing.assert_close(out_ours, out_geoopt,
                                    atol=1e-7, rtol=1e-7)


# --------------------------------------------------------------------
# Autograd stress — chained operations beyond simple boundary cases.
# Same suite as `LorentzManifold` to confirm the κ-stereographic
# branches are equally autograd-clean.
# --------------------------------------------------------------------


@pytest.mark.parametrize("k", KAPPA_VALUES)
class TestAutogradStress:
    def test_long_chained_exp_backward(self, k):
        mfd = _make_manifold(kappa=k, n=4)
        v = (torch.randn(3, 4, dtype=torch.float64,
                          generator=_seed(100)) * 0.2)
        v.requires_grad_(True)
        x = mfd.exp_0(v)
        for step in range(5):
            v_step = mfd.projection(
                x, torch.randn(3, 4, dtype=torch.float64,
                                generator=_seed(110 + step)) * 0.01,
            )
            x = mfd.exp(x, v_step)
        loss = mfd.distance(x, x[0:1].expand_as(x)).sum()
        loss.backward()
        assert torch.isfinite(v.grad).all()

    def test_mobius_inverse_backward(self, k):
        """mobius_add(x, -x) = 0; backward must be finite."""
        mfd = _make_manifold(kappa=k, n=4)
        v = (torch.randn(3, 4, dtype=torch.float64,
                          generator=_seed(120)) * 0.2)
        v.requires_grad_(True)
        x = mfd.exp_0(v)
        out = mfd.mobius_add(x, -x)
        out.sum().backward()
        assert torch.isfinite(v.grad).all()

    def test_exp_log_round_trip_backward(self, k):
        """exp_x(log_x(y)) = y; backward of the round-trip identity."""
        mfd = _make_manifold(kappa=k, n=4)
        v = (torch.randn(3, 4, dtype=torch.float64,
                          generator=_seed(121)) * 0.2)
        v.requires_grad_(True)
        x = mfd.exp_0(v)
        y = mfd.exp_0(torch.randn(3, 4, dtype=torch.float64,
                                    generator=_seed(122)) * 0.2)
        out = mfd.exp(x, mfd.log(x, y))
        loss = (out - y).pow(2).sum()
        loss.backward()
        assert torch.isfinite(v.grad).all()
