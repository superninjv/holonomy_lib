"""Tests for holonomy_lib.hyperbolic.hyperbolic_heat_kernel.

Layers:
  1. Closed-form sanity for n=1, 3.
  2. Integral form for n=2 (probability mass, symmetry).
  3. Recursion identity: k^{n+2} = -(2π sinh d)^{-1} ∂_d k^n —
     verified numerically by finite differences.
  4. Curvature scaling: k_K(t, d) = K^{n/2} · k_1(K·t, √K·d).
"""

from __future__ import annotations

import math

import pytest
import torch

from holonomy_lib.hyperbolic import hyperbolic_heat_kernel
from holonomy_lib.manifolds import LorentzManifold


# --------------------------------------------------------------------
# Sanity at boundary / closed-form values
# --------------------------------------------------------------------


def test_n1_is_gaussian():
    """H^1 heat kernel is the standard Gaussian on R."""
    mfd = LorentzManifold(n=1)
    t = torch.tensor(0.5, dtype=torch.float64)
    d = torch.linspace(0, 3.0, 7, dtype=torch.float64)
    k = hyperbolic_heat_kernel(t, d, mfd)
    expected = (4.0 * math.pi * 0.5) ** -0.5 \
        * torch.exp(-d * d / (4.0 * 0.5))
    torch.testing.assert_close(k, expected, atol=1e-12, rtol=1e-12)


def test_n3_at_zero_distance():
    """k^3_t(0) = (4πt)^{-3/2} · exp(-t) · 1  (the d/sinh d → 1 limit)."""
    mfd = LorentzManifold(n=3)
    t_vals = torch.tensor([0.1, 0.5, 1.0, 2.0], dtype=torch.float64)
    d = torch.zeros_like(t_vals)
    k = hyperbolic_heat_kernel(t_vals, d, mfd)
    expected = (4.0 * math.pi * t_vals) ** -1.5 * torch.exp(-t_vals)
    torch.testing.assert_close(k, expected, atol=1e-12, rtol=1e-12)


def test_n3_davies_mandouvalos_closed_form():
    """k^3_t(d) = (4πt)^{-3/2} · exp(-t - d²/4t) · d/sinh d."""
    mfd = LorentzManifold(n=3)
    t = torch.tensor(0.7, dtype=torch.float64)
    d = torch.tensor([0.1, 0.5, 1.0, 2.0], dtype=torch.float64)
    k = hyperbolic_heat_kernel(t, d, mfd)
    expected = (
        (4.0 * math.pi * t) ** -1.5
        * torch.exp(-t - d * d / (4.0 * t))
        * d / torch.sinh(d)
    )
    torch.testing.assert_close(k, expected, atol=1e-12, rtol=1e-12)


# --------------------------------------------------------------------
# Symmetry and positivity
# --------------------------------------------------------------------


def test_kernel_positive():
    """Heat kernel is a probability density — positive everywhere."""
    mfd = LorentzManifold(n=2)
    t = torch.tensor(1.0, dtype=torch.float64)
    d = torch.linspace(0.1, 3.0, 8, dtype=torch.float64)
    k = hyperbolic_heat_kernel(t, d, mfd)
    assert (k >= 0).all()
    assert torch.isfinite(k).all()


@pytest.mark.parametrize("n", [1, 2, 3, 5])
def test_kernel_decays_with_distance(n):
    """At fixed t, the kernel decreases as d grows (for d >> √t)."""
    mfd = LorentzManifold(n=n)
    t = torch.tensor(0.5, dtype=torch.float64)
    d = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)
    k = hyperbolic_heat_kernel(t, d, mfd)
    # Monotonically decreasing
    assert (k[1] < k[0]) and (k[2] < k[1])


# --------------------------------------------------------------------
# Curvature scaling
# --------------------------------------------------------------------


def test_curvature_scaling_n3():
    """k^n_{−K, t}(d) = K^{n/2} · k^n_{−1, K·t}(√K · d)."""
    K = 2.0
    mfd_unit = LorentzManifold(n=3, k=-1.0)
    mfd_k = LorentzManifold(n=3, k=-K)
    t = torch.tensor(0.5, dtype=torch.float64)
    d = torch.tensor([0.5, 1.0, 1.5], dtype=torch.float64)
    k_curved = hyperbolic_heat_kernel(t, d, mfd_k)
    k_unit_scaled = (
        K ** 1.5
        * hyperbolic_heat_kernel(K * t, math.sqrt(K) * d, mfd_unit)
    )
    torch.testing.assert_close(
        k_curved, k_unit_scaled, atol=1e-12, rtol=1e-12,
    )


# --------------------------------------------------------------------
# Recursion identity
# --------------------------------------------------------------------


def test_recursion_identity_n3_to_n5():
    """Numerical check: k^5_t(d) = -(2π sinh d)^{-1} · ∂_d k^3_t(d).

    Use a central finite difference on `k^3` to approximate the
    derivative; compare to the result from the autograd-recursion
    path used internally for n=5.
    """
    mfd_3 = LorentzManifold(n=3)
    mfd_5 = LorentzManifold(n=5)
    t = torch.tensor(0.5, dtype=torch.float64)
    d = torch.tensor([0.5, 1.0, 1.5], dtype=torch.float64)
    eps = 1e-5
    k_plus = hyperbolic_heat_kernel(t, d + eps, mfd_3)
    k_minus = hyperbolic_heat_kernel(t, d - eps, mfd_3)
    fd = (k_plus - k_minus) / (2.0 * eps)
    expected_n5 = -fd / (2.0 * math.pi * torch.sinh(d))
    actual_n5 = hyperbolic_heat_kernel(t, d, mfd_5)
    # Finite differences are O(eps²) accurate; 1e-6 absolute is fine.
    torch.testing.assert_close(actual_n5, expected_n5, atol=1e-6, rtol=1e-6)


# --------------------------------------------------------------------
# Integral path (n=2) self-consistency
# --------------------------------------------------------------------


def test_n2_quadrature_node_count_convergence():
    """Doubling quadrature nodes shouldn't change the result much
    once the integrand is well-resolved."""
    mfd = LorentzManifold(n=2)
    t = torch.tensor(0.5, dtype=torch.float64)
    d = torch.tensor(1.0, dtype=torch.float64)
    k_32 = hyperbolic_heat_kernel(t, d, mfd, n_quad=32)
    k_64 = hyperbolic_heat_kernel(t, d, mfd, n_quad=64)
    # Should agree to integration-error tolerance
    assert (k_32 - k_64).abs().max().item() < 1e-9


# --------------------------------------------------------------------
# Provenance
# --------------------------------------------------------------------


def test_provenance():
    from holonomy_lib import provenance

    mfd = LorentzManifold(n=3)
    t = torch.tensor(1.0, dtype=torch.float64)
    d = torch.tensor([0.5, 1.0], dtype=torch.float64)
    with provenance.record() as reg:
        _ = hyperbolic_heat_kernel(t, d, mfd)
    ops = {n.op_id for n in reg}
    assert "holonomy_lib.hyperbolic.hyperbolic_heat_kernel" in ops


# --------------------------------------------------------------------
# Autograd-finite — backward through the recursion at n ≥ 5
# --------------------------------------------------------------------


def test_heat_kernel_backward_n3():
    """Closed-form path (n=3): backward w.r.t. d must work."""
    mfd = LorentzManifold(n=3)
    t = torch.tensor(0.5, dtype=torch.float64)
    d = torch.tensor([0.5, 1.0, 1.5], dtype=torch.float64,
                      requires_grad=True)
    k = hyperbolic_heat_kernel(t, d, mfd)
    k.sum().backward()
    assert torch.isfinite(d.grad).all()


def test_heat_kernel_backward_n5_recursion():
    """Recursion path (n=5): the previous create_graph=False + detach()
    broke autograd through `distances`. With create_graph=True, the
    gradient should flow."""
    mfd = LorentzManifold(n=5)
    t = torch.tensor(0.5, dtype=torch.float64)
    d = torch.tensor([0.5, 1.0, 1.5], dtype=torch.float64,
                      requires_grad=True)
    k = hyperbolic_heat_kernel(t, d, mfd)
    k.sum().backward()
    assert d.grad is not None, "no gradient — recursion detached the graph"
    assert torch.isfinite(d.grad).all()
    # Gradient should be non-trivial (k decreases as d grows away
    # from 0, so ∂k/∂d < 0 for d > peak).
    assert (d.grad != 0).any()
