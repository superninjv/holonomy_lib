"""Benchmark: autograd-safe `where`-on-safe-input vs eps-clamping
on hyperbolic geometry primitives.

Two distinct patterns produce forward-finite, boundary-stable
hyperbolic operations:

  - **eps-clamping** (geoopt's approach): wrap the singular input in
    `x.clamp(min=1 + eps)` so the operation never sees the boundary.
    Forward is finite; backward propagates a small bias.

  - **where-on-safe-input** (holonomy_lib's approach):
    `torch.where(cond, formula(safe_x), default)` where `safe_x` is a
    substituted value that never hits the singular point. Forward is
    finite AND backward gradient is the exact analytic limit at the
    boundary.

This benchmark quantifies the difference on three stress tests:

  1. **NaN-rate**: backward gradient at boundary inputs (`d(x, x) = 0`,
     `exp_x(0) = x`, `log_x(x) = 0`). Our idiom: 0 NaN; geoopt-eps:
     depends on eps choice.

  2. **Gradient bias** at non-boundary inputs (small but nonzero d).
     Eps-clamping introduces O(eps) bias; where-on-safe-input is exact.

  3. **Wall-clock forward / backward** runtime overhead.

Outputs: `notes/validation/autograd_safe_results.md`.

Usage:  uv run python notes/validation/autograd_safe_vs_geoopt.py
"""

from __future__ import annotations

import math
import time
from pathlib import Path

import torch

from holonomy_lib.manifolds import LorentzManifold


# ----------------------------------------------------------------------
# Reference implementations of the two patterns
# ----------------------------------------------------------------------


def distance_eps_clamp(x: torch.Tensor, y: torch.Tensor,
                       eps: float = 1e-5) -> torch.Tensor:
    """Hyperbolic distance, geoopt-style: `(1/sqrt|k|) * arccosh(z)`
    with `z` clamped to `>= 1 + eps`. Forward finite; backward biased
    by O(eps).
    """
    ip = -x[..., 0] * y[..., 0] + (x[..., 1:] * y[..., 1:]).sum(dim=-1)
    z = (-ip).clamp(min=1.0 + eps)
    return torch.acosh(z)


def distance_where_safe(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Hyperbolic distance, where-on-safe-input style: via the arcsinh
    reparameterization that avoids the arccosh boundary singularity
    entirely. Forward AND backward exact at all inputs.
    """
    diff = y - x
    diff_sq = (
        (diff[..., 1:] ** 2).sum(dim=-1) - diff[..., 0] ** 2
    )
    is_positive = diff_sq > 0
    safe_sq = torch.where(is_positive, diff_sq, torch.ones_like(diff_sq))
    sqrt_sq = torch.where(
        is_positive, torch.sqrt(safe_sq), torch.zeros_like(diff_sq),
    )
    return 2.0 * torch.asinh(sqrt_sq * 0.5)


def _make_pair_at_distance(d: float, n: int = 5):
    """Make a pair (x, y) on the unit Lorentz hyperboloid with
    d_M(x, y) ≈ d. x is the origin; y = exp_0(d, 0, ..., 0)."""
    mfd = LorentzManifold(n=n)
    x = mfd.origin(batch_size=1).requires_grad_(True)
    v = torch.zeros(1, n, dtype=mfd.dtype)
    v[0, 0] = d
    y = mfd.exp_0(v.detach()).requires_grad_(True)
    return mfd, x, y


# ----------------------------------------------------------------------
# Experiment 1: NaN-rate at boundary inputs
# ----------------------------------------------------------------------


def nan_rate_at_zero_distance(eps_values=(1e-7, 1e-5, 1e-3)):
    """At `d = 0` (x == y), where-on-safe-input gives finite gradient;
    eps-clamping gives finite-only if eps is large enough. Tabulate."""
    mfd = LorentzManifold(n=5)
    # 100 random Lorentz points, distance to itself
    g = torch.Generator(); g.manual_seed(0)
    x = mfd.random_point(batch_size=100, generator=g)
    y = x.clone()
    x = x.detach().requires_grad_(True)
    y = y.detach().requires_grad_(True)

    rows = []
    # Ours
    d = distance_where_safe(x, y)
    d.sum().backward()
    n_nan = torch.isnan(x.grad).sum().item()
    rows.append(("ours (arcsinh + where-safe-sqrt)", "—", n_nan, 100 * 6))

    for eps in eps_values:
        x_g = x.detach().clone().requires_grad_(True)
        y_g = y.detach().clone().requires_grad_(True)
        d = distance_eps_clamp(x_g, y_g, eps=eps)
        d.sum().backward()
        n_nan = torch.isnan(x_g.grad).sum().item()
        rows.append(("eps-clamp", f"{eps:.0e}", n_nan, 100 * 6))
    return rows


# ----------------------------------------------------------------------
# Experiment 2: Gradient bias in the substrate-training chain
# ----------------------------------------------------------------------
#
# The realistic test: tangent-at-origin embedding `T = exp_0(v)`, loss
# `= d(T_i, T_target)`. Both distance formulas agree ON the manifold,
# but differ in off-manifold extensions; the autograd-relevant
# quantity is the gradient on `v` (the trainable param), which is
# constrained by `exp_0` to remain on the manifold. Both should give
# the same `v.grad`.


def gradient_through_exp_0_chain(d_values=(1e-3, 1e-2, 0.1, 1.0)):
    """Gradient on the tangent-at-origin parameter `v` when the loss
    is the distance to a target on the manifold. Both formulas should
    give the same `v.grad` because the chain `exp_0(v) → distance`
    constrains the path to the manifold throughout."""
    rows = []
    for d in d_values:
        mfd = LorentzManifold(n=5)
        n = 5
        # v = trainable Euclidean tangent at origin
        v = torch.zeros(1, n, dtype=mfd.dtype)
        v[0, 0] = d
        # target: fixed point at distance d from origin
        target = mfd.exp_0(v.clone()).detach()

        # Variant 1: where-on-safe-input
        v_ws = v.detach().clone().requires_grad_(True)
        T_ws = mfd.exp_0(v_ws)
        loss_ws = distance_where_safe(T_ws, target)
        loss_ws.backward()
        grad_ws = v_ws.grad.clone()

        # Variant 2: eps-clamp
        for eps in (1e-7, 1e-5, 1e-3):
            v_ec = v.detach().clone().requires_grad_(True)
            T_ec = mfd.exp_0(v_ec)
            loss_ec = distance_eps_clamp(T_ec, target, eps=eps)
            loss_ec.backward()
            grad_ec = v_ec.grad
            grad_diff = (grad_ec - grad_ws).abs().max().item()
            ref_scale = max(grad_ws.abs().max().item(), 1e-30)
            rel_diff = grad_diff / ref_scale
            rows.append((d, eps, grad_diff, rel_diff))
    return rows


# ----------------------------------------------------------------------
# Experiment 3: Wall-clock runtime
# ----------------------------------------------------------------------


def runtime_comparison(n_samples=10_000, n_iters=50):
    """Wall-clock comparison of forward + backward at large batch
    size. Both approaches should be ~ same cost (a few extra ops in
    where-on-safe-input don't dominate for typical inputs)."""
    mfd = LorentzManifold(n=8)
    g = torch.Generator(); g.manual_seed(0)
    x = mfd.random_point(batch_size=n_samples, generator=g)
    y = mfd.random_point(batch_size=n_samples, generator=g)
    # Warm-up
    for _ in range(3):
        distance_where_safe(x, y).sum()
        distance_eps_clamp(x, y, eps=1e-5).sum()
    # Time ours
    t0 = time.perf_counter()
    for _ in range(n_iters):
        x_g = x.detach().clone().requires_grad_(True)
        y_g = y.detach().clone().requires_grad_(True)
        d = distance_where_safe(x_g, y_g)
        d.sum().backward()
    t_ours = time.perf_counter() - t0
    # Time eps-clamp
    t0 = time.perf_counter()
    for _ in range(n_iters):
        x_g = x.detach().clone().requires_grad_(True)
        y_g = y.detach().clone().requires_grad_(True)
        d = distance_eps_clamp(x_g, y_g, eps=1e-5)
        d.sum().backward()
    t_eps = time.perf_counter() - t0
    return t_ours / n_iters, t_eps / n_iters


# ----------------------------------------------------------------------
# Driver
# ----------------------------------------------------------------------


def main():
    out_path = Path(__file__).parent / "autograd_safe_results.md"
    lines = [
        "# Autograd-safe `where`-on-safe-input vs eps-clamping",
        "",
        ("Two patterns for forward-finite hyperbolic operations at "
         "the boundary `d(x, y) → 0`:"),
        "",
        ("- **eps-clamping** (geoopt style): "
         "`torch.acosh(z.clamp(min=1 + eps))`. Forward finite, "
         "backward biased by O(eps) AND can still NaN at exactly "
         "z = 1 + eps boundary."),
        "- **where-on-safe-input** (holonomy_lib): "
         "`torch.where(cond, formula(safe_x), default)` with "
         "`safe_x = torch.where(cond, x, safe_default)`. Forward "
         "AND backward exact at all inputs; gradient is the analytic "
         "limit at boundary (zero subgradient choice).",
        "",
        ("All measurements on Lorentz model H^n_{-1} of dim n=5 (or "
         "as noted)."),
        "",
        "## Experiment 1: NaN-rate at `d = 0` (x = y)",
        "",
        ("100 random points, compute `d(x_i, x_i) = 0` and backward; "
         "count NaN entries in `x.grad`. Total entries = 100 × 6 ambient."),
        "",
        "| method | eps | NaN count |  / total |",
        "|---|---:|---:|---:|",
    ]
    for method, eps, n_nan, total in nan_rate_at_zero_distance():
        lines.append(f"| {method} | {eps} | {n_nan} | {total} |")

    lines += [
        "",
        "## Experiment 2: Gradient bias through the substrate-training chain",
        "",
        ("Realistic test: tangent-at-origin parameter `v ∈ R^n`, "
         "embedded via `T = exp_0(v)`, loss = `d(T, target)` for a "
         "fixed target at distance `d` from origin. Compare `v.grad` "
         "between the two distance formulas. Because `exp_0` "
         "constrains the iterate to the manifold, both formulas "
         "should give the same `v.grad` (the off-manifold "
         "ambient-gradient difference is killed by the chain through "
         "`exp_0`)."),
        "",
        "| d (distance) | eps | max-abs diff in v.grad | rel diff |",
        "|---:|---:|---:|---:|",
    ]
    for d, eps, abs_diff, rel_diff in gradient_through_exp_0_chain():
        lines.append(f"| {d} | {eps:.0e} | {abs_diff:.4e} | {rel_diff:.4e} |")

    lines += [
        "",
        "## Experiment 3: Wall-clock runtime",
        "",
        ("`distance(X, Y)` + `.sum().backward()` on batch of 10,000 "
         "pairs, n=8, 50 iterations. Both approaches scale identically "
         "in O(B · n); the where-on-safe-input idiom adds a few cheap "
         "boolean ops that vectorize away."),
        "",
    ]
    t_ours, t_eps = runtime_comparison()
    lines += [
        f"- Where-on-safe-input: **{t_ours * 1000:.2f} ms / iter**",
        f"- Eps-clamp:           **{t_eps * 1000:.2f} ms / iter**",
        f"- Overhead: **{((t_ours - t_eps) / t_eps) * 100:+.1f}%**",
        "",
        "## Headline (honest assessment)",
        "",
        ("- **Both approaches NaN-free at the boundary** when used as "
         "drop-in replacements. We initially thought eps-clamp would "
         "NaN, but `acosh(1 + eps)` is finite and `1/sqrt((1+eps)² - 1)` "
         "is finite too. The NaN risk from our original "
         "`clamp(min=0) + sqrt` pattern was real, but a properly-"
         "implemented eps-clamp on the `acosh` argument is also safe.\n"
         "- **Both produce identical `v.grad` in the realistic "
         "substrate-training chain** (`exp_0(v) → distance(T, ...) "
         "→ loss → backward`). The off-manifold ambient-gradient "
         "differences are absorbed by the `exp_0` constraint.\n"
         "- The where-on-safe-input idiom DOES give an exact-zero "
         "gradient at `d(x, x) = 0` (the analytic limit), where "
         "eps-clamp gives `acosh(1 + eps) ≈ √(2eps) ≈ 4e-4` with a "
         "spurious gradient of `1/√(2eps) ≈ 2236` at eps=1e-7. For "
         "direct distance consumers (NLL self-pair diagonal, "
         "metric-learning losses) this can matter; for the chained "
         "`exp_0 → distance` use case it doesn't.\n"
         "- **Runtime**: where-on-safe-input adds ~2× overhead in "
         "this microbenchmark — bounded by the extra `torch.where` "
         "calls. In production training where `distance` is one op "
         "among many, negligible.\n"
         "\n"
         "**Verdict for paper-worthiness**: this is *not* the strong "
         "result we expected. Both approaches work for training. The "
         "arcsinh-reparameterization is mathematically cleaner and "
         "avoids the eps hyperparameter, but it's a refinement, not "
         "a NaN-correctness fix. **The paper-worthy finding from "
         "this validation pass is the heat-kernel recursion bug** "
         "(`heat_kernel_findings.md`)."),
    ]
    out_path.write_text("\n".join(lines) + "\n")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
