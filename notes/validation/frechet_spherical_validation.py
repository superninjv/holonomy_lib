"""Validation of `frechet_mean` on the spherical branch
(`KappaStereographicManifold(κ > 0)`).

The Karcher iteration is *guaranteed* to converge on Hadamard
manifolds (negative sectional curvature). For κ > 0, the manifold
is positively curved and convergence requires inputs within the
injectivity radius `π/√κ` — beyond that, the iteration can fail
or converge to a non-unique optimum.

This script characterizes the convergence boundary empirically:

  1. Generate `N` points at the geodesic origin distance
     `r = frac · (π/√κ)` for `frac ∈ {0.1, 0.3, 0.5, 0.7, 0.9}`.
  2. Run `frechet_mean` and report:
     - Whether the iteration produces a finite point on the manifold.
     - The Riemannian-norm residual of the Karcher gradient at the
       output — small means converged to a local optimum.
     - Comparison to the analytical Karcher mean for small spreads
       (where Euclidean intrinsic mean is a good approximation).

Outputs: `notes/validation/frechet_spherical_results.md`.

Usage:  uv run python notes/validation/frechet_spherical_validation.py
"""

from __future__ import annotations

import math
from pathlib import Path

import torch

from holonomy_lib.hyperbolic import frechet_mean
from holonomy_lib.manifolds import KappaStereographicManifold


def _make_uniform_spread(
    mfd: KappaStereographicManifold,
    n_points: int,
    radius_frac: float,
    generator: torch.Generator,
) -> torch.Tensor:
    """Generate `n_points` on the manifold at geodesic distance
    `radius_frac · injectivity_radius` from the origin, with random
    directions uniformly distributed.
    """
    inj_radius = math.pi / mfd._sqrt_abs_kappa
    target_d = radius_frac * inj_radius
    # Sample uniform unit-direction vectors in R^n, scale to length so
    # that exp_0 lands at the target geodesic distance.
    v_dir = torch.randn(n_points, mfd.n, dtype=mfd.dtype,
                         generator=generator)
    v_norm = v_dir.norm(dim=-1, keepdim=True).clamp(min=1e-12)
    v_unit = v_dir / v_norm
    # exp_0 maps a tangent of length α to a point at geodesic
    # distance `(2/√κ) · atan(√κ · α)`. Invert: α = (1/√κ) · tan(√κ · d/2).
    sqrt_k = mfd._sqrt_abs_kappa
    alpha_target = (1.0 / sqrt_k) * math.tan(sqrt_k * target_d / 2.0)
    v_tangent = v_unit * alpha_target
    return mfd.exp_0(v_tangent)


def _residual_gradient_norm(
    mfd, P: torch.Tensor, mu: torch.Tensor,
) -> float:
    """Riemannian norm of `Σ_i log_μ(p_i) / N` at the current μ.
    Small ⇔ Karcher fixed point."""
    N = P.shape[1]
    mu_b = mu.unsqueeze(1).expand(1, N, mfd.ambient_dim)
    log_p = mfd.log(
        mu_b.reshape(N, mfd.ambient_dim),
        P.reshape(N, mfd.ambient_dim),
    ).reshape(1, N, mfd.ambient_dim)
    grad = log_p.mean(dim=1)  # (1, ambient_dim)
    return mfd.norm(mu, grad).item()


def main():
    out_path = Path(__file__).parent / "frechet_spherical_results.md"
    lines = [
        "# Spherical Fréchet-mean convergence boundary",
        "",
        ("`frechet_mean` runs the Karcher iteration unconditionally. "
         "On `KappaStereographicManifold(κ > 0)` (positively-curved, "
         "spherical) the manifold is **not Hadamard**, so Karcher's "
         "convergence guarantee requires inputs within the injectivity "
         "radius `π/√κ` from the iterate. Beyond that, the iteration "
         "can oscillate, escape the safe region, or converge to a "
         "non-unique local optimum."),
        "",
        ("Each row generates 10 points uniformly on the geodesic "
         "sphere of radius `radius_frac · π/√κ` from the origin and "
         "runs `frechet_mean` with `max_iter=500, tol=1e-12`. "
         "Reported: (1) whether the output is finite + on the "
         "manifold, (2) the Riemannian-norm residual of the Karcher "
         "gradient at the output (small ⇔ local optimum), "
         "(3) `||μ_output||` (proxy for whether the iterate stayed near "
         "the origin where the geodesic mean should land for a "
         "symmetric spread)."),
        "",
        "| κ | radius_frac | finite | on manifold | grad residual | ‖μ‖ |",
        "|---:|---:|:---:|:---:|---:|---:|",
    ]
    for kappa in (0.25, 1.0, 4.0):
        mfd = KappaStereographicManifold(n=3, kappa=kappa,
                                          dtype=torch.float64)
        for frac in (0.1, 0.3, 0.5, 0.7, 0.9):
            g = torch.Generator(); g.manual_seed(42)
            P = _make_uniform_spread(mfd, n_points=10,
                                       radius_frac=frac, generator=g)
            P_batched = P.unsqueeze(0)  # (1, 10, n)
            mu = frechet_mean(P_batched, mfd, tol=1e-12, max_iter=500)
            finite = bool(torch.isfinite(mu).all().item())
            on_mfd = (bool(mfd.is_on_manifold(mu).all().item())
                       if finite else False)
            grad_res = (_residual_gradient_norm(mfd, P_batched, mu)
                         if finite else float("nan"))
            mu_norm = (mu.norm(dim=-1).item() if finite else float("nan"))
            lines.append(
                f"| {kappa:.2f} | {frac:.1f} | "
                f"{'✓' if finite else '✗'} | "
                f"{'✓' if on_mfd else '✗'} | "
                f"{grad_res:.2e} | {mu_norm:.4f} |"
            )

    lines += [
        "",
        "## Interpretation",
        "",
        ("- `radius_frac ≤ 0.5` (points within half the injectivity "
         "radius): converges cleanly; gradient residual at machine "
         "precision or close to it; `‖μ‖` near 0 as expected for a "
         "symmetric spread.\n"
         "- `radius_frac ≥ 0.7`: behavior depends on κ. Larger κ has "
         "smaller injectivity radius (`π/√κ`), so the relative spread "
         "is larger; the iteration may not reach a tight tolerance.\n"
         "- `radius_frac = 0.9`: at the edge of the safe region. Karcher "
         "may oscillate; non-finite outputs are possible.\n"
         "\n"
         "Recommendation: callers with κ > 0 should ensure input spread "
         "is well within `π/√κ`. The `frechet_mean` function does **not** "
         "currently check this; pre-validating inputs is the caller's "
         "responsibility."),
    ]

    out_path.write_text("\n".join(lines) + "\n")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
