"""Validation of `hyperbolic_heat_kernel` for n ∈ {1, 2, 3, 5, 7, 9}.

Two independent checks per (n, t, d):

1. **Heat-equation residual**: the heat kernel must solve the
   radial heat equation on H^n. We compute

       residual(t, r) = ∂_t k - Δ_radial k
                      = (k(t+dt, r) - k(t-dt, r)) / (2·dt)
                        - [ ∂_r² k + (n-1)·coth(r)·∂_r k ]

   and report the relative residual `|residual| / |∂_t k|`. A
   correct heat kernel has `residual → 0`; we tabulate the residual
   over (n, t, d) to identify the safe-d regime.

2. **Probability mass**: the heat kernel is a probability density,
   so

       ∫_{H^n} k_t(d) · dV(d) = ω_{n-1} · ∫_0^∞ k_t(r) · sinh^{n-1}(r) dr = 1

   where `ω_{n-1}` is the surface area of the unit (n-1)-sphere.
   Compute via Gauss-Legendre on [0, R_max], report the deviation
   from 1.

Outputs: a markdown table at `notes/validation/heat_kernel_results.md`.

Usage:  uv run python notes/validation/heat_kernel_validation.py
"""

from __future__ import annotations

import math
from pathlib import Path

import torch
from scipy.special import gamma, roots_legendre

from holonomy_lib.hyperbolic import hyperbolic_heat_kernel
from holonomy_lib.manifolds import LorentzManifold


def _heat_equation_residual(n: int, t: float, r: float, dt: float, dr: float):
    """Compute relative residual of `∂_t k - Δ_radial k` at (t, r) for H^n.

    Uses central finite differences in both t and r. The residual is
    normalized by `|∂_t k|` to give a dimensionless relative error.
    """
    mfd = LorentzManifold(n=n)
    t_tensor = torch.tensor(t, dtype=torch.float64)
    r_tensor = torch.tensor(r, dtype=torch.float64)
    # ∂_t k via central difference
    k_plus_t = hyperbolic_heat_kernel(t_tensor + dt, r_tensor, mfd).item()
    k_minus_t = hyperbolic_heat_kernel(t_tensor - dt, r_tensor, mfd).item()
    dk_dt = (k_plus_t - k_minus_t) / (2.0 * dt)
    # k at (t, r) and (t, r ± dr)
    k_0 = hyperbolic_heat_kernel(t_tensor, r_tensor, mfd).item()
    k_plus_r = hyperbolic_heat_kernel(
        t_tensor, r_tensor + dr, mfd,
    ).item()
    k_minus_r = hyperbolic_heat_kernel(
        t_tensor, r_tensor - dr, mfd,
    ).item()
    # ∂_r k and ∂_r² k via central difference
    dk_dr = (k_plus_r - k_minus_r) / (2.0 * dr)
    d2k_dr2 = (k_plus_r - 2.0 * k_0 + k_minus_r) / (dr * dr)
    # Δ_radial k = ∂_r² k + (n-1) · coth(r) · ∂_r k
    coth_r = math.cosh(r) / math.sinh(r)
    lap_k = d2k_dr2 + (n - 1) * coth_r * dk_dr
    residual = dk_dt - lap_k
    # Relative — normalize by max(|dk_dt|, |lap_k|) to avoid 0/0
    scale = max(abs(dk_dt), abs(lap_k), 1e-300)
    return abs(residual) / scale


def _probability_mass(n: int, t: float, n_quad: int = 200,
                     R_max: float | None = None):
    """Numerically integrate `k_t(r) · sinh^{n-1}(r)` from 0 to R_max
    via Gauss-Legendre quadrature, then multiply by the surface area
    of S^{n-1}: ω_{n-1} = 2 · π^{n/2} / Γ(n/2).

    `R_max` defaults to `((n-1)/2) · t · 4 + 5·sqrt(t) + 5` — the
    drift-speed for the spectral-bottom mode times t plus a Gaussian
    spread plus margin. Captures the bulk of the kernel at any
    practical (n, t).
    """
    if R_max is None:
        spectral_drift = ((n - 1) / 2.0) * t
        gauss_spread = 5.0 * math.sqrt(t)
        R_max = 4.0 * spectral_drift + gauss_spread + 5.0
    mfd = LorentzManifold(n=n)
    # Gauss-Legendre nodes on [-1, 1] → rescale to [0, R_max]
    nodes_np, weights_np = roots_legendre(n_quad)
    nodes = 0.5 * R_max * (nodes_np + 1.0)
    weights = 0.5 * R_max * weights_np
    r_t = torch.from_numpy(nodes).double()
    t_t = torch.tensor(t, dtype=torch.float64)
    k = hyperbolic_heat_kernel(t_t, r_t, mfd).numpy()
    sinh_n = torch.sinh(r_t).numpy() ** (n - 1)
    integral = (weights * k * sinh_n).sum()
    # Surface area of S^{n-1}
    omega_n_minus_1 = 2.0 * math.pi ** (n / 2.0) / gamma(n / 2.0)
    return omega_n_minus_1 * integral


def main():
    """Run the validation and write the report."""
    out_path = Path(__file__).parent / "heat_kernel_results.md"
    lines = [
        "# Heat kernel validation",
        "",
        ("`hyperbolic_heat_kernel` checked against two independent "
         "consistency properties: heat-equation residual and "
         "probability mass."),
        "",
        "## (1) Heat-equation residual",
        "",
        ("For each `(n, t, r)`, we compute "
         "`|∂_t k - Δ_radial k| / max(|∂_t k|, |Δ_radial k|)` via "
         "central finite differences (`dt = dr = 1e-5`). A correct "
         "kernel has residual → 0; finite-difference noise floor is "
         "~ `eps_double / dr² ≈ 1e-6` for second derivatives."),
        "",
        ("Below the noise floor (relative residual ≲ 1e-6 at this "
         "discretization) the kernel is consistent with the heat "
         "equation. Larger residuals would indicate either (a) bugs "
         "in our recursion or (b) noise amplification in the "
         "`1/sinh(d)` denominator for small `r`."),
        "",
        "| n | t | r | residual |",
        "|---:|---:|---:|---:|",
    ]

    dt = 1e-5
    dr = 1e-5
    config_pde = [
        (n, t, r)
        for n in (1, 2, 3, 4, 5, 6, 7, 9)
        for t in (0.1, 0.5, 1.0, 2.0)
        for r in (0.2, 0.5, 1.0, 2.0, 4.0)
    ]
    for n, t, r in config_pde:
        try:
            res = _heat_equation_residual(n, t, r, dt, dr)
            res_str = f"{res:.2e}"
        except Exception as exc:
            res_str = f"ERROR: {exc}"
        lines.append(f"| {n} | {t} | {r} | {res_str} |")

    # ----- Probability mass -----
    lines += [
        "",
        "## (2) Probability mass",
        "",
        ("For each `(n, t)`, we numerically integrate "
         "`k_t(r) · sinh^{n-1}(r)` over `[0, 10]` via 100-point "
         "Gauss-Legendre, then multiply by the surface area of "
         "`S^{n-1}`. The result should be ≈ 1 for any `t > 0`."),
        "",
        "| n | t | mass | error |",
        "|---:|---:|---:|---:|",
    ]
    config_mass = [
        (n, t)
        for n in (1, 2, 3, 4, 5, 6, 7, 9)
        for t in (0.1, 0.5, 1.0, 2.0, 5.0)
    ]
    for n, t in config_mass:
        try:
            mass = _probability_mass(n, t)
            err = abs(mass - 1.0)
            lines.append(f"| {n} | {t} | {mass:.6f} | {err:.2e} |")
        except Exception as exc:
            lines.append(f"| {n} | {t} | ERROR: {exc} | |")

    out_path.write_text("\n".join(lines) + "\n")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
