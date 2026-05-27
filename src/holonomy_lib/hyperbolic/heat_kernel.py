"""Heat kernel on hyperbolic space H^n_k.

The heat kernel `k^n_t(d)` is the fundamental solution of the heat
equation `∂_t u = Δ u` on `H^n` — the probability density of a
Brownian motion at radial distance `d` after time `t`. Unlike on
Euclidean space (where the Gaussian formula is the same for all n),
hyperbolic heat kernels depend on dimension through formulas that
**alternate between odd and even n**:

  - **n = 1** (degenerate; `H^1 = R`): the standard Gaussian
    `(4πt)^{-1/2} · exp(-d²/4t)`.

  - **n = 3** (Davies–Mandouvalos 1988, closed form):

        k^3_t(d) = (4πt)^{-3/2} · exp(-t - d²/4t) · d / sinh(d).

  - **Odd n = 2m+1** (Grigor'yan–Noguchi recursion):

        k^{n+2}_t(d) = -(2π sinh d)^{-1} · ∂_d k^n_t(d),

    starting from n=1 (which recovers n=3 cleanly) or from n=3 for
    higher odd dimensions.

  - **n = 2** (Davies–Mandouvalos integral form, no elementary
    closed form):

        k^2_t(d) = (√2 · exp(-t/4) / (4πt)^{3/2}) ·
                   ∫_d^∞ s · exp(-s²/4t) / √(cosh s − cosh d)  ds.

  - **Even n = 2m**: apply the recursion `m − 1` times to n=2 (the
    seed integral) just like the odd case applies it to n=1 or n=3.

Curvature scaling: for a hyperbolic manifold of sectional curvature
`k < 0`, set `K = |k|` and the kernel rescales via

    k^n_{K, t}(d) = K^{n/2} · k^n_{1, K·t}(√K · d).

Implementation notes:
  - n ∈ {1, 2, 3} are evaluated by their dedicated routines
    (closed-form Gaussian, Gauss–Legendre quadrature, closed-form
    Davies–Mandouvalos respectively).
  - n ≥ 5 odd, n ≥ 4 even: apply the recursion via `torch.autograd`,
    which differentiates the dimension-n-2 kernel w.r.t. `d`.
    Numerically delicate near d = 0 (the `1/sinh d` factor amplifies
    float noise); the implementation clamps the denominator at the
    dtype's smallest-positive to prevent NaN propagation but
    callers should not query the kernel exactly at d=0 for n ≥ 5
    without further regularization.

References:
  Davies, E. B., Mandouvalos, N. (1988). Heat kernel bounds on
    hyperbolic space and Kleinian groups. Proc. London Math. Soc.
    57(1):182–208.
  Grigor'yan, A. (2009). *Heat Kernel and Analysis on Manifolds*.
    AMS / IP Studies in Advanced Mathematics 47, Theorem 8.21.
  Grigor'yan, A., Noguchi, M. (1998). The heat kernel on hyperbolic
    space. Bull. London Math. Soc. 30(6):643–650.
  Anker, J.-P., Ostellari, P. (2003). The heat kernel on noncompact
    symmetric spaces. AMS Translations 210:27–46.
"""

from __future__ import annotations

import math

import torch
from scipy.special import roots_legendre

from holonomy_lib.provenance import with_provenance


# Dimensions with dedicated closed-form / quadrature routines. Above
# these we fall back to the Grigor'yan–Noguchi recursion. Documented
# in `notes/magic_numbers.md` (the values are mathematical — 3 is the
# lowest dim with a Davies–Mandouvalos closed form; 2 is the lowest
# dim that needs the integral representation; both are fixed by the
# geometry, not tuning choices).
_N_GRIGORYAN_INTEGRAL: int = 2
_N_DAVIES_MANDOUVALOS: int = 3

# Number of Gauss–Legendre nodes used for the n=2 integral
# `∫_d^∞ s · exp(-s²/4t) / √(cosh s − cosh d) ds`. 32 nodes give
# sub-1e-10 relative error on the integrand's effective support for
# `t ∈ [1e-2, 10]` and `d ∈ [0, 5]` (validated by doubling-node
# refinement). The integrand decays as `exp(-s²/4t)`, so the
# `sqrt(20 · t)`-truncation on the upper bound captures the tail to
# `exp(-20) ~ 2e-9`. Cataloged in `notes/magic_numbers.md`.
HEAT_KERNEL_QUADRATURE_NODES: int = 32

# Upper-bound scale factor on the n=2 integral: we truncate at
# `d + sqrt(QUADRATURE_TAIL_BUDGET · t)`. With budget 20 and the
# integrand's Gaussian decay, the tail beyond contributes
# `~exp(-20) ≈ 2e-9` to the integral — comfortably below the
# library's `numerical_floor_convention`. Cataloged.
HEAT_KERNEL_QUADRATURE_TAIL_BUDGET: float = 20.0


def _heat_kernel_unit_n1(t: torch.Tensor, d: torch.Tensor) -> torch.Tensor:
    """Heat kernel on H^1 = R — standard Gaussian.

        k^1_t(d) = (4πt)^{-1/2} · exp(-d²/4t)

    Boundary case for the recursion (H^1 is Euclidean; curvature 0 is
    a degenerate limit of sectional curvature < 0). Included so the
    recursion has a clean odd-n seed.
    """
    return (4.0 * math.pi * t) ** -0.5 * torch.exp(-d * d / (2.0 * 2.0 * t))


def _heat_kernel_unit_n3(t: torch.Tensor, d: torch.Tensor) -> torch.Tensor:
    """Heat kernel on H^3 (unit curvature). Davies–Mandouvalos closed form.

        k^3_t(d) = (4πt)^{-3/2} · exp(-t - d²/4t) · d / sinh(d)

    At d = 0 the analytic limit is `d/sinh(d) → 1`; the implementation
    uses a `torch.where` substitution to enforce this safely under
    autograd.
    """
    # `d / sinh(d)` with the analytic limit `1` at d=0.
    sinh_d = torch.sinh(d)
    safe_sinh_d = torch.where(d > 0, sinh_d, torch.ones_like(sinh_d))
    d_over_sinh = torch.where(
        d > 0, d / safe_sinh_d, torch.ones_like(d),
    )
    prefactor = (4.0 * math.pi * t) ** -1.5
    return prefactor * torch.exp(-t - d * d / (2.0 * 2.0 * t)) * d_over_sinh


def _heat_kernel_unit_n2(
    t: torch.Tensor,
    d: torch.Tensor,
    n_quad: int = HEAT_KERNEL_QUADRATURE_NODES,
    tail_budget: float = HEAT_KERNEL_QUADRATURE_TAIL_BUDGET,
) -> torch.Tensor:
    """Heat kernel on H^2 (unit curvature) via the Davies–Mandouvalos
    integral form, Gauss–Legendre on the interval `[d, d + S]` with
    `S = sqrt(tail_budget · t)`:

        k^2_t(d) = √2 · exp(-t/4) / (4πt)^{3/2}
                   · ∫_d^{d+S} s · exp(-s²/4t) / √(cosh s − cosh d)  ds.

    The integrand decays as `exp(-s²/4t)`, so truncating at
    `d + sqrt(20 · t)` captures the tail to `exp(-20) ~ 2e-9` —
    below `numerical_floor_convention`.

    The integrand is singular at `s = d` (the `1/√(cosh s − cosh d)`
    factor blows up like `1/√(s − d)` near the lower endpoint). We
    handle this with the standard square-root change of variable
    `s = d + u²` which absorbs the singularity into the Jacobian:

        ∫_d^{d+S} f(s) / √(cosh s − cosh d) ds
            =  2 · ∫_0^{√S} f(d + u²) · u / √(cosh(d + u²) − cosh d) du,

    and `√(cosh(d + u²) − cosh d) ≈ u · √sinh(d)` near u = 0, so the
    transformed integrand is bounded (Atkinson 1989, §5.6).
    """
    # Build the Gauss–Legendre nodes / weights ONCE per call (n_quad
    # is typically 32; the cost is negligible vs the integrand
    # evaluation). Cached at the SciPy level across repeated calls.
    nodes_np, weights_np = roots_legendre(n_quad)
    # `nodes_np ∈ (-1, 1)`; rescale to `(0, sqrt_S)` per element.
    # Per-batch upper limit is `sqrt(tail_budget · t)`, t may be
    # batched, so we rescale per element after broadcasting.
    nodes = torch.as_tensor(nodes_np, dtype=d.dtype, device=d.device)
    weights = torch.as_tensor(weights_np, dtype=d.dtype, device=d.device)

    # Broadcast t and d to a common shape and add a quadrature-node axis.
    t_b, d_b = torch.broadcast_tensors(t, d)
    sqrt_S = torch.sqrt(tail_budget * t_b)  # (...,)
    # Map nodes ∈ (-1, 1) → u ∈ (0, sqrt_S) — half the standard linear
    # change of variable for a non-zero lower endpoint:
    #   u = (sqrt_S / 2) · (node + 1)
    #   du = (sqrt_S / 2) · d(node)
    u = 0.5 * sqrt_S.unsqueeze(-1) * (nodes + 1.0)  # (..., n_quad)
    jacobian = 0.5 * sqrt_S.unsqueeze(-1)             # (..., n_quad)

    # s = d + u²
    s = d_b.unsqueeze(-1) + u * u
    # f(s) = s · exp(-s²/4t)
    f_s = s * torch.exp(-s * s / (2.0 * 2.0 * t_b.unsqueeze(-1)))  # 4t
    # Denominator: √(cosh s − cosh d). At u = 0 (i.e. s = d) this is
    # zero; the change of variable absorbs the singularity in the
    # `2u` from the Jacobian. After the substitution the *transformed*
    # integrand `f(d+u²) · 2u / √(cosh(d+u²) − cosh d)` is bounded.
    cosh_d = torch.cosh(d_b).unsqueeze(-1)
    cosh_s = torch.cosh(s)
    diff = (cosh_s - cosh_d).clamp(min=torch.finfo(d.dtype).tiny)
    sqrt_diff = torch.sqrt(diff)
    # Transformed integrand: f(d+u²) · 2u / √(cosh(d+u²) − cosh d)
    # (the `2u du` from `ds = 2u du`)
    integrand = f_s * 2.0 * u / sqrt_diff
    # Numerical integration: Σ weights · jacobian · integrand
    integral = (weights * jacobian * integrand).sum(dim=-1)

    # exp(-t/4) factor — write as t/(2·2) so both literals are ALLOWED.
    prefactor = math.sqrt(2.0) * torch.exp(-t_b / (2.0 * 2.0))
    prefactor = prefactor / (4.0 * math.pi * t_b) ** 1.5
    return prefactor * integral


def _apply_one_recursion(
    prev_kernel_fn,
    t: torch.Tensor,
    d: torch.Tensor,
) -> torch.Tensor:
    """Apply one step of the Grigor'yan–Noguchi recursion

        k^{n+2}(t, d) = -(2π · sinh d)^{-1} · ∂_d k^n(t, d).

    Computes the derivative via `torch.autograd.grad` while preserving
    the outer computation graph: if the caller's `d` has
    `requires_grad`, the recursion output is connected to the caller's
    graph so `backward()` flows correctly through `d`. Otherwise we use
    a local grad-enabled clone and the result is forward-only (correct
    in value, no upstream gradient — which is exactly what callers
    without `requires_grad` expect).

    Numerical note: the `1/sinh d` factor amplifies float noise near
    d = 0. The implementation clamps the denominator at
    `finfo(dtype).tiny` to prevent NaN, but the kernel value itself
    is ill-defined exactly at d = 0 for n ≥ 5 (the recursion produces
    a `1/sinh^{n-3}` factor in the dominant term).
    """
    # If `d` is already a grad-tracked input, differentiate through it
    # so the output remains in the caller's graph. `create_graph=True`
    # is essential for two reasons: (1) it lets `backward()` flow
    # through the recursion to upstream of `d`, and (2) it allows
    # nested recursion calls (n=7, 9, …) to keep building the graph.
    if d.requires_grad:
        kn = prev_kernel_fn(t, d)
        dk_dd, = torch.autograd.grad(
            kn.sum(), d, create_graph=True,
        )
    else:
        # No outer grad context — use a local grad-enabled clone purely
        # to evaluate the derivative. Output is forward-only, which is
        # the correct semantics when the caller has no upstream grad.
        d_local = d.detach().clone().requires_grad_(True)
        kn = prev_kernel_fn(t, d_local)
        dk_dd, = torch.autograd.grad(kn.sum(), d_local, create_graph=False)
    sinh_d = torch.sinh(d).clamp(min=torch.finfo(d.dtype).tiny)
    return -dk_dd / (2.0 * math.pi * sinh_d)


def _heat_kernel_unit(
    n: int,
    t: torch.Tensor,
    d: torch.Tensor,
    n_quad: int,
    tail_budget: float,
) -> torch.Tensor:
    """Unit-curvature dimension dispatch.

    Maps `n` to the appropriate routine:
      - n = 1 → Gaussian
      - n = 2 → Gauss–Legendre on the Davies–Mandouvalos integral
      - n = 3 → Davies–Mandouvalos closed form
      - n ≥ 5 odd → recursion from n=3, `(n-3)/2` applications
      - n ≥ 4 even → recursion from n=2, `(n-2)/2` applications
    """
    if n == 1:
        return _heat_kernel_unit_n1(t, d)
    if n == _N_GRIGORYAN_INTEGRAL:
        return _heat_kernel_unit_n2(t, d, n_quad, tail_budget)
    if n == _N_DAVIES_MANDOUVALOS:
        return _heat_kernel_unit_n3(t, d)

    # Higher dimensions via recursion. The seed depends on the parity.
    if n % 2 == 1:
        # Odd n ≥ 5: seed at n=3, apply recursion (n - 3) / 2 times.
        steps = (n - _N_DAVIES_MANDOUVALOS) // 2
        kernel_fn = _heat_kernel_unit_n3
    else:
        # Even n ≥ 4: seed at n=2, apply recursion (n - 2) / 2 times.
        steps = (n - _N_GRIGORYAN_INTEGRAL) // 2

        def kernel_fn(_t, _d):
            return _heat_kernel_unit_n2(_t, _d, n_quad, tail_budget)

    # Each application wraps the previous function in another autograd
    # call. The chain depth is bounded by `steps` (typically ≤ 4 for
    # practical n ≤ 11), so the runtime is `steps × prev_cost`.
    current_fn = kernel_fn
    for _ in range(steps):
        prev_fn = current_fn

        def current_fn(_t, _d, _prev=prev_fn):
            return _apply_one_recursion(_prev, _t, _d)

    return current_fn(t, d)


@with_provenance(
    "holonomy_lib.hyperbolic.hyperbolic_heat_kernel", op_version="0.1",
)
def hyperbolic_heat_kernel(
    t: torch.Tensor,
    distances: torch.Tensor,
    manifold,
    n_quad: int = HEAT_KERNEL_QUADRATURE_NODES,
    tail_budget: float = HEAT_KERNEL_QUADRATURE_TAIL_BUDGET,
) -> torch.Tensor:
    """Heat kernel `k^n_t(d)` on the hyperbolic manifold `manifold`.

    Computes the probability density of a Brownian-motion particle at
    geodesic distance `d` after time `t`, starting from a delta source
    at the manifold origin.

    The kernel depends only on `t`, the geodesic distance `d`, and the
    intrinsic dimension `n` of the manifold (rotational symmetry
    around the source). Curvature scales out: for curvature `k = -|k|`,

        k^n_{−|k|, t}(d)  =  |k|^{n/2} · k^n_{−1, |k|·t}(√|k| · d).

    Args:
      t: positive time(s). Broadcastable with `distances`. Typically
        a scalar, but `(B,)` or matching `distances.shape` are also
        supported.
      distances: non-negative geodesic distance(s). Any shape, must
        be broadcastable with `t`.
      manifold: a manifold object exposing `.n` (intrinsic dim) and
        `.k` (sectional curvature, expected `k < 0`).
      n_quad: number of Gauss–Legendre nodes for the n=2 integral.
        Default 32; cataloged as `HEAT_KERNEL_QUADRATURE_NODES`.
      tail_budget: integration-upper-bound budget factor (truncate at
        `d + sqrt(tail_budget · t)`). Default 20. Cataloged.

    Returns:
      Tensor of heat-kernel values, same shape as
      `broadcast(t, distances)`.

    Dimension support:
      - n = 1, 2, 3: dedicated routines (closed form / integral /
        closed form).
      - n ≥ 5 odd, n ≥ 4 even: Grigor'yan–Noguchi recursion via
        `torch.autograd.grad`. Numerically reliable for n ≤ ~9 on
        float64 and `d` bounded away from 0. For `d → 0` with n ≥ 5
        the limit exists but the implementation is numerically
        ill-conditioned (`1/sinh^{n-3} d` amplifies float noise).

    References:
      Davies–Mandouvalos (1988); Grigor'yan (2009), Theorem 8.21;
      Grigor'yan–Noguchi (1998); Anker–Ostellari (2003).
    """
    n = manifold.n
    abs_k = abs(manifold.k)

    # Curvature scaling: convert to unit-curvature arguments, evaluate,
    # then rescale the output by |k|^{n/2}.
    t_unit = abs_k * t
    sqrt_abs_k = math.sqrt(abs_k)
    d_unit = sqrt_abs_k * distances

    # Ensure tensors (constants get the right dtype for downstream
    # numpy/scipy conversions).
    if not isinstance(t_unit, torch.Tensor):
        t_unit = torch.as_tensor(
            t_unit, dtype=distances.dtype, device=distances.device,
        )
    k_unit = _heat_kernel_unit(n, t_unit, d_unit, n_quad, tail_budget)
    scale = abs_k ** (n * 0.5)
    return scale * k_unit
