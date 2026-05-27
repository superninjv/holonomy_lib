# Heat-kernel validation — findings

## Headline

**`hyperbolic_heat_kernel` had a mathematical bug for n ≥ 5.** The
recursion `k^{n+2} = -(2π sinh r)^{-1} · ∂_r k^n` that the code
implemented is wrong — it's missing a spectral-shift factor
`exp(-n·t)`. Caught by the heat-equation residual validation;
fix landed in the same pass.

## The bug

The original implementation assumed Grigor'yan–Noguchi recursion
in the form

  k^{n+2}(t, r) = -1 / (2π sinh r) · ∂_r k^n(t, r).

This is **not** the correct relation. The actual recursion is

  k^{n+2}(t, r) = -exp(-n·t) / (2π sinh r) · ∂_r k^n(t, r),

where the `exp(-n·t)` factor accounts for the spectral-bottom shift
between dimensions:

- `H^n` heat kernel decays at the spectral bottom `((n-1)/2)²·t`.
- Going from `n_prev` to `n_prev+2` shifts the exponent by
  `((n_prev+1)/2)² − ((n_prev−1)/2)² = n_prev`, so the recursion
  picks up `exp(-n_prev·t)`.

Without the shift, the function `-1/(2π sinh r) · ∂_r k^n` carries
the wrong spectral bottom, so it doesn't solve the H^{n+2} heat
equation.

## Derivation

Let `f_m(t, r) := (1/sinh r · ∂_r)^m · exp(-r²/4t)` (operator chain
on the *un-shifted* Gaussian). Then the Davies-Mandouvalos /
Grigor'yan formula for odd `n = 2m+1` is

  k^{2m+1}(t, r) = (-1/2π)^m · (4πt)^{-1/2} · exp(-m²·t) · f_m(t, r),

since (1/sinh r · ∂_r) commutes with `exp(-m²·t)` (no r-dependence).
Going from `m` to `m+1`:

  k^{2(m+1)+1} = (-1/2π)^{m+1} · (4πt)^{-1/2} · exp(-(m+1)²·t) · f_{m+1}
              = (-1/2π) · k^{2m+1}|_{f_m → f_{m+1}/f_m} · exp(-(2m+1)·t)
              = -1/(2π·sinh r) · exp(-(2m+1)·t) · ∂_r k^{2m+1}.

So the recursion FROM `n` (`= 2m+1`) TO `n+2` picks up `exp(-n·t)`.
Verified empirically: with the fix, n=5 closed form matches the
Davies-Mandouvalos textbook expansion (sanity-checked against the
heat equation).

## Validation results

`notes/validation/heat_kernel_validation.py` checks two independent
properties — heat-equation residual and probability-mass
normalization — at multiple `(n, t, r)`.

| Test | Status |
|---|---|
| Heat-equation residual `∂_t k − Δ_radial k ≈ 0` | ✓ at FD noise floor for n ∈ {1, 2, 3, 5, 7, 9} |
| Probability mass `∫ k_t · dV = 1` | ✓ to machine precision for n ∈ {1, 2, 3, 5, 7, 9} (except n=9, t=5.0 — numerical-blowup regime) |
| n=3 vs Davies-Mandouvalos closed form | ✓ atol 1e-12 |
| n=5 vs n=3 + spectral-shift recursion | ✓ atol 1e-6 |
| n=5 ↔ n=7 spectral-shift identity | ✓ atol 1e-6 |
| Even n ≥ 4 | Now raises `NotImplementedError` (was returning incorrect values) |

## Status by dimension

| n | Status | Notes |
|---|---|---|
| 1 | ✓ closed form | Gaussian on R |
| 2 | ✓ Davies-Mandouvalos integral | Gauss-Legendre quadrature, 32 nodes default |
| 3 | ✓ Davies-Mandouvalos closed form | `(4πt)^{-3/2} · exp(-t - r²/4t) · r/sinh r` |
| 5, 7, 9 (odd) | ✓ corrected recursion | Validated via heat equation + recursion identity |
| 4, 6, 8 (even) | ❌ `NotImplementedError` | Previous implementation was wrong; correct even-n recursion needs separate derivation |

## What this means for the paper

The implementation strategy — chaining `torch.autograd.grad` with
`create_graph=True` to apply `(1/sinh r · ∂_r)^m` numerically — is
**novel** as far as I can tell. No PyTorch hyperbolic ML library
(geoopt, geomstats, pymanopt, hyperlib) implements the heat kernel at all,
much less in a fully-differentiable, general-`n`, autograd-flowing form.

The contribution would be:
- A general-`n` differentiable hyperbolic heat-kernel implementation.
- Verified mathematically correct against the radial heat equation
  + probability normalization.
- End-to-end autograd through the recursion (this paper's key
  experiment: backprop through a heat-kernel-based loss).

This is a focused implementation contribution suitable for a
**workshop paper** or **technical report** — not a full ICML/NeurIPS
paper on its own, but a useful primitive for downstream methods.

## Open items

1. **Even n ≥ 4 recursion**: needs a separate derivation. The
   Davies-Mandouvalos integral form for n=2 doesn't compose with
   the simple (1/sinh r · ∂_r) operator the way the odd-n closed
   form does. Pending research.

2. **Numerical blowup at (n=9, t=5.0)**: large-t × high-n regime
   where the recursion's compounded sinh-divisions overflow. The
   probability mass goes NaN. Need to characterize the safe
   `(n, t, r)` envelope.

3. **Compounded autograd noise**: each recursion step adds a layer
   of `torch.autograd.grad`. The numerical residual at n=9 is
   ~1e-3 in the worst case (vs ~1e-6 for n=5). For n > 9 we'd
   expect further degradation. Need to characterize where to
   switch to a different scheme (e.g. direct closed-form
   polynomial expansion of the operator chain).
