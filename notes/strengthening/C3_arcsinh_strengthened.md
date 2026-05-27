# C3 strengthened: arcsinh reparameterization for hyperbolic distance

Consolidated strengthening evidence for the claim:

> Replacing the textbook hyperbolic distance form `d_k(x, y) =
> (1/√|k|) · arccosh(k·⟨x,y⟩_M)` with the equivalent
> `(2/√|k|) · arcsinh(√|k|·‖y − x‖_M / 2)` eliminates two distinct
> failure modes (arccosh's boundary derivative singularity AND
> catastrophic cancellation in `z = k·⟨x,y⟩_M` for x ≈ y) without
> introducing an `eps` hyperparameter.

## (1) Algebraic equivalence — sympy verified

`notes/verification/arcsinh_reparam_sympy.py` proves the identity on
the unit hyperboloid H¹_{-1}: for any geodesic distance α,

```
arccosh(cosh α) = 2·arcsinh(sqrt(2(cosh α - 1))/2)
```

reduces to `α = α` (sympy.simplify residual = 0). Generalizes to
H^n_k via the same half-angle identity `cosh A - 1 = 2·sinh²(A/2)`.

## (2) The two distinct failure modes

### (A) arccosh's boundary derivative singularity

`d/dz arccosh(z) = 1/√(z² − 1)`. At `z = 1` (x = y, the on-manifold
self-pair case), this is `1/0 = ∞`. Even though `arccosh(1) = 0` in
forward, backward propagates `∞ · ∂z/∂x = NaN`.

arcsinh has derivative `1/√(1 + arg²)` — entire, value 1 at `arg = 0`.
No boundary singularity. Backward is finite at the analogous
on-manifold self-pair.

### (B) Catastrophic cancellation in z = k·⟨x,y⟩_M for x ≈ y

For x, y on the hyperboloid with geodesic distance α:
```
z = k·⟨x,y⟩_M = cosh(α) (for k = -1) = 1 + α²/2 + O(α⁴)
```

Computing `z` involves multiplying coords and summing — float
arithmetic returns `1.0000…` for small α, with the meaningful
information in the last few bits. Then `arccosh(z) = arccosh(1 + small)`
loses precision (`arccosh(1 + δ) ≈ √(2δ)`, so a δ at the noise floor
gives an arccosh at the square-root of the noise floor).

The arcsinh form computes `‖y - x‖_M²` from coordinate differences:
```
‖y - x‖_M² = -(y_0 - x_0)² + Σ(y_i - x_i)² = 2·(cosh α - 1) ≈ α².
```
No near-1 subtraction. Information preserved.

## (3) Float64 demonstration — worked example

`notes/strengthening/C3_arcsinh_worked_example_results.md` runs all
three forms (textbook, eps-clamp, arcsinh) in PyTorch float64 at
geodesic distances α ∈ {1e-4, 1e-6, 1e-8, 1e-10, 0}:

| α | textbook | eps-clamp (eps=1e-5) | arcsinh (ours) |
|---:|---:|---:|---:|
| 1e-4 | 1.00e-4 ✓ | 4.47e-3 (constant!) | 1.00e-4 ✓ |
| 1e-6 | 1.00e-6 ✓ | 4.47e-3 | 1.00e-6 ✓ |
| 1e-8 | **0** ← all bits lost | 4.47e-3 | 1.00e-8 ✓ |
| 1e-10 | **0** ← all bits lost | 4.47e-3 | 1.00e-10 ✓ |
| 0 (x=y) | 0 fwd, NaN bwd | 4.47e-3 | 0 fwd, 0 bwd ✓ |

**Read the eps-clamp column carefully**: for any α below the clamp
threshold, it returns `arccosh(1 + eps) ≈ √(2·eps) ≈ 4.47e-3`,
**independent of the true distance**. That's a constant forward bias
of ~4.5e-3 — substantial. The arcsinh form tracks the true value all
the way down.

For backward at x = y:
| form | forward | backward x.grad |
|---|---:|---:|
| textbook arccosh | 0.0 | 1/2 NaN |
| eps-clamp (eps=1e-5) | 4.5e-3 | finite (biased) |
| arcsinh (ours) | 0 | finite (exact zero) |

## (4) Implementation in `LorentzManifold`

`src/holonomy_lib/manifolds/lorentz.py:distance`:

```python
diff = y - x
diff_sq = (diff[..., 1:] ** 2).sum(dim=-1) - diff[..., 0] ** 2
arg = self._sqrt_abs_k * _safe_sqrt(diff_sq) * 0.5
return (2.0 * self._inv_sqrt_abs_k) * torch.asinh(arg)
```

The `_safe_sqrt` helper (also our research; see `_safe_sqrt` etc.
section in C3's "related work" below) handles the boundary case
`diff_sq = 0` (exact x = y) via `torch.where(x > 0, sqrt(x), 0)` —
the masked-out branch never sees the singular `sqrt(0)` derivative.

The same idiom drives `log`:
```python
diff = y - x
diff_sq = (diff[..., 1:] ** 2).sum(dim=-1) - diff[..., 0] ** 2
arg = self._sqrt_abs_k * _safe_sqrt(diff_sq) * 0.5
inv_sinhc = _safe_arcsinhc(arg)            # arcsinh(arg)/arg, smooth at 0
inv_cosh_factor = torch.rsqrt(arg * arg + 1.0)
scale = inv_sinhc * inv_cosh_factor
return scale.unsqueeze(-1) * u             # u = projection of y onto T_x
```

The factor `arcsinh(arg)/(arg · √(arg²+1))` is mathematically
equivalent to `α/sinh(α)` (the textbook `log = β · u/‖u‖_x`
multiplier), but the arcsinh-form expression has no boundary
singularity (arcsinh smooth, `arg²+1 ≥ 1` always).

## (5) Related work — `_safe_*` helper idiom

The `_safe_sqrt` / `_safe_sinhc` / `_safe_arcsinhc` pattern
`torch.where(cond, formula(safe_x), default)` is folklore in
PyTorch numerical code:

- **geoopt** uses `clamp(min=eps)` for `arccosh` inputs — the bias
  approach we benchmark against above.
- **PyTorch3D** has similar `safe_*` patterns scattered through
  `_eps`-style helpers.
- **PyTorch documentation** mentions the `where`-on-safe-input
  pattern in the autograd FAQ.

What we contribute: (a) the **arcsinh reparameterization** as the
*specific* application to hyperbolic distance — eliminating both
the eps and the cancellation simultaneously, vs eps-clamp which
only fixes the boundary divergence; (b) the **consistent
application** through `distance` / `log` / `norm` etc. in
`LorentzManifold` so the whole API is autograd-safe at every
boundary; (c) the **demonstration** above that this matters in
practice for substrate-style NLL training where `d(x, x)` self-pairs
appear in the partition function.

## (6) Performance impact

Both `torch.acosh` and `torch.asinh` are single CUDA/CPU kernels of
comparable cost. Our `distance` adds one `_safe_sqrt` (one
`torch.where`, one `torch.sqrt`) — measurable but small overhead vs
the eps-clamp path. From the `autograd_safe_vs_geoopt.py` benchmark:

| form | forward+bwd, batch=10k, n=8 | overhead |
|---|---:|---|
| eps-clamp | ~1.0 ms / iter | baseline |
| arcsinh + `_safe_*` | ~1.3 ms / iter | +30% |

This is **microbench overhead in isolation**. In a realistic
training loop where `distance` is one op among many, the relative
cost shrinks. Trade-off: ~30% per-call cost vs eliminating an
eps hyperparameter + an O(eps) forward bias + the boundary NaN risk.

## (7) Paper section

This document fills **§3.2 ("Equivalent math, very different float
behavior")** of the paper. The arccosh-vs-arcsinh case study is a
concrete demonstration of "implementation correctness ≠
mathematical correctness" — both forms are exact on paper, but only
one is reliable in float64 + autograd.

## Status update

| Claim | Was | Now |
|---|---|---|
| C3 (arcsinh reparameterization) | 🔴 | 🟢 (sympy-verified equivalence; numerical demonstration; related-work positioned; performance characterized) |
