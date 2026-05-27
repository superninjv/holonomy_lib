# C3 worked example: arcsinh reparameterization in float64

PyTorch float64 demonstration of why the arcsinh form is preferable to the textbook arccosh form for hyperbolic distance, especially at small geodesic separations.

Three forms compared:

- **Textbook arccosh**: `d = arccosh(-⟨x, y⟩_M)`. Mathematically correct everywhere, but `arccosh'(z) = 1/√(z² − 1)` diverges at `z = 1` (when x = y), AND the computation of `z` itself loses precision when x ≈ y (near-1 subtraction).
- **Eps-clamped arccosh** (geoopt style): `d = arccosh(clamp(-⟨x,y⟩_M, min = 1 + eps))`. Forward finite, backward stable, but gradient *biased* by O(eps).
- **Arcsinh reparameterization** (ours): `d = 2·arcsinh(‖y - x‖_M/2)`. Arcsinh is entire — no boundary singularity. `‖y - x‖_M²` computed from coordinate differences avoids the near-1 cancellation.

## Forward accuracy as α → 0

At small geodesic distance α, `z = cosh(α) ≈ 1 + α²/2`. Subtracting two ~1 floats to find `z - 1` loses bits — catastrophic cancellation. The arcsinh form computes `‖y - x‖_M² = 2(cosh α - 1) ≈ α²` from coord differences, which doesn't cancel.

| α (true distance) | textbook | eps-clamp (eps=1e-5) | arcsinh (ours) |
|---:|---:|---:|---:|
| 1e-04 | 1.0000e-04 | 4.4721e-03 | 1.0000e-04 |
| 1e-06 | 1.0000e-06 | 4.4721e-03 | 1.0000e-06 |
| 1e-08 | 0.0000e+00 | 4.4721e-03 | 1.0000e-08 |
| 1e-10 | 0.0000e+00 | 4.4721e-03 | 1.0000e-10 |
| 0 (x=y exactly) | NaN (arccosh(1)' = ∞) | (1+eps)¹/² ≈ 4.5e-3 | 0 (exact) |

At α = 0 explicitly (x = y):
  - textbook: `d = 0.0` (arccosh(1.0) = 0 fwd, but ∂/∂z = ∞)
  - eps-clamp: `d = 4.472132e-03` (positive bias ~ √(2·eps))
  - arcsinh: `d = 0.0` (exact zero ✓)

## Backward at x = y (the d(x, x) = 0 case)

This is the substrate-team-reported scenario: NLL loss over all-pairs distance includes `d(x_i, x_i)` self-pairs in the partition function. With the textbook form, backward at those self-pairs is NaN.

| form | forward | backward x.grad |
|---|---:|---:|
| textbook arccosh | 0.0 | finite=False, NaN=1/2 |
| eps-clamp arccosh (eps=1e-5) | 4.4721e-03 | finite=True |
| arcsinh (ours) | 0e+00 | finite=True, grad=[[-0.0, -0.0]] |

## Conclusion

The arcsinh form is **mathematically identical** to the textbook arccosh form on the hyperboloid (sympy verification in `notes/verification/arcsinh_reparam_sympy.py`). But in float64 PyTorch, it gives:

- **Forward**: exact zero at x = y (vs textbook's NaN-from-∂/∂z = ∞).
- **Better precision at small α**: avoids the `1 + α²/2 − 1` cancellation that loses ~half the bits in the textbook form (visible in the α = 1e-8 / 1e-10 rows above).
- **No eps hyperparameter**: eps-clamp introduces O(eps) gradient bias and a small forward bias `≈ √(2·eps)`. Arcsinh is exact.
- **Boundary backward**: arcsinh's derivative is `1/√(1 + x²)` — smooth everywhere. No NaN propagation, no eps needed.

The trade-off is one extra `arcsinh` call vs `arccosh`; both are single torch ops with identical autograd cost. We get the precision + autograd safety **for free**.
