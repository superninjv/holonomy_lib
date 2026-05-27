# Cross-manifold end-to-end validation

Substrate-style training loop on each manifold: tangent-at-origin embedding, all-pairs distance, NLL loss with cyclic target, SGD on `v`. 20 steps. Records: gradient finiteness throughout training, on-manifold-ness of final embedding, loss decrease.

## Riemannian manifolds (substrate training loop)

| manifold | grad finite | on manifold | loss start | loss end | decreased |
|---|:---:|:---:|---:|---:|:---:|
| Lorentz (k=-1) | ✓ | ✓ | 17.9179 | 15.0398 | ✓ |
| KappaStereographic (kappa=-1) | ✓ | ✓ | 20.8583 | 14.8663 | ✓ |
| KappaStereographic (kappa=+0.5, spherical) | ✓ | ✓ | 19.9680 | 14.2720 | ✓ |
| KappaStereographic (kappa learnable = -1.0) | ✓ | ✓ | 20.8583 | 14.8663 | ✓ |

## Lorentzian manifold (causal / curvature primitives)

`LorentzianManifold` is flat pseudo-Riemannian; the tangent-at-origin substrate pattern doesn't apply directly. We instead verify the causal classification + curvature-tensor primitives behave as expected.

- **LorentzianManifold(n=4) — flat**
  - causal_classification: ✓
  - curvature_tensors_zero: ✓
  - metric_is_minkowski: ✓

## Summary

End-to-end smoke test of the four-stage hyperbolic extension (Stages 1–4 + the autograd / scale-invariance / heat-kernel fixes + learnable κ). All paths are functional; gradient stays finite throughout; embeddings stay on the respective manifolds.
