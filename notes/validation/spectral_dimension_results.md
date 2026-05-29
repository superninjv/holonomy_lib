# spectral_dimension — validation against known d_s

`spectral.spectral_dimension(eigenvalues, t)` estimates the spectral
dimension as `d_s = -2 · slope of log p(t) vs log t`, where
`p(t) = mean_i exp(-t·λ_i)` is the heat-kernel return probability. This
is the reference-comparison check: feed Laplacian spectra of structures
whose `d_s` is known in closed form and confirm recovery.

Run: `uv run python notes/validation/spectral_dimension_validation.py`

## Window rule (one rule, all structures)

`p(t) ~ t^{-d_s/2}` holds only in the power-law window
`1/λ_max ≪ t ≪ 1/λ_gap`. We fit the **asymptotic tail**: the band of
`t` still above the finite-size floor `p → (#zero modes)/n` (mask
`p > 4·floor`), with `t_lo = max(2, t_hi/100)` to skip the small-`t`
transient. This single rule is applied to every structure below — no
per-case tuning. (The lattice return-probability slope approaches
`−d_s/2` *from above*, so the tail is the cleanest place to read it.)

## Results

| structure | known `d_s` | recovered | t-window | err |
|---|---:|---:|---|---:|
| 1-D ring (cycle, n=4096) | 1.0000 | 1.0000 | [8.3e2, 8.3e4] | <0.1% |
| 2-D torus (64×64) | 2.0000 | 2.0309 | [2, 81] | 1.5% |
| 3-D torus (20³) | 3.0000 | 3.0943 | [2, 13] | 3.1% |
| Sierpinski gasket (level 7, n=3282) | 1.3652 | 1.3201 | [9.7, 9.7e2] | 3.3% |

Sierpinski known value `= 2·ln3/ln5 = 1.36521…` (Rammal–Toulouse 1983).

## Reading

- The integer-dimensional lattices recover `d_s ∈ {1, 2, 3}` to within
  ~3% — the residual is the finite-`t` approach of the torus slope to its
  asymptote, not an error in the estimator (the 1-D ring, with a far
  wider clean window, recovers `1.0000`).
- The **Sierpinski gasket recovers the non-integer `2·ln3/ln5`** to ~3%.
  Residual sources: finite level-7 truncation and the log-periodic
  oscillation intrinsic to a self-similar spectrum (the slope wobbles
  around the mean power law). This is the headline check — the primitive
  reads a genuinely fractional dimension, not just integers.

## Why this primitive matters downstream (the substrate bridge)

Spectral dimension is the instrument for a question the substrate
research raised directly: **has a learned representation collapsed to a
low effective dimension?** A point cloud or operator that nominally lives
in `R^D` but has collapsed onto an effective 1-D manifold registers
`d_s ≈ 1` regardless of `D` — the return probability decays like a line,
not like a `D`-dimensional space. `spectral_dimension` reads that
collapse off the Laplacian spectrum without any embedding or
visualization. The substrate's observed drift toward an effectively
1-D geometry is exactly the regime this measures (see
`notes/substrate_provenance.md`).
