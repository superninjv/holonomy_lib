# What synoros-substrate and holonomy_lib gave each other

`holonomy_lib` and its sibling research project `synoros-substrate`
co-evolve: the library is the stabilized, trusted math core; the
substrate is the live research consumer. Per the project charter,
primitives developed in the substrate migrate into the library as they
stabilize, and the substrate then depends on the library instead of
carrying its own copy. This note records what actually flowed each way,
separating what the git history proves from what is a grounded reading
of why a primitive appeared.

## Substrate → library (what the substrate contributed)

**Verified from history:**

- **Genesis.** The library was seeded from a substrate session
  (commit `ae65b58`, "Initial commit — synoros-lib seeded from
  synoros-substrate session"). holonomy_lib exists because the substrate
  work needed a clean, trustworthy place for its geometry to live.
- **`FixedRankManifold`** was ported directly from the substrate and
  rewritten batched-first (commit `7a366af`, "manifolds: port
  FixedRankManifold from substrate, batched-first").
- **Design patterns the substrate handed up**, recorded in the source:
  - `HeterogeneousKappaManifold` exists to support the substrate team's
    *smooth κ-field plus per-point residual δ* parameterization — the
    library provides the math primitive, the substrate owns the
    parameterization (`manifolds/heterogeneous_kappa.py`, lines 27, 82).
  - `LorentzianManifold`'s framing is the substrate's
    *substrate-as-spacetime* idea — time as a geometric dimension, points
    time-/light-/space-like separated (`manifolds/lorentzian.py`, line 20).

**Recent additions (v0.5.2, 2026-05-28, while the substrate research was
active) — the most likely "thing it added while working":** three
primitives entered the library in the unreleased v0.5.2 window. None
self-documents a substrate link in its docstring, but each answers a need
the substrate's current research raised, so the attribution is a grounded
reading rather than a git fact — worth the user confirming:

- **`spectral.spectral_dimension`** — the spectral dimension `d_s` of a
  Laplacian spectrum from the heat-kernel return-probability decay
  (`p(t) ~ t^{-d_s/2}`), non-integer allowed. This is the instrument for
  the substrate's sharpest current finding: that the learned substrate
  *collapses toward ~1D*. You cannot report a collapse without a
  dimension estimator; this is it.
- **`sheaf.HeterogeneousGraphSheaf`** — a cellular sheaf with per-node
  (heterogeneous) stalk dimensions and ragged restriction maps. This is
  the geometric form of the substrate's *relations-as-forces, per-node
  rank* structure (each concept carries its own stalk; relations are the
  restriction maps between them).
- **`manifolds.comparison`** — Bishop-Gromov model-space geodesic sphere
  area and ball volume `S(κ,N,r)`, `V(κ,N,r)` for real (non-integer)
  dimension and signed curvature. The comparison reference for the
  substrate's curvature- and dimension-dependent volume/flux diagnostics
  (and a companion to `spectral_dimension`'s fractional dimension).

The substrate's own integration commit (`47cca57`, "v3: holonomy_lib
integration ... + full-signal measurement layer") aligns with these
landing as the substrate's *measurement layer* — tools to quantify what
its geometry is doing.

## Library → substrate (what the library gave back)

- The substrate's v3 **integrates holonomy_lib's manifolds and
  provenance** (substrate commit `47cca57`) and **refactored its
  differential-geometry core to be PyTorch-native** through the library
  (substrate commit `c5c856b`, "refactor diff-geo substrate to
  PyTorch-native"). The substrate stopped carrying its own geometry and
  started depending on the stabilized library — the migration direction
  the charter intends.

## Reading

This is genuine co-evolution, not a one-way port. The substrate seeds and
stress-tests primitives against a live research need; the ones that
survive harden in the library with citations, an audit, and tests, and
the substrate then consumes them. For the paper, this supports the
"substrate it serves" framing and could become a short
provenance/related-systems paragraph — as a description of the
development loop, not as an empirical result of the substrate research
(which remains inconclusive; see the assessment in the paper handoff).
