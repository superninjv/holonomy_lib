# synoros-lib

A research-oriented PyTorch math library for ML experimentation.

## What this is

A consolidated, GPU-native, audit-compliant PyTorch library covering the mathematical fields needed for modern ML and cognitive-substrate work:

- Topology (basic, simplicial, persistent homology)
- Differential geometry (Riemannian manifolds, tangent spaces, parallel transport)
- Algebra (linear, tensor, group representations, Lie algebras)
- Spectral theory (Laplacians, eigendecomposition, embeddings, heat kernels)
- Information geometry (Fisher metric, divergences, natural gradient)
- Tensor calculus (Einstein notation, contractions, decompositions)
- Optimization (Riemannian, constrained, manifold-aware)
- Probability (conjugate priors, manifold-valued distributions)

## Why this exists

Modern ML research keeps reinventing the same mathematical primitives across projects, often picking arbitrary constants because the right derivations aren't readily available. Existing libraries cover slices (geoopt = Riemannian optimization; geomstats = differential geometry; pymanopt = manifolds CPU-only; gudhi = topology) but no single library:

1. Covers the breadth needed for cognitive-substrate / Riemannian-ML / information-geometric work
2. Is GPU-native (PyTorch) AND batched-first
3. Enforces audit discipline — no arbitrary numerical constants; every constant is derived, a universal invariant, or experimentally-set with documented procedure
4. Cites and grounds every operation in the published math literature

This library aims to do all four.

## Status

Initial primitives shipped across `manifolds`, `algebra`, `tensor_calculus`, `spectral`, `discrete_geometry`, and `provenance` (per-operation content-addressable hex for mechanistic interpretability). See `AGENT.md` for the full inventory, signatures, and citations — that's the reference you (or an LLM agent using this library) want when you need to know "what's available, what does it take, what should I cite."

`HANDOFF.md` is the original build plan from the seeding session; some scope language there is outdated (says "what synoros-substrate needs" — actually general-purpose).

## Relationship to synoros-substrate

This library is predicated on the experience of building [synoros-substrate](https://github.com/superninjv/synoros-substrate) — a cognitive-substrate research project that repeatedly hit the same pattern: needing mathematical primitives that don't exist in a single library, having to reinvent them, then catching the result drifting because someone (often Claude) added an arbitrary numerical constant.

synoros-substrate's `synoros_library/manifolds/` is the seed for this library's geometry section. Other primitives developed there will migrate here. Once mature, synoros-substrate will depend on this library rather than carrying its own math implementations.

## Audit discipline

Every numerical constant in the library must be one of:

- **(a) Derived from inputs** — function of dimensions, structure, corpus properties
- **(b) Universal invariant** — π, e, log N, 1/N, √N, mathematical identities
- **(c) Experimentally tuned** — value documented with procedure and scale-of-validity

A numeric literal that isn't (a), (b), or (c) is a bug. The audit tool catches them at CI time.

## License

TBD (probably MIT, matching the math-library norm).
