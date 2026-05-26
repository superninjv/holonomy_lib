# holonomy-lib — agent operating constraints

This project's purpose, scope, and architectural commitments are in `README.md` and `HANDOFF.md`. Read those first.

## Hard constraints (binding for any agent working in this repo)

1. **Discuss-look-test research method.** Before adding a primitive, find existing implementations in geoopt / geomstats / pymanopt / gudhi / etc. Decide: port directly, wrap, or reimplement. Don't reinvent without checking.

2. **Every numerical constant has a derivation or a catalog entry in `notes/magic_numbers.md`.** Per the project audit. Three categories:
   - Derived from inputs (function of dims/structure/problem)
   - Universal invariant (π, e, log N, 1/N, √N, math identities)
   - Experimentally tuned with documented procedure + scale-of-validity

3. **Citations are non-optional.** Every public function has a `References:` section in its docstring pointing to the paper/textbook for its math. No "trust me" implementations.

4. **GPU-first, batched-first.** Default: operations take leading batch dim, work on `torch.Tensor` on `cuda`. CPU and single-element are special cases. Verify shapes work for B=0, B=1, B>1.

5. **Library code only.** No research experiments, no model classes, no training loops, no architectural commitments. This IS the library other projects depend on.

6. **No fluent prose without source-line verification.** If you find yourself writing confident prose about how a function should work, stop and read the file at the relevant lines.

7. **Tests before commit.** Unit tests for correctness, property tests where applicable (e.g., manifold operations preserve their invariants), comparison tests against established libraries.

## NEVER DO

- Don't add features speculatively. Build what a consumer (synoros-substrate or similar) actually needs.
- Don't pick architectural choices unilaterally. When unsure, present options + tradeoffs and stop.
- Don't claim a primitive "works" without tests AND a comparison to an established library (where one exists).
- Don't break the audit. CI runs it. Failing audit = failing build.
- Don't add a numerical literal because "it's just a small number" — derive it, mark it universal, or catalog it as posited.
- Don't add backwards-compat shims or feature flags. This is a research library; we change it, we don't preserve.

## Relationship to synoros-substrate

Sibling repo at `~/projects/synoros-substrate/`. That project's experience shaped this library's design. Primitives developed there should migrate here as they stabilize; once migrated, synoros-substrate depends on holonomy-lib instead of carrying its own copy. The two repos co-evolve.

For now: synoros-substrate has its own `src/holonomy_library/` which is the parent of this library. Migration happens piece-by-piece — DON'T break synoros-substrate's imports while migrating.

## Posture

Peer, not prescription. Critical feedback welcome. Mathematical rigor matters; engineering shortcuts that compromise it are not acceptable. The point of this library is that researchers can TRUST the implementations — every shortcut here costs that trust everywhere downstream.

## What "done" looks like for a primitive

A primitive (e.g., `manifolds.fixed_rank.FixedRankManifold`) is "done" when:

- [ ] Implemented in PyTorch on configurable device, batched-first
- [ ] Type-annotated; shapes documented in docstrings
- [ ] Docstring includes `References:` section with paper/textbook citation
- [ ] Unit tests verify correctness on small examples
- [ ] Property tests verify mathematical invariants (e.g., projection is idempotent, retraction stays on manifold)
- [ ] Comparison test against an established library (e.g., pymanopt for manifolds, gudhi for topology) where one exists — checking we get same results
- [ ] Passes audit (no undocumented numerical literals)
- [ ] Cataloged in `notes/magic_numbers.md` if it has any tunable/derived constants

If any of these is missing, the primitive isn't done.

## Operational

- Python 3.12+
- PyTorch 2.x with CUDA support (where possible)
- Dev dependencies in `pyproject.toml` (pytest, ruff, mypy)
- `uv` is the package manager
- CI: GitHub Actions (TBD), at minimum runs tests + audit
