# Changelog

All notable changes to `holonomy_lib` are documented here. Format
follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/);
version numbers follow [Semantic Versioning](https://semver.org).

## [0.4.0] - 2026-05-27

Provenance agent-API redesign. The v0.3.0 MCP server was structurally
limited: agents could navigate the DAG but couldn't inspect tensor
content beyond global mean/std/min/max, slice tensors, run linear
algebra on cached values, or express anything other than zero-fill
substitutions. A driven validation pass against a non-trivial
"find the anomalous batch" question hit 7 falls-down before
dropping back to Python. v0.4 reshapes the surface around what
agents actually need.

The core move: a new `holonomy_lib.provenance.agent` module holds
the canonical tool inventory. Each tool is a Python function
decorated with `@agent_tool`. The module emits native LLM tool-use
schemas (`to_anthropic_schema()`, `to_openai_schema()`); the
existing `mcp.py` becomes a thin transport adapter that iterates
the same registered tools. Three call sites (native tool-use, MCP,
direct Python) all hit the same underlying functions.

### Added: agent module

- `@agent_tool(description=, name=)` decorator. Inspects signatures
  + docstrings + resolved type hints (via `typing.get_type_hints`,
  so `from __future__ import annotations` modules work transparently);
  registers a `ToolSpec` in module-level `_AGENT_TOOLS`.
- `to_anthropic_schema()` / `to_openai_schema()`. Provider-specific
  schema dumpers. Both share the same JSON-schema mapping (str /
  int / float / bool / list / dict / Optional). The `registry:
  ProvenanceRegistry` parameter is automatically stripped from the
  LLM-facing schema; transport adapters pre-bind it.
- `list_tools()` / `get_tool(name)`. Inventory helpers.

### Added: inspection tools (the v0.3.0 falls-down list, addressed)

- `tensor_slice(hex_id, expr)`: numpy-syntax slicing of a cached
  tensor. Returns raw values inline if the slice has at most
  `TENSOR_SLICE_INLINE_LIMIT = 256` elements; otherwise returns
  shape + stats summary. Index parser accepts only digits, '-',
  ':', ',' (injection-safe).
- `tensor_per_batch_summary(hex_id)`: per-batch mean/std/min/max.
  Fixes the v0.3.0 "global stats hide anomalies" wall: the
  revalidation drive confirms it picks the anomalous batch in a
  fixture where v0.3.0's get_tensor_summary lumped everything
  together.
- `tensor_eigenvalues(hex_id, k)`: top-k eigvalsh per batch. Default
  k = `TENSOR_SPECTRAL_DEFAULT_K = 10`.
- `tensor_singular_values(hex_id, k)`: top-k SVD per batch.
- `tensor_norm(hex_id, order)`: Frobenius or spectral norm.
  Per-batch for batched inputs.
- `tensor_compare(hex_a, hex_b, metric)`: pairwise comparison.
  Metrics: max_abs / frobenius / cosine.
- `op_docstring(op_id)`: returns the registered op's signature +
  docstring + op_version. Fixes the v0.3.0 "discovery is weak" wall.

### Added: replay recipe DSL

- `replay_with(target_hex, recipe)` replaces the v0.3.0 MCP `replay`
  tool's zero-fill-only substitution. The recipe is a dict with a
  `kind` field; supported kinds:
  - `zeros_like`: same shape/dtype, all zeros.
  - `from_hex` + `hex`: substitute with another cached tensor.
  - `perturb` + `noise_std` + `seed`: original + Gaussian noise.
    Seed is required; no implicit defaults.
  - `scale` + `factor`: multiply original by a scalar.
  - `swap_batch` + `i` + `j`: swap two batch elements along dim 0.
  - `literal` + `values`: explicit nested-list values (small tensors
    only; JSON arrays balloon agent prompts).
- Internal `_build_substitute()` helper is testable in isolation;
  `replay_with` is the agent-facing wrapper that builds + replays
  + summarizes new outputs.

### Changed

- `mcp.py` refactored to be transport-only. Drops ~80 lines of
  inline tool definitions; iterates over `agent.list_tools()` and
  pre-binds the registry argument. Resolves stringified type hints
  before pydantic / FastMCP sees them (the old inline approach
  worked only because pydantic resolved each function's annotations
  in the local scope where it was defined).
- The v0.3.0 MCP nav tools (`list_ops`, `where`, `node_info`,
  `ancestors`, `get_tensor_summary`) are preserved by name and
  signature for backward compat with existing MCP clients; they
  just live in `agent.py` now instead of `mcp.py`.
- `replay` (zero-fill via shape spec) is dropped from the MCP
  surface in favor of `replay_with` (`{"kind": "zeros_like"}` is
  equivalent for the v0.3 use case).
- The `hex` parameter is renamed to `hex_id` in agent tools to
  avoid shadowing Python's `hex()` builtin AND match the v0.3
  convention.

### Cataloged constants

- `TENSOR_SLICE_INLINE_LIMIT = 256` (🔬 experimentally-set,
  agent.py).
- `TENSOR_SPECTRAL_DEFAULT_K = 10` (🔬).
- `PYTHON_SLICE_ARITY = 3` (✅ derived; Python language constant).

### Tests

`tests/provenance/test_agent.py` adds 52 new tests covering the
decorator + schema generators (Phase 1), every inspection tool
(Phase 2), and every recipe kind + end-to-end replay roundtrip
(Phase 3). `tests/provenance/test_mcp.py` updated to assert the
v0.3 nav tools still register AND the new v0.4 inspection tools
land.

Tests: 659 -> 702 passing. Audit: 0 undocumented, 28 cataloged.

## [0.3.0] - 2026-05-27

Provenance module sweep: performance, robustness, visualization, and
an agent-access layer (MCP + Jupyter). Ten commits across four phases.
The construction side (decorator, record(), hex) was already mature;
this release rounds out the consumption side and addresses the two
biggest perf gaps (hashing cost on big tensors, memory growth from
cache_tensors). Tests: 622 -> 659 passing.

### Added: performance

- **Sketch hashing (opt-in)**: `record(hash_mode="sketch")` and
  `ProvenanceRegistry(hash_mode="sketch")`. Hashes
  `shape + dtype + SKETCH_SAMPLES = 64` strided samples + `sum` + `std`
  instead of the full tensor bytes. **15× faster** on 8 MB (n=1024
  float64) inputs; crossover with full mode at ~n=256. Hexes are not
  portable across modes; the chosen mode is stamped in to_dict() and
  round-trips through save()/load().
- **On-disk tensor cache**: `record(cache_to_disk=path)` mirrors every
  cached output to a directory via torch.save. Memory eviction under
  `max_cache_size` drops the in-memory copy but the disk copy persists
  and `get_tensor()` reloads on demand. New
  `ProvenanceRegistry.clear(delete_disk=False)` cleans the in-memory
  caches and optionally the disk files. ~2.3× I/O overhead vs memory-
  only caching at n=1024.
- **Extended benchmark suite**: `tests/benchmarks/bench_provenance.py`
  now covers hashing scale (n=16/256/1024), cache + replay overhead,
  and sketch/disk variants. Baseline + post-Phase-1 numbers preserved
  in `notes/benchmark_provenance_v0.2.md` and `_v0.3.md`.

### Added: robustness

- **Replay completion**: `ProvenanceRegistry.replay()` now works for
  class-method calls and ops taking tuple-of-tensor inputs (e.g.,
  `FixedRankPoint = (U, S, Vt)`). New
  `@provenance.register_provenance_class("FixedRankManifold")`
  decorator opts a class into replay; classes must define
  `_from_signature(cls, sig: dict)` as the inverse of
  `_provenance_signature`. `FixedRankManifold` and `SPDManifold` are
  registered out of the box. Unregistered classes still hit a clear
  `NotImplementedError` with actionable text.
- **User-input caching for replay**: when `cache_tensors=True`,
  user-supplied input tensors are now cached in a separate
  `_user_input_cache` (not bounded by `max_cache_size`) so replay
  can find them. Without this, chains like `mfd.exp(S, V)` where S
  is a user tensor couldn't replay past the substitution.
- **Op-version drift detector on load()**: new
  `ProvenanceVersionWarning` (subclass of UserWarning) is emitted
  when any loaded node's `op_version` differs from the currently-
  installed `OP_REGISTRY[op_id]` version. New `strict: bool = False`
  kwarg converts to ValueError. Unknown op_ids (recorded by a module
  not imported in this process) are flagged in the same diagnostic.

### Added: consumption / visualization

- **Mermaid + Graphviz exports**: `to_mermaid()` returns a flowchart
  string suitable for inline GitHub Markdown / JupyterLab; `to_graphviz()`
  returns DOT source. Neither imports the optional rendering library
  (output is a string the caller pipes to whatever renderer they
  prefer.
- **diff_summary(other)**: human-readable rendering of `diff()`,
  bucketed into Cache hits / Drift / Only in self / Only in other.
  Drift (same op_id, different inputs) is the interesting category
  for "did my refactor preserve semantics" questions.
- **ancestors_with_tensors(hex)**: returns
  `dict[hex, (ProvenanceNode, Optional[Tensor])]`. One call instead
  of `ancestors()` + N `get_tensor()` calls.

### Added: agent access

- **to_llm_context(max_ops, show_shapes, show_params)**: compact text
  summary suitable for placing in an LLM agent's prompt. Format:
  header (op + cached tensor counts, hash_mode), Ops by op_id with
  counts, Notable shapes, Roots (no provenance-internal parents),
  Leaves (no provenance-internal consumers).
- **MCP server (`holonomy_lib.provenance.mcp`, optional extra)**:
  exposes a saved registry as six MCP tools (`list_ops`, `where`,
  `node_info`, `ancestors`, `get_tensor_summary`, `replay`). Entry
  point: `python -m holonomy_lib.provenance.mcp` reads the registry
  from `HOLONOMY_PROVENANCE_REGISTRY` env var, starts the server on
  stdio. Install via `pip install 'holonomy-lib[mcp]'`. File-loaded
  registries only in v0.3; live-process attachment is v0.4.
- **Jupyter cell magic (`holonomy_lib.provenance.jupyter`, optional
  extra)**: `%load_ext holonomy_lib.provenance.jupyter` enables
  `%%record_provenance` which wraps a cell in `record()`, binds the
  registry to `_prov` (or a custom name via the magic line), and
  renders the Mermaid DAG below the cell output. Install via
  `pip install 'holonomy-lib[jupyter]'`.

### Changed

- `ProvenanceRegistry.to_dict()` schema bumped 0.2 -> 0.3; older saved
  registries default to `hash_mode="full"` and `cache_to_disk=None`
  on load for backward compat.
- `SKETCH_SAMPLES = 64` cataloged in `notes/magic_numbers.md` with
  scale-of-validity (empirically zero collisions on 200 random 16² and
  no known structural collisions on Laplacian inputs).

### New optional-extras groups

- `[mcp]`: `mcp>=0.9` for the MCP server.
- `[jupyter]`: `ipython` for the cell magic.

## [0.2.1] - 2026-05-27

Packaging-metadata-only release. No code changes.

### Changed

- `pyproject.toml` description synced with the GitHub repo description:
  adds cellular sheaves, SO(3) Lie primitives, and information geometry
  to the listed capabilities (these modules shipped in 0.2.0 but the
  description hadn't caught up).

## [0.2.0] - 2026-05-27

Six new modules and several extensions on top of the v0.1.0 seed.
Tests: 543 → 613 passing. Module count: 6 → 12.

### Added: new modules

- **`optimization`**: Riemannian optimizers wrapping the existing
  manifold `projection` + `retraction` API.
  - `RiemannianSGD(manifold, lr)` + functional `riemannian_sgd_step`.
  - Works with `FixedRankManifold` (SVD-triple state) and
    `SPDManifold` (square-matrix state).
  - Adam / RMSProp deliberately omitted: adaptive preconditioning is
    user-side ergonomics, not part of the math of optimization on a
    manifold. Compose your own from `manifold.projection` +
    `retraction` + `torch.optim.Adam` buffers.
  - Refs: Absil-Mahony-Sepulchre (2008) §4.1; Bonnabel (2013).

- **`simplicial`**: Simplicial complex data structures + boundary
  operators + Vietoris-Rips construction.
  - `DenseSimplicialComplex`: batched, padded `(B, n_k_max, k+1)`
    simplex tables with validity masks. `boundary(k)` returns the
    dense Koszul-signed boundary matrix.
  - `SparseSimplicialComplex`: single-instance, sparse-CSC boundary
    matrices for the persistent-homology reduction kernel.
  - `vietoris_rips_dense` (batched) + `vietoris_rips_sparse`
    (single-instance) + `pairwise_distances`.
  - Refs: Munkres (1984); Hausmann (1995); Bauer (2021).

- **`topology`**: Hodge Laplacians + Betti numbers + batched
  persistent homology on simplicial complexes.
  - `hodge_laplacian(complex, k)`: `L_k = ∂_{k+1} ∂_{k+1}^T + ∂_k^T ∂_k`,
    kernel-dim = k-th Betti.
  - `betti_numbers(complex, max_dim)`: near-zero eigenvalue counting
    on each `L_k`. Closed-form verified: S¹ → (1, 1), S² → (1, 0, 1),
    T² → (1, 2, 1).
  - `persistence_diagrams(points, max_dim=2, max_radius=inf)`: batched
    H₀ + H₁ + H₂ on Vietoris-Rips filtrations. H₀ via union-find on
    sorted edges; H₁/H₂ via Z/2 left-to-right boundary-matrix reduction
    with Bauer-Kerber-Reininghaus clearing.
  - `reduction_backend="torch"` runs end-to-end on the filtration's
    device (CPU or GPU); current path is a same-algorithm torch port
    (not yet a custom CUDA kernel) and is ~21× slower than CPython
    sets at n=80; the GPU win is a v0.3 follow-up.
  - Refs: Eckmann (1944); Lim (2020); Schaub et al. (2020);
    Edelsbrunner-Letscher-Zomorodian (2002); Cohen-Steiner-Edelsbrunner-
    Harer (2007) stability.

- **`info_geometry`**: divergences on probability distributions.
  - `bregman_divergence(p, q, potential)` for any convex generator.
  - `kl_divergence_categorical(p, q)`, `kl_divergence_gaussian(...)`
    (Cholesky-stable closed-form).
  - Fisher information metric + natural gradient added in the v0.1
    roadmap sweep.
  - Refs: Bregman (1967); Banerjee et al. (2005); Amari (2016);
    Cover-Thomas (2006).

- **`sheaf`**: cellular sheaves on graphs and their Laplacians.
  - `GraphSheaf` dataclass; `sheaf_coboundary`, `sheaf_laplacian`
    (`δ^T δ`, PSD), `sheaf_dirichlet_energy`.
  - Reduces to the standard combinatorial graph Laplacian under
    trivial stalks. Orientation-flip on a 3-cycle correctly drops
    kernel dim 1 → 0 (the monodromy test).
  - v1 is dense-only with `SHEAF_DENSE_BYTES_CAP = 2 GiB` pre-flight
    guard; node-edge sheaves on graphs only (higher-dim cellular
    sheaves on simplicial complexes planned).
  - Rejects self-loops + duplicate edges at construction (callers
    must pre-process).
  - Refs: Hansen-Ghrist (2019); Bodnar et al. (2022) Neural Sheaf
    Diffusion; Curry (2014).

- **`lie`**: SO(3) primitives + real spherical harmonics.
  - `so3.axis_angle_to_matrix` (Rodrigues), `matrix_to_axis_angle`
    (dual-branch log: trace-based away from π, quaternion-based
    near π).
  - `so3.so3_exp` / `so3_log` on so(3) (3×3 skew matrices).
  - `so3.random_so3(batch_size, generator)`: Haar-uniform via the
    quaternion-from-3-uniforms construction (Shoemake 1992); chi-squared
    sanity test with p < 1e-6 bound in the suite.
  - `so3.compose(R1, R2)`: group product.
  - `real_spherical_harmonics(directions, l_max)`: closed-form Y_lm
    for `l_max ≤ 4`. Per-l block norm preserved under SO(3) rotation
    (full mixing via Wigner-D matrices is a v0.3 follow-up).
  - `SO3_LOG_NEAR_PI_RAD = 1e-7` is empirically calibrated for
    **float64**; do not change to e3nn / pytorch3d's `1e-2` (correct
    for float32) without re-running the empirical comparison in the
    `so3.py` docstring.
  - Refs: Hall (2015) §3.1; Shoemake (1992); Edmonds (1957);
    Cohen et al. (2018) Spherical CNNs.

### Added: extensions to existing modules

- **`spectral` additions**:
  - `magnetic.*`: magnetic Laplacian for directed graphs (Furutani
    2020), with sign-magnetic extension for signed-directed graphs
    (Fiorini 2023; He et al. 2023).
  - `heat_kernel_chebyshev`: Hammond-Vandergheynst-Gribonval (2011)
    Chebyshev-polynomial heat kernel.
  - `effective_resistance`, `commute_time` (Klein-Randić 1993).
  - `diffusion_map` (Coifman-Lafon 2006).
  - Sparse-COO/CSR/CSC paths for all four Laplacian variants
    (combinatorial, symmetric-normalized, random-walk, Kunegis signed);
    end-to-end with sparse `lanczos_eigsh`.

- **`algebra` additions**:
  - `lanczos_eigsh(A, k, which="LA"|"SA", sigma=σ, n_iter=…)`: Lanczos
    top-k eigensolver with full reorthogonalization (Paige 1972).
  - `"SA"` mode is shift-and-invert (Ericsson-Ruhe 1980): factor once
    outside the iteration, each step is a `lu_solve` against
    `A − σI`. Raises `RuntimeError("shift-invert breakdown")` if σ
    coincides with an eigenvalue. For graph Laplacians (which have 0
    in spectrum), use a small negative shift.
  - Sparse-input dispatch on the same API.

- **`discrete_geometry` additions**:
  - `forman_ricci_simple` + `forman_ricci_augmented`: combinatorial
    Forman-Ricci curvature (Sreejith et al. 2016; Samal et al. 2018).
  - Performance: pair tiling (`SINKHORN_TILE_DEFAULT = 256`) +
    sync-cadence-every-8-iters → 13× speedup on Ollivier-Ricci at
    n=64 (22.6s → 1.7s).

- **`provenance` additions**:
  - Class-method support for `FixedRankManifold` / `SPDManifold`
    methods; `self` is canonicalized into a provenance-signature
    dict. `replay()` raises `NotImplementedError` for class-method
    nodes (the canonicalization isn't reversible) and for
    tuple-of-tensors inputs (unpacked `point[i]` keys don't
    reassemble); both have explicit error messages.
  - `blake3` opt-in (faster hash), hooks API, SAELens-style dataset
    emission, diff API, persistence (`save` / `load`).

### Changed

- **Self-loops dropped at entry to ALL graph primitives** (codified
  in `CONVENTIONS.md`). Sheaf v1 also rejects duplicate edges at
  `GraphSheaf` construction time; call sites must pre-process.
- `FixedRankManifold.retraction` auto-switches to Halko-Martinsson-
  Tropp randomized SVD at low `r/min(m, n)` ratios (the common case);
  25× speedup at 1024×1024×32 (193 ms → 7.6 ms).

### Notes

- The `v0.1.0` git tag was placed mid-development, before the full
  "v0.1 roadmap" cycle completed. Everything from `optimization`
  onward in the list above actually ships in this `v0.2.0` release.

## [0.1.0] - 2026-05-26

Initial public release. Six seed modules:

- **`manifolds`**: `FixedRankManifold` (Vandereycken 2013),
  `SPDManifold` (Pennec et al. 2006 affine-invariant metric).
- **`algebra`**: `truncated_svd` exact (Eckart-Young) + randomized
  (Halko-Martinsson-Tropp 2011).
- **`tensor_calculus`**: `hosvd`, `mode_product`, `mode_unfolding`
  (Kolda-Bader 2009).
- **`spectral`**: combinatorial / symmetric-normalized / random-walk
  / signed Laplacians (Chung; von Luxburg; Kunegis); batched
  `laplacian_eigenmaps` embedding.
- **`discrete_geometry`**: `ollivier_ricci_curvature` via batched
  log-domain Sinkhorn on all-pairs shortest paths (Ollivier 2009;
  Cuturi 2013); `discrete_ricci_flow` + `ricci_flow_with_surgery`
  ("Perelman on networks", Sia et al. 2019; Ni-Lin-Luo-Gao 2019).
- **`provenance`**: content-addressable Merkle DAG of math
  primitives, `@with_provenance` decorator, `record()` context
  manager, `ProvenanceRegistry` with substitution / replay /
  SAELens-style emission for mechanistic interpretability.

- `audit.py`: build-gate enforcing the no-magic-numbers discipline:
  every numerical literal must be derived, a universal invariant, or
  cataloged in `notes/magic_numbers.md` with scale-of-validity.

[0.4.0]: https://github.com/superninjv/holonomy_lib/compare/v0.3.0...v0.4.0
[0.3.0]: https://github.com/superninjv/holonomy_lib/compare/v0.2.1...v0.3.0
[0.2.1]: https://github.com/superninjv/holonomy_lib/compare/v0.2.0...v0.2.1
[0.2.0]: https://github.com/superninjv/holonomy_lib/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/superninjv/holonomy_lib/releases/tag/v0.1.0
