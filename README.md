# holonomy_lib

> **A research-grade PyTorch math library**: GPU-native, batched-first,
> audit-clean, with every primitive grounded in a citation.
> Differential geometry, spectral graph theory, discrete Ricci flow,
> tensor decompositions, Riemannian optimization, simplicial topology,
> batched persistent homology, and content-addressable provenance for
> mechanistic interpretability, all under one roof. Developed by
> independent and Synoros researchers for the *substrate* research.

[![License: BSD-3-Clause](https://img.shields.io/badge/License-BSD%203--Clause-blue.svg)](https://opensource.org/licenses/BSD-3-Clause)
[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.x](https://img.shields.io/badge/PyTorch-2.x-ee4c2c.svg)](https://pytorch.org/)
[![tests: 457 passing](https://img.shields.io/badge/tests-457%20passing-brightgreen.svg)](#testing)
[![audit: clean](https://img.shields.io/badge/audit-clean-brightgreen.svg)](#audit-discipline)

---

## What this is

A consolidated PyTorch math library for research at the intersection of
**differential geometry**, **spectral graph theory**, **computational
topology**, and **mechanistic interpretability**: the mathematics that
modern ML keeps reinventing project by project. Nine modules, 457
tests, every numerical constant derived or cataloged with a
scale-of-validity, every primitive cited to the paper that defines it.

The name **holonomy** comes from differential geometry: the
transformation a vector accumulates when parallel-transported around a
closed loop. It captures the library's character: every operation
defined by the geometry it preserves, every result traceable back to
its inputs through a content-addressable provenance DAG.

| Module | Primitives | What it gives you |
|---|---|---|
| `manifolds` | `FixedRankManifold`, `SPDManifold` | Riemannian geometry on low-rank matrices and SPD cones (Vandereycken 2013; Pennec et al. 2006) |
| `algebra` | `truncated_svd` (exact + randomized), **`lanczos_eigsh`** | Halko-Martinsson-Tropp randomized SVD; Eckart-Young exact; Lanczos top-k eigensolver with full reorthogonalization (Paige 1972) |
| `tensor_calculus` | `hosvd`, `mode_product`, `mode_unfolding` | Truncated HOSVD with Kolda-Bader n-mode product |
| `spectral` | `combinatorial`/`symmetric_normalized`/`random_walk`/`signed` Laplacians, `laplacian_eigenmaps`, `magnetic.*` (directed), `heat_kernel_chebyshev`, **`effective_resistance`**, **`commute_time`**, **`diffusion_map`** | Chung; von Luxburg; Kunegis (signed); Furutani 2020 (magnetic Hermitian); Hammond-Vandergheynst-Gribonval 2011 (Chebyshev heat kernel); Klein-Randić 1993 (resistance); Coifman-Lafon 2006 (diffusion maps) |
| `discrete_geometry` | `ollivier_ricci_curvature`, `discrete_ricci_flow`, `ricci_flow_with_surgery`, `forman_ricci_simple`, `forman_ricci_augmented` | Sinkhorn-W₁ Ollivier on graphs (Ollivier 2009; Cuturi 2013; Sia/Ni-Lin-Luo-Gao 2019), the **Perelman-on-networks** flow + surgery primitive, and the cheap combinatorial Forman alternative (Sreejith et al. 2016; Samal et al. 2018) |
| **`info_geometry`** | **`bregman_divergence`**, **`kl_divergence_categorical`**, **`kl_divergence_gaussian`** | Bregman divergence for any convex generator plus closed-form KL for the standard exponential families (Bregman 1967; Banerjee et al. 2005; Amari 2016) |
| **`optimization`** | **`RiemannianSGD`** | Steepest descent on `FixedRankManifold` / `SPDManifold` via the existing projection + retraction API (Absil-Mahony-Sepulchre 2008, §4.1) |
| **`simplicial`** | **`DenseSimplicialComplex`**, **`SparseSimplicialComplex`**, **`vietoris_rips_*`** | Simplicial complex data structures + boundary operators + Vietoris-Rips construction; foundation for Hodge + persistent homology (Munkres 1984; Hausmann 1995; Bauer 2021) |
| **`topology`** | **`hodge_laplacian`**, **`betti_numbers`**, **`persistence_diagrams`** | Hodge Laplacians + Betti numbers on simplicial complexes (Eckmann 1944; Lim 2020), plus batched persistent homology H₀+H₁+H₂ of Vietoris-Rips filtrations via union-find + Z/2 matrix reduction (Edelsbrunner-Letscher-Zomorodian 2002; Cohen-Steiner-Edelsbrunner-Harer 2007 stability) |
| `provenance` | `@with_provenance`, `record()`, `ProvenanceRegistry` | Content-addressable Merkle DAG of math primitives; substitution / replay / SAELens emission for mech interp |

---

## Why use this

Existing libraries cover slices of what's here, but none cover all four
properties this library guarantees:

1. **Breadth**: Riemannian manifolds, spectral graph theory, tensor
   decompositions, Ricci-curvature, and content-addressable provenance,
   under one import root.
2. **GPU-native, batched-first**: every operation takes a leading
   batch dim, runs on `cuda`/`rocm`/`mps`/`cpu`, verified for `B ∈ {0,
   1, > 1}`.
3. **Audit-clean**: every numerical constant is derived, a universal
   invariant, or experimentally tuned with documented scale-of-validity.
   CI enforces this.
4. **Cited**: every public function has a `References:` section
   pointing to the paper that defines its math. No "trust me"
   implementations.

| | this lib | geoopt | geomstats | pymanopt | gudhi | ripser |
|---|:-:|:-:|:-:|:-:|:-:|:-:|
| Riemannian manifolds + optimizers | ✓ | ✓ | ✓ | ✓ | – | – |
| Spectral graph theory (4+ Laplacians) | ✓ | – | – | – | – | – |
| Magnetic Laplacian (directed graphs) | ✓ | – | – | – | – | – |
| Ollivier-Ricci + Forman-Ricci curvature | ✓ | – | – | – | – | – |
| Discrete Ricci flow + surgery | ✓ | – | – | – | – | – |
| Tucker / HOSVD | ✓ | – | – | – | – | – |
| Chebyshev heat kernel + diffusion maps | ✓ | – | – | – | – | – |
| Simplicial complexes + Hodge Laplacians | ✓ | – | – | – | ✓ | – |
| Batched persistent homology (H₀+H₁+H₂) | ✓ | – | – | – | – | – |
| GPU-native (PyTorch) | ✓ | ✓ | partial | – | – | – |
| Batched-first | ✓ | ✓ | partial | – | – | – |
| Content-addressable provenance | ✓ | – | – | – | – | – |
| Audit / no-magic-numbers | ✓ | – | – | – | – | – |
| Information geometry (Bregman + KL) | ✓ | – | ✓ | – | – | – |

---

## Installation

The library has a small dependency surface: `torch >= 2.0`, `numpy`,
`scipy`. Everything else, Riemannian optimizers, simplicial complexes,
Hodge Laplacians, persistent homology, is shipped natively. You do
**not** need to install `pymanopt`, `geoopt`, `gudhi`, `ripser`, or
similar to use the corresponding primitives.

### Standard install (CPU or default CUDA)

```bash
pip install holonomy-lib
```

This pulls torch's default wheel (CPU or CUDA 12, depending on
platform) automatically. Python ≥ 3.12.

### ROCm / older CUDA / specific torch wheel

Install your preferred torch wheel **first**, then the library. pip /
uv will respect the already-installed torch:

```bash
# AMD ROCm 6.4:
pip install --index-url https://download.pytorch.org/whl/rocm6.4 torch
pip install holonomy-lib

# CUDA 12.1 specifically:
pip install --index-url https://download.pytorch.org/whl/cu121 torch
pip install holonomy-lib

# CPU only:
pip install --index-url https://download.pytorch.org/whl/cpu torch
pip install holonomy-lib
```

### From source (development)

```bash
git clone https://github.com/superninjv/holonomy_lib
cd holonomy_lib
uv venv
uv pip install -e ".[dev]"
```

### Optional extras

- `holonomy-lib[provenance-extras]`: `blake3` (faster hash), `networkx`
  (DAG export), `pandas` (DataFrame export). Used only inside specific
  provenance helpers; the library degrades gracefully without them.
- `holonomy-lib[comparison]`: pymanopt, geoopt, geomstats, tensorly,
  gudhi, ripser, GraphRicciCurvature, networkx, autograd. Required
  ONLY for running the cross-comparison test suite locally; the
  library itself never imports these.
- `holonomy-lib[dev]`: pytest, ruff, mypy.
- `holonomy-lib[all]`: provenance-extras + dev (the typical
  contributor install).

---

## Quick start

```python
import torch
from holonomy_lib.manifolds import SPDManifold
from holonomy_lib.optimization import RiemannianSGD
from holonomy_lib.spectral import laplacian, laplacian_eigenmaps
from holonomy_lib.discrete_geometry import ricci_flow_with_surgery
from holonomy_lib.topology import persistence_diagrams
from holonomy_lib import provenance

# 1. Riemannian geometry on SPD covariance matrices
mfd = SPDManifold(n=8, dtype=torch.float64)
S = mfd.random_point(batch_size=4)       # (4, 8, 8) SPD
T = mfd.random_point(batch_size=4)
d = mfd.distance(S, T)                    # affine-invariant geodesic
V = mfd.log(S, T)                         # Lie-algebra-style log
T_recon = mfd.exp(S, V)                   # exp_S(log_S(T)) ≈ T

# 2. Riemannian gradient descent ON the SPD manifold
opt = RiemannianSGD(mfd, lr=0.5)
point = S.clone()
for _ in range(50):
    ambient_grad = -mfd.log(point, T)     # gradient of (1/2) d(point, T)^2
    point = opt.step(point, ambient_grad)
# `point` now sits on the SPD manifold, close to T.

# 3. Graph spectral embedding
A = (torch.rand(1, 50, 50) > 0.7).double()
A = (A + A.mT) * 0.5                      # symmetrize
vals, vecs = laplacian_eigenmaps(A, k=4, laplacian_type="symmetric_normalized")

# 4. Perelman-on-networks: community detection via Ricci flow + surgery
A_after = ricci_flow_with_surgery(
    A, n_steps=20, surgery_period=5, surgery_threshold=3.0,
    dt=0.5, alpha=0.0,
)
# Disconnected components in A_after correspond to detected communities.

# 5. Batched persistent homology on point clouds
points = torch.randn(8, 30, 2, dtype=torch.float64)   # 8 point clouds of 30 pts
diagrams, masks = persistence_diagrams(
    points, max_dim=2, max_radius=2.5,
)
# diagrams[0]: (8, max_h0, 2)  birth/death pairs for H_0 (components)
# diagrams[1]: (8, max_h1, 2)  for H_1 (loops)
# diagrams[2]: (8, max_h2, 2)  for H_2 (voids)
# masks[k] tells you which pair-slots are valid per batch element.

# 6. Mech-interp-style provenance: every primitive emits a Merkle DAG node
with provenance.record(cache_tensors=True) as reg:
    L = laplacian.combinatorial(A)
    vals, vecs = laplacian_eigenmaps(A, k=4)

# Look up any operation by content-addressable hex
for node in reg:
    print(f"{node.hex}  {node.op_id}")
```

See `CONTENTS.md` for the complete inventory of primitives, signatures,
and citations.

---

## Performance

Benchmarks: `notes/benchmark_baseline.md` (before optimization) and
`notes/benchmark_optimized.md` (current). All times CPU, single-thread,
PyTorch 2.12, float64.

### Highlight: discrete Ricci curvature

The signature primitive, Ollivier-Ricci curvature via batched
log-domain Sinkhorn over all-pairs shortest-path costs, got two
optimizations:

- **Pair tiling**: the Sinkhorn dual update used to materialize a `(B,
  n², n, n)` intermediate (128 MB per iter at n=64). Tiled
  implementation processes pairs in chunks of `SINKHORN_TILE_DEFAULT =
  256`, capping the inner broadcast at ~16 MB.
- **Sync cadence**: the `.item()` convergence check used to fire every
  iter, forcing a GPU→CPU sync. Now checks every 8 iters; same
  asymptotic work, 8× fewer host syncs.

| graph size (n) | before | after | **speedup** |
|---:|---:|---:|---:|
| 16  | 34.0 ms | 18.0 ms | 1.9× |
| 32  | 273 ms  | 133 ms  | 2.1× |
| **64** | **22.6 s** | **1.7 s** | **13×** |

### Highlight: Riemannian retraction on low-rank manifolds

`FixedRankManifold.retraction` used to do a full SVD then slice top-r.
At low `r/min(m, n)` ratios (the common case for the fixed-rank
manifold), it now auto-switches to Halko-Martinsson-Tropp randomized
SVD with documented oversampling.

| m × n × r | before (full SVD) | after (auto) | **speedup** |
|---|---:|---:|---:|
| 64 × 64 × 8 | 0.31 ms | 0.31 ms | 1.0× (parity) |
| 256 × 256 × 16 | 7.4 ms | 1.3 ms | 5.8× |
| **1024 × 1024 × 32** | **193 ms** | **7.6 ms** | **25×** |

### Highlight: Lanczos vs dense `eigh` on big symmetric matrices

The library's `algebra.lanczos_eigsh` with full reorthogonalization
beats `torch.linalg.eigvalsh` once the matrix is big enough that
computing the full spectrum becomes wasteful. Single-batch top-1
eigenvalue at CPU, float64:

| n | dense `eigvalsh` | `lanczos_eigsh` (n_iter=30) | **speedup** |
|---:|---:|---:|---:|
| 128 | 0.44 ms | 2.66 ms | 0.2× (Lanczos overhead dominates) |
| 512 | 7.62 ms | 4.84 ms | 1.6× |
| **1024** | **46.5 ms** | **11.0 ms** | **4.2×** |

The same `lanczos_eigsh` accepts sparse-CSC inputs (via the dispatch
added in Phase 3), so it's the natural top-k path on the sparse-Hodge
Laplacians produced by the `topology` module.

### Highlight: Batched persistent homology

`topology.persistence_diagrams` computes H₀ + H₁ + H₂ for a batch of
point clouds in parallel. H₀ runs via union-find on sorted filtration
edges (no boundary-matrix reduction needed). H₁ and H₂ use Z/2
left-to-right reduction (Edelsbrunner-Letscher-Zomorodian 2002) on
sparse-CSC boundary matrices, batching across point clouds.

Closed-form verification: a noisy 30-point unit circle reliably
recovers one persistent H₁ bar (the loop) with persistence > 0.2 in
the default `max_radius` range; the bottleneck stability theorem
(Cohen-Steiner-Edelsbrunner-Harer 2007) is verified under
ε-perturbation in the test suite.

---

## Audit discipline

Every numerical constant must be in one of three categories:

| Category | Example |
|---|---|
| ✅ **Derived from inputs** | `1 / N` for normalization; `1 / sqrt(d)` for Laplacian normalization |
| ⚖️ **Universal invariant** | `1e-9` numerical floor; `0.5` halving; `2π`; `1024` (KB↔MB) |
| 🔬 **Experimentally tuned** | `SINKHORN_TILE_DEFAULT = 256`, cataloged with scale-of-validity |

Each constant in category 🔬 has a row in [`notes/magic_numbers.md`](notes/magic_numbers.md)
with the procedure used to pick it, the regime where it's valid, and what
to re-derive when scale changes. The audit tool
(`python -m holonomy_lib.audit src/ --strict`) is run in CI; it fails
the build on any uncataloged literal.

---

## Provenance for mechanistic interpretability

Every public primitive is decorated with `@with_provenance`. Inside a
`provenance.record()` block, calls emit Merkle-DAG nodes whose hex
identity is `hash(op_id || op_version || canonical(params) ||
input_hexes)`. Same op, same inputs ⇒ same hex (deterministic, content-
addressable).

This unlocks TransformerLens-style activation patching and SAELens-
style dataset emission for the mathematical primitives, not just neural
network internals:

```python
with provenance.record(cache_tensors=True) as reg:
    out = pipeline(A)

# Find a specific operation
lap_node = reg.where(op_id="holonomy_lib.spectral.laplacian.combinatorial")[0]

# Ablation: substitute zeros and replay only the downstream computation
new = reg.replay({lap_node.hex: torch.zeros_like(reg.get_tensor(lap_node.hex))})
# `new` contains the re-executed outputs of every node downstream of the substitution.

# Emit a SAELens-style dataset for training feature extractors
for tensor, metadata in reg.to_sae_dataset(op_id="holonomy_lib.algebra.linear.truncated_svd"):
    yield tensor, metadata
```

Pluggable hash function (blake3 if installed, else SHA-256). Persist
the DAG with `reg.save(path)` / `ProvenanceRegistry.load(path)`.

---

## Testing

```bash
# Full test suite
uv run pytest

# Just one module
uv run pytest tests/manifolds

# Run the audit (build gate)
uv run python -m holonomy_lib.audit src/ --strict

# Run benchmarks (excluded from the test suite; runs on demand)
uv run python -m tests.benchmarks.run --out notes/benchmark_latest.md

# Run on a GPU machine; parity tests light up automatically
uv run pytest tests/test_device_parity.py
```

**Comparison tests** run against established libraries when installed:
`pymanopt` for FixedRankManifold, `geoopt` for SPDManifold, `tensorly`
for HOSVD, `scipy.sparse.csgraph` for Laplacians,
`GraphRicciCurvature` + `networkx` for Ollivier-Ricci. The tests skip
silently if a comparison library isn't installed.

---

## Project structure

```
holonomy_lib/
├── src/holonomy_lib/          # the library
│   ├── manifolds/             # FixedRankManifold, SPDManifold
│   ├── algebra/               # truncated_svd
│   ├── tensor_calculus/       # hosvd, mode_product, mode_unfolding
│   ├── spectral/              # Laplacians, eigenmaps
│   ├── discrete_geometry/     # Ollivier-Ricci, flow, surgery
│   ├── provenance/            # content-addressable hex protocol
│   └── audit.py               # CI gate: no magic numbers
├── tests/                     # 269 tests across all modules
│   └── benchmarks/            # device-agnostic timing harness
├── notes/
│   ├── magic_numbers.md       # cataloged constants with scale-of-validity
│   ├── scrutiny.md            # findings + fixes from review passes
│   ├── benchmark_baseline.md  # before optimization
│   └── benchmark_optimized.md # after
└── CONTENTS.md                # primitive inventory and quick reference
```

---

## Roadmap

Open frontiers, prioritized by research leverage:

1. **Sign-magnetic Laplacian** for signed-directed graphs (Fiorini
   2023; He et al. 2023). The plain magnetic Laplacian is in; the
   signed extension treats negative edges with a separate phase.
2. **Shift-and-invert Lanczos** for smallest-eigenvalue mode (current
   `lanczos_eigsh` is largest-algebraic only).
3. **Fisher information metric** and natural-gradient optimizers
   building on `info_geometry`.
4. **Riemannian Adam / trust-region** optimizers (SGD is in; adaptive
   schemes belong in user code per design but a built-in variant
   could ship as a convenience wrapper).
5. **Sparse graph backend**: most spectral primitives are currently
   dense `(B, n, n)`. A sparse path would unlock large-graph regimes.
6. **GPU-native H₁/H₂ matrix reduction**: PH currently uses a
   Python-set sparse reduction (sequential per complex; batches across
   point clouds). A GPU kernel for the reduction would unlock larger
   single complexes.
7. **Class-method provenance** for `FixedRankManifold` /
   `SPDManifold` methods (currently only top-level functions are
   decorated).

Contributions welcome via PR; see [Contributing](#contributing).

---

## Citation

If this library helps your research, please cite it:

```bibtex
@software{holonomy_lib,
  author = {Jack},
  title = {holonomy\_lib: GPU-native research math for differential
           geometry, spectral graph theory, and mechanistic interpretability},
  year = {2026},
  url = {https://github.com/superninjv/holonomy_lib},
}
```

The library implements algorithms from many sources; please also cite
the original paper for the specific primitive you use (each public
function lists its references in its docstring).

---

## Contributing

See [`CONVENTIONS.md`](CONVENTIONS.md) for the full coding standards
(batched-first API shape, self-loop policy, numerical conventions,
performance patterns, magic-numbers catalog, citation requirements,
provenance, testing). Contributions welcome. Hard constraints
(binding for any code in this repo):

- **Citations are non-optional.** Every public function has a
  `References:` section pointing to the paper for its math.
- **Every numerical constant has a derivation or a catalog entry** in
  `notes/magic_numbers.md`. The audit tool enforces this.
- **Tests before merge**: unit tests for correctness, property tests
  for invariants, comparison tests against established libraries where
  one exists.
- **GPU-first, batched-first**: operations take a leading batch dim,
  work on `torch.Tensor` on `cuda`/`rocm`/`cpu`. Verify shapes for
  `B ∈ {0, 1, > 1}`.

Open an issue first for non-trivial additions so we can align on
approach.

---

## License

BSD 3-Clause. See [LICENSE](LICENSE).
