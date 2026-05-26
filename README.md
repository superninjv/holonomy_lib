# holonomy_lib

> **A research-grade PyTorch math library**: GPU-native, batched-first,
> audit-clean, with every primitive grounded in a citation. Differential
> geometry, spectral graph theory, discrete Ricci flow, tensor
> decompositions, and content-addressable provenance for mechanistic
> interpretability, all under one roof.

[![License: BSD-3-Clause](https://img.shields.io/badge/License-BSD%203--Clause-blue.svg)](https://opensource.org/licenses/BSD-3-Clause)
[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.x](https://img.shields.io/badge/PyTorch-2.x-ee4c2c.svg)](https://pytorch.org/)
[![tests: 269 passing](https://img.shields.io/badge/tests-269%20passing-brightgreen.svg)](#testing)
[![audit: clean](https://img.shields.io/badge/audit-clean-brightgreen.svg)](#audit-discipline)

---

## What this is

A consolidated PyTorch math library for research at the intersection of
**differential geometry**, **spectral graph theory**, and **mechanistic
interpretability**: the mathematics that modern ML keeps reinventing
project by project. Six modules, 269 tests, every numerical constant
derived or cataloged with a scale-of-validity, every primitive cited to
the paper that defines it.

The name **holonomy** comes from differential geometry: the
transformation a vector accumulates when parallel-transported around a
closed loop. It captures the library's character: every operation
defined by the geometry it preserves, every result traceable back to
its inputs through a content-addressable provenance DAG.

| Module | Primitives | What it gives you |
|---|---|---|
| `manifolds` | `FixedRankManifold`, `SPDManifold` | Riemannian geometry on low-rank matrices and SPD cones (Vandereycken 2013; Pennec et al. 2006) |
| `algebra` | `truncated_svd` (exact + randomized) | Halko-Martinsson-Tropp randomized SVD; Eckart-Young exact |
| `tensor_calculus` | `hosvd`, `mode_product`, `mode_unfolding` | Truncated HOSVD with Kolda-Bader n-mode product |
| `spectral` | `combinatorial`/`symmetric_normalized`/`random_walk`/`signed` Laplacians, `laplacian_eigenmaps`, **`magnetic.*`** (directed graphs), **`heat_kernel_chebyshev`** | Chung; von Luxburg; Kunegis (signed); Furutani 2020 (magnetic Hermitian Laplacian); Hammond-Vandergheynst-Gribonval 2011 (Chebyshev heat kernel) |
| `discrete_geometry` | `ollivier_ricci_curvature`, `discrete_ricci_flow`, `ricci_flow_with_surgery`, **`forman_ricci_simple`**, **`forman_ricci_augmented`** | Sinkhorn-W₁ Ollivier on graphs (Ollivier 2009; Cuturi 2013; Sia/Ni-Lin-Luo-Gao 2019), the **Perelman-on-networks** flow + surgery primitive, and the cheap combinatorial Forman alternative (Sreejith et al. 2016; Samal et al. 2018) |
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

| | this lib | geoopt | geomstats | pymanopt | gudhi |
|---|:-:|:-:|:-:|:-:|:-:|
| Riemannian manifolds | ✓ | ✓ | ✓ | ✓ | – |
| Spectral graph theory | ✓ | – | – | – | – |
| Ollivier-Ricci curvature | ✓ | – | – | – | – |
| Tucker / HOSVD | ✓ | – | – | – | – |
| GPU-native (PyTorch) | ✓ | ✓ | partial | – | – |
| Batched-first | ✓ | ✓ | partial | – | – |
| Content-addressable provenance | ✓ | – | – | – | – |
| Audit / no-magic-numbers | ✓ | – | – | – | – |

---

## Installation

```bash
# Install PyTorch separately per platform; we don't pin in pyproject.toml
# because uv's resolver tends to prefer PyPI CUDA-13 wheels even when ROCm
# is configured.
#
#   Workstation (AMD ROCm 6.4):
#     uv pip install --index-url https://download.pytorch.org/whl/rocm6.4 torch
#
#   NVIDIA CUDA 12+:
#     uv pip install torch
#
#   CPU only:
#     uv pip install torch

# Then install the library:
uv pip install -e ".[dev]"
```

Python ≥ 3.12, PyTorch 2.x.

Optional extras: `geometry` (geoopt, geomstats, pymanopt, used by
comparison tests), `topology` (gudhi, ripser), `optimization` (higher,
torchopt), `dev` (pytest, ruff, mypy).

---

## Quick start

```python
import torch
from holonomy_lib.manifolds import SPDManifold
from holonomy_lib.spectral import laplacian, laplacian_eigenmaps
from holonomy_lib.discrete_geometry import ricci_flow_with_surgery
from holonomy_lib import provenance

# 1. Riemannian geometry on SPD covariance matrices
mfd = SPDManifold(n=8, dtype=torch.float64)
S = mfd.random_point(batch_size=4)       # (4, 8, 8) SPD
T = mfd.random_point(batch_size=4)
d = mfd.distance(S, T)                    # affine-invariant geodesic
V = mfd.log(S, T)                         # Lie-algebra-style log
T_recon = mfd.exp(S, V)                   # exp_S(log_S(T)) ≈ T

# 2. Graph spectral embedding
A = (torch.rand(1, 50, 50) > 0.7).double()
A = (A + A.mT) * 0.5                      # symmetrize
vals, vecs = laplacian_eigenmaps(A, k=4, laplacian_type="symmetric_normalized")

# 3. Perelman-on-networks: community detection via Ricci flow + surgery
A_after = ricci_flow_with_surgery(
    A, n_steps=20, surgery_period=5, surgery_threshold=3.0,
    dt=0.5, alpha=0.0,
)
# Disconnected components in A_after correspond to detected communities.

# 4. Mech-interp-style provenance: every primitive emits a Merkle DAG node
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

1. **Persistent homology on GPU**: the biggest standing gap across the
   field; gudhi / ripser are CPU.
2. **Lanczos sparse eigensolver** for top-k eigenpairs on large graphs.
3. **Hodge Laplacians** on simplicial complexes (Ribando-Gros et al.
   2024).
4. **Riemannian optimizers** (SGD, Adam, trust-region) on the manifolds
   module.
5. **Information geometry**: conjugate priors, Bregman divergences.
6. **Sign-magnetic Laplacian** for signed-directed graphs (Fiorini
   2023; He et al. 2023). The plain magnetic Laplacian is in; the
   signed extension treats negative edges with a separate phase.
7. **Effective resistance / commute-time distances** on graphs.
8. **Diffusion maps** built on the Chebyshev heat kernel.

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

Contributions welcome. Hard constraints (binding for any code in this
repo):

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
