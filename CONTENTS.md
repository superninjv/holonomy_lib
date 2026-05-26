# holonomy_lib contents

A flat inventory of every public primitive: signature, one-line summary,
and the paper to cite. See `README.md` for the project overview;
this file is the API map.

Each entry has the form:

> `module.thing(signature) → returns`
> One-line what-it-does. Math citation. Cross-references.

All tensors are batched-first: leading batch dim B, then the math dims.
Shapes use `B` for batch, `n`/`m`/`r`/etc. for math.

## Conventions (binding)

- **Batched-first.** Inputs take leading batch dim. Single-point use = pass `B = 1`.
  Operations are verified for `B ∈ {0, 1, > 1}`.
- **Device-agnostic.** Operations work on whatever device the input tensor is on.
  Use `dtype` and `device` constructor params to pin precision/placement.
- **Cited.** Every primitive's docstring has a `References:` section with the
  paper. The citations below are short; the full docstring has the eq. numbers.
- **Audit-clean.** No magic numerical literals in source. Posited constants
  cataloged in `notes/magic_numbers.md` with scale-of-validity.
- **Provenance-aware.** Top-level primitives are decorated with
  `@with_provenance`, so they emit content-addressable hex IDs when called
  inside `provenance.record()`. See §Provenance.

## Available imports (canonical paths)

```python
from holonomy_lib.manifolds import FixedRankManifold, SPDManifold
from holonomy_lib.algebra import truncated_svd
from holonomy_lib.tensor_calculus import hosvd, mode_product, mode_unfolding
from holonomy_lib.algebra import lanczos_eigsh
from holonomy_lib.spectral import (
    laplacian, magnetic, laplacian_eigenmaps, heat_kernel_chebyshev,
    effective_resistance, commute_time, diffusion_map,
)
from holonomy_lib.discrete_geometry import (
    ollivier_ricci_curvature,
    discrete_ricci_flow,
    ricci_flow_with_surgery,
    forman_ricci_simple,
    forman_ricci_augmented,
)
from holonomy_lib.info_geometry import (
    bregman_divergence,
    kl_divergence_categorical,
    kl_divergence_gaussian,
)
from holonomy_lib.optimization import RiemannianSGD, riemannian_sgd_step
from holonomy_lib.simplicial import (
    DenseSimplicialComplex, SparseSimplicialComplex,
    pairwise_distances, vietoris_rips_dense, vietoris_rips_sparse,
)
from holonomy_lib.topology import (
    betti_numbers, hodge_laplacian, persistence_diagrams,
)
from holonomy_lib import provenance
```

---

## §Manifolds: `holonomy_lib.manifolds`

Riemannian manifolds with batched-first, GPU-native operations.

### `FixedRankManifold(m, n, r, device="cpu", dtype=torch.float64)`
Fixed-rank matrix manifold M_r(m, n) ⊂ R^{m×n}. Points stored as SVD triples
(U, S, Vt) of shapes (B, m, r), (B, r), (B, r, n). Manifold dim `r·(m + n − r)`.

| Method | Signature | Returns |
|---|---|---|
| `random_point` | `(batch_size=1, generator=None)` | `(U, S, Vt)` |
| `dense` | `(point)` | `(B, m, n)` |
| `projection` | `(point, Z)` | `(B, m, n)` tangent |
| `inner` | `(point, U, V)` | `(B,)` |
| `norm` | `(point, V)` | `(B,)` |
| `retraction` | `(point, tangent)` | new `(U, S, Vt)` |
| `dim` | property | int |

Refs: Vandereycken (2013), Absil-Mahony-Sepulchre (2008), Mezzadri (2007).

### `SPDManifold(n, device="cpu", dtype=torch.float64)`
Symmetric positive definite matrices P(n) with the **affine-invariant** metric
`⟨U,V⟩_S = tr(S⁻¹ U S⁻¹ V)`. Points and tangents both `(B, n, n)` symmetric.
Manifold dim `n(n+1)/2`.

| Method | Signature | Returns |
|---|---|---|
| `random_point` | `(batch_size=1, generator=None)` | `(B, n, n)` SPD |
| `is_spd` | `(S)` | `(B,)` bool |
| `projection` | `(S, Z)` | symmetric `(B, n, n)` |
| `inner` | `(S, U, V)` | `(B,)` |
| `norm` | `(S, V)` | `(B,)` |
| `exp` | `(S, V)` | `(B, n, n)` exp_S(V) |
| `log` | `(S, T)` | `(B, n, n)` log_S(T) |
| `distance` | `(S, T)` | `(B,)` geodesic |
| `retraction` | `(S, V)` | = `exp(S, V)` |

Refs: Pennec-Fillard-Ayache (2006), Bhatia (2007), Sra-Hosseini (2015).

---

## §Algebra: `holonomy_lib.algebra`

Linear-algebra primitives.

### `truncated_svd(M, r, mode="exact", oversample=5, n_iter=2, generator=None)`
Batched top-r SVD of `M: (..., m, n) → (U: (..., m, r), S: (..., r), Vt: (..., r, n))`.
- `mode="exact"`: full SVD then truncate (Eckart-Young optimal).
- `mode="randomized"`: Halko-Martinsson-Tropp projection; faster when r ≪ min(m, n).
  Accuracy controlled by `oversample` + `n_iter`.

Refs: Eckart-Young (1936), Halko-Martinsson-Tropp (2011).

### `lanczos_eigsh(A, k, n_iter=None, oversample=10, generator=None)`
Top-k largest-algebraic eigenpairs of a batched symmetric `A: (B, n, n)`
via Lanczos iteration with full reorthogonalization (Paige 1972). Cost
`O(B · n_iter · n²)`, vs `O(B · n³)` for dense `torch.linalg.eigh` — the
right tool when `n_iter ≪ n` and only the extreme eigenpairs matter.
For smallest-k, call on `λ_max · I − A` with a known spectrum upper
bound and recover by subtraction.
Refs: Lanczos (1950), Paige (1972), Saad (2011) §6.5.

---

## §Tensor calculus: `holonomy_lib.tensor_calculus`

Multilinear algebra on tensors with leading batch dim.

### `mode_product(T, A, axis)`
n-mode product T ×_axis A. Contracts axis `axis` of `T: (B, n_1, ..., n_d)` with
the last axis of `A: (B, j, n_axis)`. Result has `j` at position `axis`.
Ref: Kolda-Bader (2009), §2.5.

### `mode_unfolding(T, axis)`
Matricize T along an axis: bring `axis` to position 1, flatten the rest.
Output `(B, n_axis, prod_of_other_modes)`. Ref: Kolda-Bader (2009), §2.4.

### `hosvd(T, ranks, mode="exact", generator=None) → (core, factors)`
Truncated Higher-Order SVD. For `T: (B, n_1, ..., n_d)`, returns
`core: (B, r_1, ..., r_d)` and a list `factors[k]: (B, n_k, r_k)` with
orthonormal columns. `T ≈ core ×_1 factors[0] ×_2 factors[1] × ... ×_d factors[d−1]`.
Refs: De Lathauwer-De Moor-Vandewalle (2000), Vannieuwenhoven et al. (2012).

---

## §Spectral: `holonomy_lib.spectral`

Graph Laplacians + spectral embedding. All take symmetric adjacency `A: (B, n, n)`.
Isolated nodes handled via Moore-Penrose convention (Cheng-Wu 2024).

### `laplacian.combinatorial(A)`
L = D − A. PSD. Eigenvalue 0 multiplicity = # connected components.
Ref: Chung (1997).

### `laplacian.symmetric_normalized(A)`
L_sym = I − D^{−1/2} A D^{−1/2}. Spectrum ⊂ [0, 2]. Ref: Chung (1997), von Luxburg (2007).

### `laplacian.random_walk(A)`
L_rw = I − D^{−1} A. Same eigenvalues as L_sym (similar via D^{1/2}). Ref: von Luxburg (2007).

### `laplacian.signed(A)`
L^σ = D^{|σ|} − A,  D^{|σ|}_{ii} = Σ_j |A_{ij}|. PSD even with negative weights.
Eigenvalue 0 iff signed graph is balanced. Ref: Kunegis et al. (2010), Thm 3.4.

### `laplacian.degree(A, signed=False)`
Weighted degree `(B, n)`. With `signed=True`, uses |A| (Kunegis convention).

### `laplacian_eigenmaps(A, k, laplacian_type="symmetric_normalized") → (eigvals, eigvecs)`
Bottom-k spectral embedding. `laplacian_type ∈ {"combinatorial", "symmetric_normalized",
"random_walk", "signed"}`. Does **not** auto-drop the trivial null eigenvector
(caller decides). Refs: Belkin-Niyogi (2003), von Luxburg (2007).

### `magnetic.combinatorial(A, q=0.25)`
Hermitian magnetic Laplacian for directed graphs:
`L^(q) = D_s − H ⊙ A_s` where `H_{ij} = exp(i·2π·q·(A_{ij} − A_{ji}))`.
Returns a complex Hermitian `(B, n, n)`; real spectrum via `linalg.eigh`.
At `q = 0` collapses to the real Laplacian of `A_s = (A + A^T)/2`.
Refs: Lieb-Loss (1993), Fanuel et al. (2017), Furutani et al. (2020).

### `magnetic.symmetric_normalized(A, q=0.25)`
Symmetric-normalized magnetic Laplacian: `L_sym^(q) = I − D_s^{−1/2}(H⊙A_s)D_s^{−1/2}`.
Spectrum ⊂ [0, 2] regardless of `q`. Use the bottom-k eigenvectors as
directed-graph eigenmaps. Refs: Furutani et al. (2020), Prop. 1.

### `heat_kernel_chebyshev(L, t, signal=None, K=30, lambda_max=2.0)`
Heat kernel `exp(−t·L)` (or `exp(−t·L) @ signal`) via Chebyshev-polynomial
expansion: `O(K · n³)` dense, or `O(K · n² · k)` for an `(n, k)` signal,
beating the `O(n³)` eigendecomposition for medium `t`. Coefficients are
modified Bessel functions `I_k(t·λ_max/2)`, computed via `scipy.special.ive`.
Refs: Hammond-Vandergheynst-Gribonval (2011), §3.

### `effective_resistance(A) → (B, n, n)`
Pairwise effective resistance `R(u, v) = (e_u − e_v)ᵀ L⁺ (e_u − e_v)`
on a weighted graph (Klein-Randić 1993). On `K_n` every edge has
`R = 2/n`; on a path `P_n` the endpoints have `R = n − 1` (series
resistance). Refs: Doyle-Snell (1984), Klein-Randić (1993).

### `commute_time(A) → (B, n, n)`
Pairwise commute time `C(u, v) = vol(A) · R(u, v)` — expected
round-trip steps of the random walk on `A`. Chandra-Raghavan-Ruzzo-
Smolensky-Tiwari (1996) identity. Ref: Lovász (1993), §5.

### `diffusion_map(A, k, t=1.0) → (transition_eigvals, embedding)`
Coifman-Lafon (2006) diffusion-map embedding at time `t`. Returns
`(B, k)` transition-matrix eigenvalues `μ_j = 1 − λ_j` and
`(B, n, k)` coordinates `Ψ_t(x_i) = (μ_j^t · φ_j(x_i))`. Drops the
trivial null eigenvector. Pairwise Euclidean distance in the
embedding is the diffusion distance. Ref: Coifman-Lafon (2006), §3.

---

## §Discrete geometry: `holonomy_lib.discrete_geometry`

Combinatorial / Ricci-style curvature on graphs. The Perelman-on-networks
thread (Ollivier curvature → flow → surgery for community detection).

### `ollivier_ricci_curvature(A, alpha=0.0, reg=0.01, n_iter=100) → (B, n, n)`
Ollivier-Ricci curvature κ on all pairs via Sinkhorn Wasserstein-1 with the
shortest-path metric. `alpha` is laziness (0 = standard Ollivier; α → 1 → LLY).
For unweighted K_n: κ(edge) = (n − 2)/(n − 1).
Refs: Ollivier (2009), Liu-Lin-Yau (2011), Cuturi (2013).

### `discrete_ricci_flow(A, n_steps, dt=1.0, alpha=0.0, normalize=True, ...) → (B, n, n)`
Iterate `w_ij(t+1) = (1 − dt · κ_ij(t)) · w_ij(t)`. Negative-curvature edges
elongate (forming necks); positive-curvature edges shrink. Optional Frobenius-norm
normalization. Refs: Sia-Jonckheere-Bogdan (2019), Ni-Lin-Luo-Gao (2019).

### `ricci_flow_with_surgery(A, n_steps, surgery_period=10, surgery_threshold=3.0, ...) → (B, n, n)`
Discrete Ricci flow with periodic edge removal: the Perelman-spirit primitive.
Every `surgery_period` steps, edges whose weight ≥ `surgery_threshold` × initial
mean weight are removed. After enough iterations, the graph splits into communities.
Inspiration: Perelman (2002, 2003 surgery, 2003 extinction). Discrete: Sia (2019),
Ni-Lin-Luo-Gao (2019), Liu-Wang-Yau-Zeng (2017).

### `forman_ricci_simple(A) → (B, n, n)`
Combinatorial Forman-Ricci curvature on every edge: no optimal-transport
solve needed, `O(B · n²)` cost. For unweighted simple graphs the formula
collapses to `κ_F(u, v) = 4 − deg(u) − deg(v)`. Cheap qualitative substitute
for Ollivier-Ricci on large graphs (Sreejith et al. 2016, eq. 1).

### `forman_ricci_augmented(A) → (B, n, n)`
Augmented form adding the 2-face (triangle) contribution:
`κ_F^aug(u, v) = κ_F_simple(u, v) + 3 · #triangles(u, v)`. Tracks
Ollivier-Ricci more closely on dense substructures while remaining
fully combinatorial. Ref: Samal et al. (2018), §"Augmented Forman".

---

## §Provenance: `holonomy_lib.provenance`

Content-addressable hex provenance for **mechanistic interpretability**. Every
decorated primitive emits a Merkle-DAG node when called inside `record()`.
Same op + same inputs ⇒ same hex (deterministic). The layer TransformerLens /
nnsight / SAELens ride on top of for geometric/spectral mech interp.

### Context managers

```python
with provenance.record(cache_tensors=False, hash_algorithm=...) as reg:
    out = pipeline(...)
    # reg is a ProvenanceRegistry
```

### `ProvenanceRegistry`: what you get inside `record()`

| Method | What it does |
|---|---|
| `reg[hex]` | Look up a ProvenanceNode by hex |
| `reg.where(op_id=..., op_version=...)` | Filter nodes by op |
| `reg.ancestors(hex)` | Walk DAG upstream |
| `reg.parents(hex)` | Direct parents |
| `reg.get_tensor(hex)` | Cached output (if `cache_tensors=True`) |
| `reg.on_op(op_id, callback)` | TransformerLens-style observation hook |
| `reg.substitute({hex: value})` | Context mgr: activation patching at call time |
| `reg.replay({hex: value})` | Re-execute downstream DAG with substitution |
| `reg.to_networkx()` | Export DAG to networkx DiGraph |
| `reg.to_dataframe()` | Export node table to pandas |
| `reg.to_dict()` | JSON-friendly export |
| `reg.to_sae_dataset(op_id=None)` | Yield `(tensor, metadata)` for SAE training |
| `reg.diff(other)` | Structural diff of two recordings |
| `reg.save(path)` / `Registry.load(path)` | JSON persistence (no tensors) |

### Decorating new primitives

```python
from holonomy_lib.provenance import with_provenance

@with_provenance("holonomy_lib.module.op_name", op_version="0.1")
def op_name(x: torch.Tensor, k: int = 3) -> torch.Tensor:
    ...
```

Hex computation: `sha256(op_id || op_version || canonical(params) || ":".join(input_hexes))`
truncated to 16 chars. Pluggable hash function (blake3 if installed, else sha256).

---

## §Information geometry: `holonomy_lib.info_geometry`

Divergences on probability distributions, treated as points on a
Riemannian manifold (Amari 2016).

### `bregman_divergence(p, q, potential)`
General Bregman divergence `D_F(p ‖ q) = F(p) − F(q) − ⟨∇F(q), p − q⟩`
for any caller-supplied convex potential `F` (which must return the
pair `(F(x), ∇F(x))` at any input). Recovers squared-Euclidean,
generalized KL, and Itakura-Saito as special cases.
Refs: Bregman (1967), Banerjee et al. (2005).

### `kl_divergence_categorical(p, q) → (B,)`
Discrete KL `KL(p ‖ q) = Σ_i p_i (log p_i − log q_i)` with the
0·log(0/x) = 0 convention. Both `p` and `q` must be on the simplex.
Refs: Cover-Thomas (2006), Amari (2016) §2.4.

### `kl_divergence_gaussian(mu_p, Sigma_p, mu_q, Sigma_q) → (B,)`
Closed-form KL between two multivariate Gaussians. Cholesky-stable:
factors `Σ_q` once and reuses the factorization for the trace and
Mahalanobis terms; pulls `log det Σ` directly from the Cholesky
diagonal. Ref: Petersen-Pedersen Matrix Cookbook eq. 380.

---

## §Topology: `holonomy_lib.topology`

Hodge Laplacians and persistent homology on simplicial complexes.
Built on `holonomy_lib.simplicial`.

### `hodge_laplacian(complex, k) → Tensor`
The k-th Hodge Laplacian `L_k = ∂_{k+1} ∂_{k+1}^T + ∂_k^T ∂_k`. Its
kernel has dimension equal to the k-th Betti number (Hodge
decomposition). Dense complex → `(B, n_k_max, n_k_max)`; sparse →
`(n_k, n_k)` dense Tensor (the product of two sparse boundary
matrices is generally dense).
Refs: Eckmann (1944), Lim (2020), Schaub et al. (2020).

### `betti_numbers(complex, max_dim, threshold=1e-9) → Tensor`
`(β_0, …, β_max_dim)` via near-zero eigenvalue counting on each
`L_k`. Closed-form verification: S¹ gives `(1, 1)`, S² gives
`(1, 0, 1)`, T² gives `(1, 2, 1)`.

### `persistence_diagrams(points_or_distances, max_dim=2, max_radius=inf) → (diagrams, masks)`
Persistent homology of the Vietoris-Rips filtration over `(B, n, d)`
point clouds (or `(B, n, n)` distance matrices). Returns per-dim
batched padded tensors `diagrams[k]: (B, max_pairs_k, 2)` of
`(birth, death)` pairs + matching validity masks. H₀ via batched
union-find on sorted filtration edges; H_{1..max_dim} via Z/2
left-to-right boundary-matrix reduction (Edelsbrunner-Letscher-
Zomorodian 2002) with the Bauer-Kerber-Reininghaus clearing
optimization. Refs: Edelsbrunner-Letscher-Zomorodian (2002);
Cohen-Steiner-Edelsbrunner-Harer (2007) stability; Bauer (2021).

---

## §Simplicial: `holonomy_lib.simplicial`

Simplicial complex data structures + boundary operators. Foundation
for `holonomy_lib.topology` (Hodge Laplacians + persistent homology).
Two representations:

### `DenseSimplicialComplex(simplices_by_dim, valid_mask, n_vertices, ...)`
Batched, padded simplex tables. `simplices_by_dim[k]` is
`(B, n_k_max, k+1)` int with a `(B, n_k_max)` validity mask.
`boundary(k) → (B, n_{k-1}_max, n_k_max)` dense Tensor with the
Koszul signs.

### `SparseSimplicialComplex(simplices_by_dim, n_vertices, ...)`
Single-instance, no batch dim. `boundary(k) → sparse-CSC`. Used by
persistent homology where the matrix-reduction kernel walks the
sparse boundary column by column.

### `vietoris_rips_sparse(distances, max_radius, max_dim) → SparseSimplicialComplex`
VR complex from a single `(n, n)` distance matrix. Incremental
k-simplex construction via shared-(k-1)-face extension.

### `vietoris_rips_dense(distances, max_radius, max_dim, dtype=…) → DenseSimplicialComplex`
Batched VR construction from `(B, n, n)` distance matrices. Pads
per-dim to the max simplex count across the batch.

### `pairwise_distances(points) → Tensor`
Euclidean distance matrix from `(n, d)` or `(B, n, d)` points.
Refs: Munkres (1984), §1; Hausmann (1995); Bauer (2021).

---

## §Optimization: `holonomy_lib.optimization`

Riemannian optimizers wrapping the existing manifold `projection` +
`retraction` API. The flow is: caller computes the ambient gradient,
the optimizer projects it to the tangent space, scales by `-lr`, and
retracts back onto the manifold.

### `RiemannianSGD(manifold, lr=1e-2)`
Stateful Riemannian steepest-descent wrapper. `opt.step(point,
ambient_grad)` returns the new point. Works with `FixedRankManifold`
(point = `(U, S, Vt)` triple; ambient grad is `(B, m, n)`) and
`SPDManifold` (point and ambient grad both `(B, n, n)`).
Refs: Absil-Mahony-Sepulchre (2008), §4.1; Bonnabel (2013).

### `riemannian_sgd_step(manifold, point, ambient_grad, lr)`
Functional one-step API for use in custom training loops; the
`RiemannianSGD` class is a thin wrapper around it.

No `RiemannianAdam` in v1: adaptive step-size schemes (Adam, RMSProp,
AdamW, ...) are user-side ergonomics, not part of the math of
optimization on a manifold. The Riemannian gradient step *is* the
SGD primitive; adaptive preconditioning happens in user code by
rescaling `ambient_grad` before calling `step()`.

---

## Planned primitives

Open frontiers the library does not yet cover:
- Persistent homology on GPU (gudhi / ripser are CPU).
- Lanczos sparse-eigensolver for large graphs (currently `linalg.eigh`
  dense only).
- Hodge Laplacians on simplicial complexes.
- Sign-magnetic Laplacian for signed-directed graphs (Fiorini 2023; He
  et al. 2023). The base magnetic Laplacian is in `spectral.magnetic`;
  the signed extension is the next step.
- Riemannian optimizers (SGD/Adam/trust-region on the manifold module).
- Conjugate priors / Bregman divergences / information geometry.
- Effective resistance / commute-time distances.
- Diffusion maps built on the Chebyshev heat kernel.
- Mech-interp class-method provenance (FixedRankManifold/SPDManifold
  method calls aren't yet captured by `record()`; top-level functions only).
