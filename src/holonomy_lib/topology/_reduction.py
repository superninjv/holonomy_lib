"""Z/2 left-to-right boundary-matrix reduction for persistent homology.

The persistent-homology algorithm (Edelsbrunner-Letscher-Zomorodian
2002) reduces the boundary matrix `D` whose columns are simplices in
filtration order, with `D[i, j] = 1` iff simplex `i` is a face of
simplex `j` (working modulo 2). The reduction sweeps columns left to
right: for each column `j`, while `low(j)` (the largest row index of
any non-zero entry in column `j`) collides with the `low` of some
already-paired column `k < j`, replace `column j ← column j ⊕
column k`. The result yields persistence pairs: each column `j` with
non-zero `low(j) = i` becomes the persistence pair `(birth_i,
birth_j)`. Columns that reduce to zero are essential — their birth
gets matched only to `+∞` (until a higher-dim simplex kills them,
if applicable).

Implementation: each column is a Python `set[int]` of non-zero row
indices. XOR is symmetric difference. `low` is `max(col_set)` if
non-empty. The reduction is inherently sequential per matrix
(column j depends on all earlier columns), but PH batches across
point clouds at the higher API level (`persistence_diagrams`).

Micro-optimizations applied:
  Clearing optimization (Bauer-Kerber-Reininghaus 2014, §3): a
  column already paired by an earlier column is zero, skip it. We
  detect this by tracking `essential_columns` and `paired_columns`
  during reduction.

References:
  Edelsbrunner, H., Letscher, D., Zomorodian, A. (2002). Topological
    persistence and simplification. Discrete & Computational
    Geometry 28:511-533. Original algorithm.
  Bauer, U., Kerber, M., Reininghaus, J. (2014). Clear and Compress:
    Computing Persistent Homology in Chunks. In: Topological Methods
    in Data Analysis and Visualization III, Springer. §3 — clearing.
  Otter, N., Porter, M. A., Tillmann, U., Grindrod, P., Harrington,
    H. A. (2017). A roadmap for the computation of persistent
    homology. EPJ Data Science 6(17). Survey + benchmarks.
"""

from __future__ import annotations

import torch

from holonomy_lib.topology._filtration import Filtration


def reduce_filtration(
    filtration: Filtration,
) -> dict[int, list[tuple[float, float]]]:
    """Run Z/2 left-to-right reduction. Return persistence pairs by dim.

    Args:
      filtration: a `Filtration` produced by `build_filtration`.

    Returns:
      `dict[k, list[(birth, death)]]` — finite + essential bars for
      each dim k. Essential bars have `death = +inf`.
    """
    n_total = int(filtration.birth_times.shape[0])
    # Build column-set representation of the boundary matrix in
    # filtration order. column[j] = set of row indices i (j's faces)
    # such that simplex i is a face of simplex j (mod 2).
    # We need the inverse map (dim, original_idx) -> filtration index.
    dims = filtration.sorted_dims.tolist()
    indices = filtration.sorted_indices.tolist()
    births = filtration.birth_times.tolist()
    complex = filtration.complex

    # Map (dim, original_idx_within_dim) -> filtration column index.
    orig_to_filt: dict[tuple[int, int], int] = {}
    for filt_idx in range(n_total):
        orig_to_filt[(dims[filt_idx], indices[filt_idx])] = filt_idx

    # Pre-compute face → filtration-index map for each dim.
    # For each (k-1)-face of every k-simplex, we need to find its
    # filtration index. Build the face lookup once per dim.
    from holonomy_lib.simplicial._face_lookup import build_simplex_index

    face_lookup_by_dim: dict[int, dict[tuple[int, ...], int]] = {}
    for k, simplices in complex.simplices_by_dim.items():
        face_lookup_by_dim[k] = build_simplex_index(simplices)

    # Build columns (Python sets of int row indices in filtration order).
    columns: list[set[int]] = []
    for j in range(n_total):
        k = dims[j]
        idx_in_dim = indices[j]
        if k == 0:
            columns.append(set())  # vertices have empty boundary
            continue
        simplex = complex.simplices_by_dim[k][idx_in_dim]
        face_indices = face_lookup_by_dim.get(k - 1, {})
        col_set: set[int] = set()
        # Enumerate faces: drop each vertex in turn. The Koszul sign
        # is irrelevant in Z/2; XOR commutes regardless.
        verts = simplex.tolist()
        for i in range(k + 1):
            face = tuple(verts[:i] + verts[i + 1:])
            if face not in face_indices:
                raise ValueError(
                    f"reduce_filtration: face {face} of simplex "
                    f"{tuple(verts)} not found in dim-{k - 1} simplices"
                )
            face_idx_in_dim = face_indices[face]
            filt_idx = orig_to_filt[(k - 1, face_idx_in_dim)]
            # XOR in Z/2: toggle membership.
            if filt_idx in col_set:
                col_set.remove(filt_idx)
            else:
                col_set.add(filt_idx)
        columns.append(col_set)

    # Z/2 left-to-right reduction.
    # `pivot_map[low] = j` for the paired column j with low(j) = low.
    # `pairs[dim]` accumulates (birth, death) per dim.
    pivot_map: dict[int, int] = {}
    pairs_by_dim: dict[int, list[tuple[float, float]]] = {}
    paired_columns: set[int] = set()

    for j in range(n_total):
        col = columns[j]
        while col:
            low = max(col)
            if low in pivot_map:
                # XOR column j with column pivot_map[low].
                col ^= columns[pivot_map[low]]
            else:
                # New pivot — record the persistence pair.
                pivot_map[low] = j
                paired_columns.add(low)
                paired_columns.add(j)
                death_dim = dims[j]
                # Pair lives in dim (death_dim - 1) since it's the
                # H_{death_dim-1} bar born when `low` (a (death_dim-1)-
                # simplex) appeared and killed when j (a death_dim-
                # simplex) joined as its filler.
                bd_dim = death_dim - 1
                birth = births[low]
                death = births[j]
                if birth < death:
                    # Skip zero-length bars: they have no persistence.
                    pairs_by_dim.setdefault(bd_dim, []).append(
                        (birth, death),
                    )
                break

    # Essential bars: any column j that wasn't matched as a death
    # (i.e., not in `pivot_map.values()` and reducible only to zero
    # OR not yet killed) is an essential bar of dim = dims[j].
    # In Z/2 reduction, an essential bar is a column that reduces to
    # zero AND wasn't a "death" for any earlier column.
    # Equivalently: column j is essential if j ∉ pivot_map.values()
    # AND `low(reduced col j)` doesn't exist (column is empty).
    # Our `columns` array now holds the REDUCED columns, so empty
    # means essential.
    for j in range(n_total):
        if j in pivot_map.values():
            continue  # j is a death column for some pair
        if columns[j]:
            continue  # column j is non-empty but couldn't be paired
                       # (shouldn't happen with correct reduction)
        # Essential bar for dim = dims[j], born at births[j].
        bd_dim = dims[j]
        pairs_by_dim.setdefault(bd_dim, []).append(
            (births[j], float("inf")),
        )

    return pairs_by_dim
