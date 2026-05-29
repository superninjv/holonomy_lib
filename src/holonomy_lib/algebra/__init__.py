# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Synoros

"""holonomy_lib.algebra: audit-clean wrappers around torch.linalg + extras.

Currently implemented:
  linear.truncated_svd:  batched truncated SVD (exact + randomized).
  lanczos_eigsh:         top-k symmetric eigenpairs via Lanczos iteration.

Planned:
  Einstein notation helpers, group representations (cyclic, dihedral),
  Lie algebras (su(n), so(n)), exterior algebra (wedge products).
  Shift-and-invert Lanczos for smallest-eigenvalue mode.
"""

from holonomy_lib.algebra.lanczos import lanczos_eigsh
from holonomy_lib.algebra.linear import truncated_svd

__all__ = ["lanczos_eigsh", "truncated_svd"]
