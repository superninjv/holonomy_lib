"""synoros_lib.algebra — audit-clean wrappers around torch.linalg + extras.

Currently implemented:
  linear.truncated_svd  — batched truncated SVD with exact or randomized mode

Planned (per HANDOFF.md §3):
  Einstein notation helpers, group representations (cyclic, dihedral),
  Lie algebras (su(n), so(n)), exterior algebra (wedge products).
"""

from synoros_lib.algebra.linear import truncated_svd

__all__ = ["truncated_svd"]
