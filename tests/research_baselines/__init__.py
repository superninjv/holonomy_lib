# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Synoros

"""Research baselines for ablation studies.

This package contains *baseline implementations* of methods we
compare against in the strengthening / paper work. They are NOT
library primitives — production code should not depend on them.
They live under `tests/` to signal that they're for empirical
comparison only, with no API-stability guarantees.

Modules:
  - `graphmore_discrete`: a small GraphMoRE-style discrete-gating
    per-node curvature mechanism, used as the comparison baseline
    for `HeterogeneousKappaManifold`'s continuous per-point κ.
"""
