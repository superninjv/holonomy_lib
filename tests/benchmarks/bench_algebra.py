"""Benchmarks for holonomy_lib.algebra.

Cases:
  truncated_svd (exact mode)        — sizes m × n, target rank r.
  truncated_svd (randomized mode)   — sizes m × n, target rank r,
                                       oversample=5, n_iter=2.

Sizes span "small enough to be a hot inner-loop primitive" (50×50) to
"big enough that fancy algorithms matter" (1024×1024). All on a single
batch — see provenance benchmarks for batched-overhead cases.
"""

from __future__ import annotations

import torch

from holonomy_lib.algebra import truncated_svd
from tests.benchmarks.harness import Bench


bench = Bench("algebra")


def _make_M(size, device, dtype):
    """Build (B, m, n) random matrix on the requested device/dtype."""
    g = torch.Generator(device="cpu")
    g.manual_seed(0)
    # Generators are device-bound in PyTorch; for cpu-generated values we
    # then move to the target. This is fine for benchmarks because the
    # setup is not what we're timing.
    M = torch.randn(
        size["B"], size["m"], size["n"], generator=g, dtype=dtype,
    ).to(device)
    return M


def _setup_truncated_svd_exact(size, device, dtype):
    M = _make_M(size, device, dtype)
    r = size["r"]
    def fn():
        return truncated_svd(M, r=r, mode="exact")
    return fn


def _setup_truncated_svd_randomized(size, device, dtype):
    M = _make_M(size, device, dtype)
    r = size["r"]
    def fn():
        return truncated_svd(M, r=r, mode="randomized")
    return fn


_svd_sizes = [
    {"B": 1, "m": 64,   "n": 64,   "r": 8},
    {"B": 1, "m": 256,  "n": 256,  "r": 16},
    {"B": 1, "m": 1024, "n": 1024, "r": 32},
    # Asymmetric (tall) matrices — common in low-rank approximation.
    {"B": 1, "m": 2048, "n": 64,   "r": 8},
    # Batched — exposes whether the kernel handles batching well.
    {"B": 16, "m": 128, "n": 128, "r": 16},
]

bench.case(
    "truncated_svd_exact", _setup_truncated_svd_exact, _svd_sizes,
    notes="Full SVD then top-r slice (Eckart-Young exact).",
)
bench.case(
    "truncated_svd_randomized", _setup_truncated_svd_randomized, _svd_sizes,
    notes="Halko-Martinsson-Tropp randomized SVD with oversample=5, n_iter=2.",
)
