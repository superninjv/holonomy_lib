"""Benchmarks for holonomy_lib.spectral (Laplacians + eigenmaps).

Laplacian variants are O(n²) construction + (eigenmaps) O(n³) eigh.
For large n the eigh dominates everything; we benchmark both phases.
"""

from __future__ import annotations

import torch

from holonomy_lib.spectral import laplacian as _L
from holonomy_lib.spectral import laplacian_eigenmaps
from tests.benchmarks.harness import Bench


bench = Bench("spectral")


def _make_adj(size, device, dtype, signed=False):
    g = torch.Generator(device="cpu"); g.manual_seed(0)
    A = torch.rand(size["B"], size["n"], size["n"], generator=g, dtype=dtype)
    A = (A + A.mT) * 0.5
    if signed:
        # Add some negative weights for signed Laplacian benchmarks.
        mask = torch.rand_like(A) < 0.3
        A = torch.where(mask, -A, A)
    A.diagonal(dim1=-2, dim2=-1).zero_()
    return A.to(device)


def _setup_combinatorial(size, device, dtype):
    A = _make_adj(size, device, dtype)
    def fn():
        return _L.combinatorial(A)
    return fn


def _setup_sym_norm(size, device, dtype):
    A = _make_adj(size, device, dtype)
    def fn():
        return _L.symmetric_normalized(A)
    return fn


def _setup_random_walk(size, device, dtype):
    A = _make_adj(size, device, dtype)
    def fn():
        return _L.random_walk(A)
    return fn


def _setup_signed(size, device, dtype):
    A = _make_adj(size, device, dtype, signed=True)
    def fn():
        return _L.signed(A)
    return fn


def _setup_eigenmaps(size, device, dtype):
    A = _make_adj(size, device, dtype)
    k = size["k"]
    def fn():
        return laplacian_eigenmaps(A, k=k, laplacian_type="symmetric_normalized")
    return fn


_laplacian_sizes = [
    {"B": 1,  "n": 16},
    {"B": 1,  "n": 64},
    {"B": 1,  "n": 256},
    {"B": 16, "n": 64},
    {"B": 1,  "n": 1024},
]
_eigenmap_sizes = [
    {"B": 1,  "n": 64,   "k": 8},
    {"B": 1,  "n": 256,  "k": 16},
    {"B": 16, "n": 64,   "k": 8},
    {"B": 1,  "n": 1024, "k": 32},
]


bench.case("laplacian.combinatorial", _setup_combinatorial, _laplacian_sizes,
            notes="L = D - A; diag_embed + subtract.")
bench.case("laplacian.symmetric_normalized", _setup_sym_norm, _laplacian_sizes,
            notes="L_sym = I - D^{-1/2} A D^{-1/2}; two broadcasts.")
bench.case("laplacian.random_walk", _setup_random_walk, _laplacian_sizes,
            notes="L_rw = I - D^{-1} A; one broadcast.")
bench.case("laplacian.signed", _setup_signed, _laplacian_sizes,
            notes="L^σ = D^{|σ|} - A; |.|-sum + diag_embed.")
bench.case("laplacian_eigenmaps", _setup_eigenmaps, _eigenmap_sizes,
            notes="Bottom-k eigenpairs via dense eigh of L_sym.")
