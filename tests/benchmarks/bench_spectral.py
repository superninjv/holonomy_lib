"""Benchmarks for holonomy_lib.spectral.

Laplacian construction is O(n²); eigh-based ops (`laplacian_eigenmaps`,
`effective_resistance`, `diffusion_map`) are O(n³). The Chebyshev
heat kernel is O(K · n²) per matmul (dense form). The magnetic
Laplacian adds the cost of a single complex outer product on top of
the real construction. The benchmarks here cover all those regimes
so optimization claims become measurable.
"""

from __future__ import annotations

import torch

from holonomy_lib.spectral import (
    diffusion_map, effective_resistance, heat_kernel_chebyshev,
    laplacian as _L, laplacian_eigenmaps, magnetic,
)
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


# ----------------- magnetic Laplacian (directed graphs) -----------------

def _make_directed_adj(size, device, dtype):
    """Asymmetric weighted adjacency for the magnetic-Laplacian
    benchmarks; the asymmetry is what the phase factor actually
    interacts with."""
    g = torch.Generator(device="cpu"); g.manual_seed(0)
    A = torch.rand(size["B"], size["n"], size["n"], generator=g, dtype=dtype)
    A.diagonal(dim1=-2, dim2=-1).zero_()
    return A.to(device)


def _setup_magnetic_combinatorial(size, device, dtype):
    A = _make_directed_adj(size, device, dtype)
    q = size.get("q", 0.25)
    def fn():
        return magnetic.combinatorial(A, q=q)
    return fn


def _setup_magnetic_sym_norm(size, device, dtype):
    A = _make_directed_adj(size, device, dtype)
    q = size.get("q", 0.25)
    def fn():
        return magnetic.symmetric_normalized(A, q=q)
    return fn


_magnetic_sizes = [
    {"B": 1,  "n": 16,  "q": 0.25},
    {"B": 1,  "n": 64,  "q": 0.25},
    {"B": 1,  "n": 256, "q": 0.25},
    {"B": 16, "n": 64,  "q": 0.25},
    # q = 0 hits the real-only short-circuit
    {"B": 1,  "n": 256, "q": 0.0},
]

bench.case("magnetic.combinatorial", _setup_magnetic_combinatorial,
            _magnetic_sizes,
            notes="Hermitian magnetic Laplacian for directed graphs; complex output.")
bench.case("magnetic.symmetric_normalized", _setup_magnetic_sym_norm,
            _magnetic_sizes,
            notes="Normalized magnetic Laplacian; spectrum in [0, 2].")


# ----------------- heat kernel via Chebyshev -----------------

def _setup_heat_kernel_dense(size, device, dtype):
    A = _make_adj(size, device, dtype)
    L = _L.symmetric_normalized(A)
    t = size.get("t", 1.0)
    K = size.get("K", 30)
    def fn():
        return heat_kernel_chebyshev(L, t=t, K=K)
    return fn


def _setup_heat_kernel_signal(size, device, dtype):
    A = _make_adj(size, device, dtype)
    L = _L.symmetric_normalized(A)
    g = torch.Generator(device="cpu"); g.manual_seed(1)
    signal = torch.randn(
        size["B"], size["n"], size["k_signal"], generator=g, dtype=dtype,
    ).to(device)
    t = size.get("t", 1.0)
    K = size.get("K", 30)
    def fn():
        return heat_kernel_chebyshev(L, t=t, K=K, signal=signal)
    return fn


_heat_dense_sizes = [
    {"B": 1, "n": 32,  "t": 1.0, "K": 30},
    {"B": 1, "n": 128, "t": 1.0, "K": 30},
    {"B": 1, "n": 512, "t": 1.0, "K": 30},
    # Compare against ground truth via larger K
    {"B": 1, "n": 128, "t": 1.0, "K": 60},
]
_heat_signal_sizes = [
    {"B": 1, "n": 128, "k_signal": 4,  "t": 1.0, "K": 30},
    {"B": 1, "n": 512, "k_signal": 4,  "t": 1.0, "K": 30},
    {"B": 1, "n": 512, "k_signal": 64, "t": 1.0, "K": 30},
]

bench.case("heat_kernel_chebyshev_dense", _setup_heat_kernel_dense,
            _heat_dense_sizes,
            notes="exp(-t·L) as a dense (B, n, n) tensor via Chebyshev.")
bench.case("heat_kernel_chebyshev_signal", _setup_heat_kernel_signal,
            _heat_signal_sizes,
            notes="exp(-t·L) @ signal via the same Chebyshev recurrence; "
                  "should beat the dense path when k_signal << n.")


# ----------------- effective resistance -----------------

def _setup_effective_resistance(size, device, dtype):
    A = _make_adj(size, device, dtype)
    def fn():
        return effective_resistance(A)
    return fn


_resistance_sizes = [
    {"B": 1, "n": 32},
    {"B": 1, "n": 128},
    {"B": 1, "n": 512},
    {"B": 8, "n": 64},
]

bench.case("effective_resistance", _setup_effective_resistance,
            _resistance_sizes,
            notes="Pairwise R via Moore-Penrose pseudoinverse; dense eigh + outer product.")


# ----------------- diffusion map -----------------

def _setup_diffusion_map(size, device, dtype):
    A = _make_adj(size, device, dtype)
    k = size["k"]
    t = size.get("t", 1.0)
    def fn():
        return diffusion_map(A, k=k, t=t)
    return fn


_diffusion_sizes = [
    {"B": 1, "n": 64,   "k": 8,  "t": 1.0},
    {"B": 1, "n": 256,  "k": 16, "t": 1.0},
    {"B": 1, "n": 1024, "k": 32, "t": 1.0},
]

bench.case("diffusion_map", _setup_diffusion_map, _diffusion_sizes,
            notes="Coifman-Lafon embedding; dominated by the eigh in laplacian_eigenmaps.")
