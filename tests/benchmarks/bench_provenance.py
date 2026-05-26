"""Benchmarks for holonomy_lib.provenance overhead.

Per-call overhead of the @with_provenance decorator (inspect.signature,
content-hash, JSON canon) shows up on every primitive call inside a
record() context. We measure with a trivial op so the timing reflects
the decorator, not the op.
"""

from __future__ import annotations

import torch

from holonomy_lib import provenance
from holonomy_lib.provenance import with_provenance
from holonomy_lib.spectral import laplacian
from tests.benchmarks.harness import Bench


bench = Bench("provenance")


# Trivial op so the timing reflects the decorator, not the math.
@with_provenance("bench.identity", op_version="0.1")
def _identity(x: torch.Tensor) -> torch.Tensor:
    return x


def _setup_decorator_outside_record(size, device, dtype):
    g = torch.Generator(device="cpu"); g.manual_seed(0)
    x = torch.randn(size["B"], size["n"], generator=g, dtype=dtype).to(device)
    def fn():
        return _identity(x)
    return fn


def _setup_decorator_inside_record(size, device, dtype):
    g = torch.Generator(device="cpu"); g.manual_seed(0)
    x = torch.randn(size["B"], size["n"], generator=g, dtype=dtype).to(device)
    def fn():
        with provenance.record() as _:
            _identity(x)
    return fn


def _setup_chained_inside_record(size, device, dtype):
    """combinatorial → identity chain — exercises tensor chaining."""
    n = size["n"]
    g = torch.Generator(device="cpu"); g.manual_seed(0)
    A = torch.rand(size["B"], n, n, generator=g, dtype=dtype)
    A = (A + A.mT) * 0.5
    A = A.to(device)
    def fn():
        with provenance.record() as _:
            L = laplacian.combinatorial(A)
            _identity(L)
    return fn


_sizes = [
    {"B": 1,  "n": 16},
    {"B": 1,  "n": 128},
    {"B": 16, "n": 16},
]


bench.case("decorator_outside_record", _setup_decorator_outside_record, _sizes,
            notes="Decorator no-op path — should be one attribute lookup.")
bench.case("decorator_inside_record", _setup_decorator_inside_record, _sizes,
            notes="Full provenance overhead per call: bind args, hash, register.")
bench.case("chained_inside_record", _setup_chained_inside_record, _sizes,
            notes="Two-op chain — exercises tensor-id → hex lookup.")
