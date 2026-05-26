"""Tests for class-method provenance on FixedRankManifold + SPDManifold.

The `@with_provenance` decorator was originally for top-level functions
only. Roadmap #7 extended it to bound methods on manifold classes:

  - Manifolds expose `_provenance_signature()` returning their
    canonical config; `_canonicalize_value` uses this in place of the
    `<Object at 0x...>` default that would have leaked memory
    addresses into the hex.
  - The decorator now unpacks tuple/list-of-tensor inputs into
    per-element content hashes, so calls like `mfd.dense((U, S, Vt))`
    contribute three independent tensor inputs instead of being
    stringified.

This module verifies both extensions: deterministic hex from class
identity, sensitivity to class-config changes, DAG chaining through
manifold method calls, and that the existing transparency property
still holds.
"""

from __future__ import annotations

import pytest
import torch

from holonomy_lib import provenance
from holonomy_lib.manifolds import FixedRankManifold, SPDManifold


def _seeded(seed: int) -> torch.Generator:
    g = torch.Generator()
    g.manual_seed(seed)
    return g


# ---------------------------------------------------------------
# Transparency: methods behave identically outside record()
# ---------------------------------------------------------------


class TestTransparency:
    def test_spd_methods_unchanged_outside_recording(self):
        mfd = SPDManifold(n=4)
        S = mfd.random_point(batch_size=2, generator=_seeded(0))
        V = torch.randn(2, 4, 4, dtype=torch.float64, generator=_seeded(1))
        V_sym = mfd.projection(S, V)
        assert V_sym.shape == (2, 4, 4)
        torch.testing.assert_close(V_sym, V_sym.mT, atol=1e-12, rtol=0)
        assert torch.isfinite(mfd.norm(S, V_sym)).all()

    def test_fixed_rank_methods_unchanged_outside_recording(self):
        mfd = FixedRankManifold(m=8, n=6, r=3)
        point = mfd.random_point(batch_size=2, generator=_seeded(0))
        M = mfd.dense(point)
        assert M.shape == (2, 8, 6)


# ---------------------------------------------------------------
# Method calls inside record() emit nodes
# ---------------------------------------------------------------


class TestRecording:
    def test_spd_projection_captured(self):
        mfd = SPDManifold(n=3)
        S = mfd.random_point(batch_size=1, generator=_seeded(0))
        Z = torch.randn(1, 3, 3, dtype=torch.float64, generator=_seeded(1))
        with provenance.record() as reg:
            mfd.projection(S, Z)
        assert len(reg) == 1
        node = next(iter(reg))
        assert node.op_id == "holonomy_lib.manifolds.SPDManifold.projection"

    def test_fixed_rank_dense_captured_with_tuple_input(self):
        """FixedRankPoint = (U, S, Vt) is a tuple-of-tensors input.
        Verify it's unpacked into three named input hexes rather than
        stringified into params."""
        mfd = FixedRankManifold(m=5, n=4, r=2)
        point = mfd.random_point(batch_size=1, generator=_seeded(0))
        with provenance.record() as reg:
            mfd.dense(point)
        assert len(reg) == 1
        node = next(iter(reg))
        assert node.op_id == "holonomy_lib.manifolds.FixedRankManifold.dense"
        # Three input hexes from the unpacked tuple: point[0], point[1], point[2]
        input_names = [edge.split("=")[0] for edge in node.input_hexes]
        assert input_names == ["point[0]", "point[1]", "point[2]"], (
            f"expected tuple unpacking; got input_names={input_names}"
        )

    def test_spd_exp_captured(self):
        mfd = SPDManifold(n=3)
        S = mfd.random_point(batch_size=1, generator=_seeded(0))
        V = mfd.projection(
            S, torch.randn(1, 3, 3, dtype=torch.float64, generator=_seeded(1)),
        )
        with provenance.record() as reg:
            mfd.exp(S, V)
        node_ids = sorted(n.op_id for n in reg)
        assert "holonomy_lib.manifolds.SPDManifold.exp" in node_ids


# ---------------------------------------------------------------
# Hex determinism + sensitivity to manifold config
# ---------------------------------------------------------------


class TestDeterminismAndSensitivity:
    def test_same_manifold_config_yields_same_hex(self):
        """Two SPDManifold instances with identical (n, device, dtype)
        must produce the same hex for the same call. If `self` were
        hashed by id(), this would fail."""
        S = torch.eye(3, dtype=torch.float64).unsqueeze(0)
        Z = torch.randn(1, 3, 3, dtype=torch.float64, generator=_seeded(0))

        mfd1 = SPDManifold(n=3)
        with provenance.record() as reg1:
            mfd1.projection(S, Z)
        h1 = next(iter(reg1)).hex

        mfd2 = SPDManifold(n=3)   # fresh instance, same config
        with provenance.record() as reg2:
            mfd2.projection(S, Z)
        h2 = next(iter(reg2)).hex

        assert h1 == h2, f"expected stable hex across instances; got {h1} vs {h2}"

    def test_different_manifold_dtype_yields_different_hex(self):
        """Changing the manifold dtype must shift the hex (since dtype
        is part of `_provenance_signature`)."""
        Z32 = torch.randn(1, 3, 3, generator=_seeded(0))
        S32 = torch.eye(3).unsqueeze(0)
        Z64 = Z32.to(torch.float64)
        S64 = S32.to(torch.float64)

        mfd32 = SPDManifold(n=3, dtype=torch.float32)
        with provenance.record() as reg32:
            mfd32.projection(S32, Z32)
        h32 = next(iter(reg32)).hex

        mfd64 = SPDManifold(n=3, dtype=torch.float64)
        with provenance.record() as reg64:
            mfd64.projection(S64, Z64)
        h64 = next(iter(reg64)).hex

        assert h32 != h64, "manifold dtype must affect hex"

    def test_different_fixed_rank_r_yields_different_hex(self):
        """Manifold rank `r` is part of its provenance signature."""
        M = torch.randn(1, 5, 5, dtype=torch.float64, generator=_seeded(0))

        # truncated_svd is the building block; use it to make valid points
        from holonomy_lib.algebra import truncated_svd

        mfd2 = FixedRankManifold(m=5, n=5, r=2)
        mfd3 = FixedRankManifold(m=5, n=5, r=3)

        U2, S2, Vt2 = truncated_svd(M[0], r=2, mode="exact")
        U3, S3, Vt3 = truncated_svd(M[0], r=3, mode="exact")
        point2 = (U2.unsqueeze(0), S2.unsqueeze(0), Vt2.unsqueeze(0))
        point3 = (U3.unsqueeze(0), S3.unsqueeze(0), Vt3.unsqueeze(0))

        with provenance.record() as reg2:
            mfd2.dense(point2)
        with provenance.record() as reg3:
            mfd3.dense(point3)

        h2 = next(iter(reg2)).hex
        h3 = next(iter(reg3)).hex
        # Different config AND different tensor content; we just confirm
        # that hexes are different (sensitivity, not identity).
        assert h2 != h3


# ---------------------------------------------------------------
# DAG chaining through manifold methods
# ---------------------------------------------------------------


class TestReplayLimitations:
    """Class-method calls record into the DAG correctly, but replay()
    cannot reconstruct the manifold instance from its provenance
    signature in v0.1. Verify a clear error fires instead of crashing
    on a downstream attribute access."""

    def test_replay_class_method_raises_clear_error(self):
        """Chain two class-method calls and substitute the upstream
        node's output. Replay walks the DAG and must hit the
        class-method node downstream, triggering the clear error."""
        mfd = SPDManifold(n=3)
        S = mfd.random_point(batch_size=1, generator=_seeded(0))
        Z = torch.randn(1, 3, 3, dtype=torch.float64, generator=_seeded(1))
        with provenance.record(cache_tensors=True) as reg:
            V = mfd.projection(S, Z)
            mfd.exp(S, V)

        nodes_by_id = {n.op_id: n for n in reg}
        proj_hex = nodes_by_id[
            "holonomy_lib.manifolds.SPDManifold.projection"
        ].hex

        # Substitute the projection output → exp downstream must replay.
        V_new = mfd.projection(
            S, torch.randn(1, 3, 3, dtype=torch.float64, generator=_seeded(2)),
        )
        with pytest.raises(NotImplementedError, match="class-method"):
            reg.replay({proj_hex: V_new})

    def test_replay_tuple_input_raises_clear_error(self):
        """FixedRankPoint = (U, S, Vt) gets unpacked into per-element
        hex keys at record time. Trigger replay through a class-method
        node and verify the clear error covers it (either via the
        class-method branch or via the tuple-input branch, both
        register as NotImplementedError)."""
        mfd = FixedRankManifold(m=5, n=4, r=2)
        # Build a 2-step chain so substitute → replay walks the DAG.
        point = mfd.random_point(batch_size=1, generator=_seeded(0))
        with provenance.record(cache_tensors=True) as reg:
            M = mfd.dense(point)
            mfd.projection(point, M)

        # Substitute the dense output → projection downstream replays.
        nodes_by_id = {n.op_id: n for n in reg}
        dense_hex = nodes_by_id[
            "holonomy_lib.manifolds.FixedRankManifold.dense"
        ].hex
        M_new = mfd.dense(mfd.random_point(batch_size=1, generator=_seeded(2)))
        with pytest.raises(NotImplementedError, match="class-method|tuple"):
            reg.replay({dense_hex: M_new})


class TestNestedDecorationEmitsBothNodes:
    """`SPDManifold.norm` calls `SPDManifold.inner` internally. Both
    are decorated, so a single `norm` call inside record() must emit
    exactly two nodes (norm + inner). Document this so downstream DAG
    cost models don't get surprised."""

    def test_norm_emits_inner_and_norm(self):
        mfd = SPDManifold(n=3)
        S = mfd.random_point(batch_size=1, generator=_seeded(0))
        V = mfd.projection(
            S, torch.randn(1, 3, 3, dtype=torch.float64, generator=_seeded(1)),
        )
        with provenance.record() as reg:
            mfd.norm(S, V)
        op_ids = sorted(n.op_id for n in reg)
        assert op_ids == [
            "holonomy_lib.manifolds.SPDManifold.inner",
            "holonomy_lib.manifolds.SPDManifold.norm",
        ]


class TestDagChaining:
    def test_spd_projection_then_exp_chains(self):
        """`exp` consumes `S` and the output of `projection`. The exp
        node's input_hexes must therefore include the projection node's
        output hex (DAG edge)."""
        mfd = SPDManifold(n=3)
        S = mfd.random_point(batch_size=1, generator=_seeded(0))
        Z = torch.randn(1, 3, 3, dtype=torch.float64, generator=_seeded(1))
        with provenance.record() as reg:
            V = mfd.projection(S, Z)
            mfd.exp(S, V)

        nodes_by_id = {n.op_id: n for n in reg}
        proj_node = nodes_by_id["holonomy_lib.manifolds.SPDManifold.projection"]
        exp_node = nodes_by_id["holonomy_lib.manifolds.SPDManifold.exp"]

        exp_input_hexes = [e.split("=")[1] for e in exp_node.input_hexes]
        assert proj_node.hex in exp_input_hexes, (
            f"exp should consume projection's output; "
            f"proj.hex={proj_node.hex}, exp.inputs={exp_node.input_hexes}"
        )
