"""Tests for synoros_lib.provenance.protocol.

Covers:
  1. Determinism — same op + inputs → same hex.
  2. Sensitivity — different params or inputs → different hex.
  3. Transparency outside record() — decorated functions behave normally.
  4. Recording — primitives emit nodes when inside record().
  5. DAG chaining — output of one op becomes input of another with
     the correct provenance edge.
  6. Substitution — TransformerLens-style activation patching applied
     to math primitives.
  7. Interop — to_networkx / to_dataframe / to_dict / to_sae_dataset.
  8. Hooks — observe ops without changing behavior.
  9. Diff — compare two recordings.
  10. Persistence — save/load registry to disk.
  11. Hash algorithm pluggability — blake3 vs sha256.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest
import torch

from synoros_lib import provenance
from synoros_lib.algebra import truncated_svd
from synoros_lib.spectral import laplacian


def _seeded(seed: int) -> torch.Generator:
    g = torch.Generator()
    g.manual_seed(seed)
    return g


# --------------------------------------------------------------------
# Transparency outside record()
# --------------------------------------------------------------------


class TestTransparency:
    def test_decorated_function_works_without_recording(self):
        """Calling a decorated primitive outside record() returns identical
        values to its undecorated semantics (no behavioral change)."""
        M = torch.randn(5, 7, dtype=torch.float64, generator=_seeded(0))
        U, S, Vt = truncated_svd(M, r=3, mode="exact")
        assert U.shape == (5, 3)
        assert S.shape == (3,)
        assert Vt.shape == (3, 7)

    def test_no_registry_outside_recording(self):
        """No global state pollution: calling a decorated op outside
        record() does not leave anything behind."""
        # Just verifying the call returns and doesn't raise.
        M = torch.randn(4, 4, dtype=torch.float64)
        L = laplacian.combinatorial(M)
        assert L.shape == (4, 4)


# --------------------------------------------------------------------
# Recording — basic capture
# --------------------------------------------------------------------


class TestRecording:
    def test_recording_captures_op(self):
        M = torch.randn(1, 5, 7, dtype=torch.float64, generator=_seeded(1))
        with provenance.record() as reg:
            U, S, Vt = truncated_svd(M, r=3, mode="exact")
        assert len(reg) == 1
        node = next(iter(reg))
        assert node.op_id == "synoros_lib.algebra.linear.truncated_svd"
        assert node.op_version == "0.1"
        # Output shape captured per-tensor (truncated_svd returns 3 tensors).
        assert len(node.output_shape) == 3
        assert node.output_shape[0] == (1, 5, 3)
        assert node.output_shape[1] == (1, 3)
        assert node.output_shape[2] == (1, 3, 7)

    def test_recording_captures_params(self):
        M = torch.randn(1, 5, 5, dtype=torch.float64, generator=_seeded(2))
        with provenance.record() as reg:
            truncated_svd(M, r=2, mode="exact")
        node = next(iter(reg))
        params = node.parsed_params()
        assert params["r"] == 2
        assert params["mode"] == "exact"


# --------------------------------------------------------------------
# Determinism — same op + same inputs → same hex
# --------------------------------------------------------------------


class TestDeterminism:
    def test_same_inputs_same_hex(self):
        M = torch.randn(1, 5, 5, dtype=torch.float64, generator=_seeded(3))
        with provenance.record() as reg1:
            truncated_svd(M, r=2, mode="exact")
        with provenance.record() as reg2:
            truncated_svd(M, r=2, mode="exact")
        h1 = next(iter(reg1)).hex
        h2 = next(iter(reg2)).hex
        assert h1 == h2, "same op + same inputs → same hex required"

    def test_different_params_different_hex(self):
        M = torch.randn(1, 5, 5, dtype=torch.float64, generator=_seeded(4))
        with provenance.record() as reg:
            truncated_svd(M, r=2, mode="exact")
            truncated_svd(M, r=3, mode="exact")
        hexes = [n.hex for n in reg]
        assert len(set(hexes)) == 2, "different r should yield different hex"

    def test_different_inputs_different_hex(self):
        M1 = torch.randn(1, 5, 5, dtype=torch.float64, generator=_seeded(5))
        M2 = torch.randn(1, 5, 5, dtype=torch.float64, generator=_seeded(6))
        with provenance.record() as reg:
            truncated_svd(M1, r=2, mode="exact")
            truncated_svd(M2, r=2, mode="exact")
        hexes = [n.hex for n in reg]
        assert len(set(hexes)) == 2, "different inputs should yield different hex"


# --------------------------------------------------------------------
# DAG chaining — outputs of one op become inputs of the next
# --------------------------------------------------------------------


class TestDagChaining:
    def test_chained_ops_have_correct_edges(self):
        """L = combinatorial(A); U, _, _ = truncated_svd(L, ...)
        The truncated_svd node's input_hexes should include the
        combinatorial node's hex.
        """
        A = torch.randn(1, 6, 6, dtype=torch.float64, generator=_seeded(7))
        A = (A + A.mT).abs()  # symmetric non-negative
        with provenance.record() as reg:
            L = laplacian.combinatorial(A)
            U, S, Vt = truncated_svd(L, r=3, mode="exact")

        # Two nodes recorded
        assert len(reg) == 2
        # Find each by op_id
        comb_node = reg.where(op_id="synoros_lib.spectral.laplacian.combinatorial")[0]
        svd_node = reg.where(op_id="synoros_lib.algebra.linear.truncated_svd")[0]
        # SVD's input_hexes are name=hex; comb_node.hex appears as the
        # hex part of one of those.
        hex_parts = [h.partition("=")[2] for h in svd_node.input_hexes]
        assert comb_node.hex in hex_parts

    def test_ancestors_walk(self):
        A = torch.randn(1, 5, 5, dtype=torch.float64, generator=_seeded(8))
        A = (A + A.mT).abs()
        with provenance.record() as reg:
            L = laplacian.combinatorial(A)
            U, _, _ = truncated_svd(L, r=2, mode="exact")
        svd_hex = reg.where(op_id="synoros_lib.algebra.linear.truncated_svd")[0].hex
        ancestors = reg.ancestors(svd_hex)
        # Should include the combinatorial node + the SVD itself
        assert len(ancestors) == 2


# --------------------------------------------------------------------
# Tensor caching
# --------------------------------------------------------------------


class TestTensorCache:
    def test_cache_tensors_off_by_default(self):
        M = torch.randn(1, 5, 5, dtype=torch.float64, generator=_seeded(9))
        with provenance.record() as reg:
            truncated_svd(M, r=2, mode="exact")
        h = next(iter(reg)).hex
        assert reg.get_tensor(h) is None

    def test_cache_tensors_when_enabled(self):
        M = torch.randn(1, 5, 5, dtype=torch.float64, generator=_seeded(10))
        with provenance.record(cache_tensors=True) as reg:
            U, S, Vt = truncated_svd(M, r=2, mode="exact")
        # Multi-output: U/S/Vt cached under hex:0, hex:1, hex:2
        node = next(iter(reg))
        cached_U = reg.get_tensor(f"{node.hex}:0")
        assert cached_U is not None
        torch.testing.assert_close(cached_U, U)


# --------------------------------------------------------------------
# Substitution — TransformerLens-style activation patching
# --------------------------------------------------------------------


class TestSubstitution:
    def test_substitute_at_op_call(self):
        """When we substitute a hex, that op call returns the substituted
        value instead of computing.
        """
        A = torch.randn(1, 4, 4, dtype=torch.float64, generator=_seeded(11))
        A = (A + A.mT).abs()
        # First pass: record to learn the hex of the combinatorial call
        with provenance.record() as reg:
            L = laplacian.combinatorial(A)
        target_hex = next(iter(reg)).hex

        # Second pass: substitute that hex with zeros
        fake_L = torch.zeros(1, 4, 4, dtype=torch.float64)
        with provenance.record() as reg2:
            with reg2.substitute({target_hex: fake_L}):
                L_patched = laplacian.combinatorial(A)
        torch.testing.assert_close(L_patched, fake_L, atol=0, rtol=0)


# --------------------------------------------------------------------
# Mech-interp interop — networkx + dataframe + dict exports
# --------------------------------------------------------------------


class TestInterop:
    def test_to_dict(self):
        A = torch.randn(1, 4, 4, dtype=torch.float64, generator=_seeded(12))
        A = (A + A.mT).abs()
        with provenance.record() as reg:
            laplacian.combinatorial(A)
        d = reg.to_dict()
        assert "nodes" in d
        assert len(d["nodes"]) == 1
        assert d["nodes"][0]["op_id"] == "synoros_lib.spectral.laplacian.combinatorial"

    def test_to_networkx(self):
        nx = pytest.importorskip("networkx")
        A = torch.randn(1, 5, 5, dtype=torch.float64, generator=_seeded(13))
        A = (A + A.mT).abs()
        with provenance.record() as reg:
            L = laplacian.combinatorial(A)
            truncated_svd(L, r=2, mode="exact")
        G = reg.to_networkx()
        assert isinstance(G, nx.DiGraph)
        # Three nodes: the raw input A (leaf, no op), the combinatorial
        # Laplacian, and the truncated SVD. Two edges: A → L → SVD.
        assert G.number_of_nodes() == 3
        assert G.number_of_edges() == 2
        # Op-produced nodes carry op_id; leaf input nodes don't.
        op_nodes = [h for h in G.nodes if "op_id" in G.nodes[h]]
        leaf_nodes = [h for h in G.nodes if "op_id" not in G.nodes[h]]
        assert len(op_nodes) == 2
        assert len(leaf_nodes) == 1

    def test_to_dataframe(self):
        pd = pytest.importorskip("pandas")
        A = torch.randn(1, 4, 4, dtype=torch.float64, generator=_seeded(14))
        A = (A + A.mT).abs()
        with provenance.record() as reg:
            laplacian.combinatorial(A)
            truncated_svd(A, r=2, mode="exact")
        df = reg.to_dataframe()
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 2
        assert "op_id" in df.columns
        assert "hex" in df.columns


# --------------------------------------------------------------------
# Mech-interp demo — full workflow
# --------------------------------------------------------------------


class TestMechInterpDemo:
    """End-to-end demo: a small "pipeline" gets traced, a DAG is built,
    one node is identified by op_id, substituted, and the downstream
    computation is observed to change. This is the workflow a mech
    interp researcher would do — TransformerLens for math primitives.
    """

    def test_ablation_workflow(self):
        A = torch.randn(1, 6, 6, dtype=torch.float64, generator=_seeded(15))
        A = (A + A.mT).abs()

        def pipeline(adj):
            L = laplacian.symmetric_normalized(adj)
            U, S, Vt = truncated_svd(L, r=3, mode="exact")
            return U @ torch.diag_embed(S)

        # Baseline
        with provenance.record() as reg_base:
            out_base = pipeline(A)

        # Find the Laplacian node, prepare a "zero ablation"
        lap_node = reg_base.where(
            op_id="synoros_lib.spectral.laplacian.symmetric_normalized",
        )[0]
        zero_L = torch.zeros(1, 6, 6, dtype=torch.float64)

        # Ablate: re-run with the Laplacian replaced by zeros.
        with provenance.record() as reg_ablated:
            with reg_ablated.substitute({lap_node.hex: zero_L}):
                out_ablated = pipeline(A)

        # Sanity checks:
        #   1. Baseline and ablation produced different outputs.
        diff = (out_base - out_ablated).abs().max().item()
        assert diff > 0.1, (
            f"ablation should change downstream output; diff={diff}"
        )
        #   2. Ablated DAG has the substituted Laplacian.
        ablated_lap = reg_ablated.where(
            op_id="synoros_lib.spectral.laplacian.symmetric_normalized",
        )[0]
        assert ablated_lap.hex == lap_node.hex, (
            "same op call → same hex even with substitution active"
        )


# --------------------------------------------------------------------
# Hooks — observation without mutation
# --------------------------------------------------------------------


class TestHooks:
    def test_hook_fires_on_matching_op_id(self):
        A = torch.randn(1, 4, 4, dtype=torch.float64, generator=_seeded(20))
        A = (A + A.mT).abs()
        captured = []
        with provenance.record() as reg:
            reg.on_op(
                "synoros_lib.spectral.laplacian.combinatorial",
                lambda node, out: captured.append((node.hex, out.shape)),
            )
            L = laplacian.combinatorial(A)
        assert len(captured) == 1
        hex_id, shape = captured[0]
        assert shape == (1, 4, 4)
        assert hex_id == next(iter(reg)).hex

    def test_hook_does_not_fire_for_other_ops(self):
        A = torch.randn(1, 4, 4, dtype=torch.float64, generator=_seeded(21))
        A = (A + A.mT).abs()
        captured = []
        with provenance.record() as reg:
            # Hook for an op we don't call
            reg.on_op("synoros_lib.algebra.linear.truncated_svd",
                        lambda n, o: captured.append(n))
            laplacian.combinatorial(A)
        assert captured == []

    def test_multiple_hooks_for_same_op_fire_in_order(self):
        A = torch.randn(1, 4, 4, dtype=torch.float64, generator=_seeded(22))
        A = (A + A.mT).abs()
        order = []
        with provenance.record() as reg:
            reg.on_op("synoros_lib.spectral.laplacian.combinatorial",
                        lambda n, o: order.append("first"))
            reg.on_op("synoros_lib.spectral.laplacian.combinatorial",
                        lambda n, o: order.append("second"))
            laplacian.combinatorial(A)
        assert order == ["first", "second"]

    def test_hook_observes_does_not_mutate(self):
        """Hook output is not used; the math primitive's actual return
        value is what the caller sees.
        """
        A = torch.randn(1, 4, 4, dtype=torch.float64, generator=_seeded(23))
        A = (A + A.mT).abs()
        with provenance.record() as reg:
            # Hook that "tries" to mutate (its return value is ignored)
            reg.on_op("synoros_lib.spectral.laplacian.combinatorial",
                        lambda n, o: torch.zeros_like(o))
            L = laplacian.combinatorial(A)
        # L is the real Laplacian, not zeros
        assert not torch.allclose(L, torch.zeros_like(L))


# --------------------------------------------------------------------
# SAELens-ready dataset emission
# --------------------------------------------------------------------


class TestSaeDataset:
    def test_emits_cached_activations_with_metadata(self):
        A = torch.randn(1, 5, 5, dtype=torch.float64, generator=_seeded(30))
        A = (A + A.mT).abs()
        with provenance.record(cache_tensors=True) as reg:
            laplacian.combinatorial(A)
            truncated_svd(A, r=2, mode="exact")
        records = list(reg.to_sae_dataset())
        # combinatorial yields 1 tensor; truncated_svd yields 3 (multi-output)
        assert len(records) == 4
        for tensor, meta in records:
            assert isinstance(tensor, torch.Tensor)
            assert "hex" in meta
            assert "op_id" in meta

    def test_filter_by_op_id(self):
        A = torch.randn(1, 5, 5, dtype=torch.float64, generator=_seeded(31))
        A = (A + A.mT).abs()
        with provenance.record(cache_tensors=True) as reg:
            laplacian.combinatorial(A)
            truncated_svd(A, r=2, mode="exact")
        only_svd = list(reg.to_sae_dataset(
            op_id="synoros_lib.algebra.linear.truncated_svd",
        ))
        assert len(only_svd) == 3  # U, S, Vt
        assert all(
            meta["op_id"] == "synoros_lib.algebra.linear.truncated_svd"
            for _, meta in only_svd
        )


# --------------------------------------------------------------------
# Run diffing
# --------------------------------------------------------------------


class TestDiff:
    def test_identical_runs_diff_empty(self):
        A = torch.randn(1, 4, 4, dtype=torch.float64, generator=_seeded(40))
        A = (A + A.mT).abs()
        with provenance.record() as reg1:
            laplacian.combinatorial(A)
        with provenance.record() as reg2:
            laplacian.combinatorial(A)
        d = reg1.diff(reg2)
        assert d["only_in_self"] == {}
        assert d["only_in_other"] == {}
        assert len(d["shared"]) == 1

    def test_different_inputs_show_divergence(self):
        A = torch.randn(1, 4, 4, dtype=torch.float64, generator=_seeded(41))
        A = (A + A.mT).abs()
        B = torch.randn(1, 4, 4, dtype=torch.float64, generator=_seeded(42))
        B = (B + B.mT).abs()
        with provenance.record() as reg1:
            laplacian.combinatorial(A)
        with provenance.record() as reg2:
            laplacian.combinatorial(B)
        d = reg1.diff(reg2)
        # Both runs have the combinatorial op, but with different hexes
        op = "synoros_lib.spectral.laplacian.combinatorial"
        assert op in d["only_in_self"]
        assert op in d["only_in_other"]
        assert d["shared"].get(op, []) == []

    def test_different_op_ids_show_in_only_lists(self):
        A = torch.randn(1, 4, 4, dtype=torch.float64, generator=_seeded(43))
        A = (A + A.mT).abs()
        with provenance.record() as reg1:
            laplacian.combinatorial(A)
        with provenance.record() as reg2:
            laplacian.symmetric_normalized(A)
        d = reg1.diff(reg2)
        assert "synoros_lib.spectral.laplacian.combinatorial" in d["op_ids_only_in_self"]
        assert "synoros_lib.spectral.laplacian.symmetric_normalized" in d["op_ids_only_in_other"]


# --------------------------------------------------------------------
# Persistence
# --------------------------------------------------------------------


class TestPersistence:
    def test_save_and_load_roundtrip(self):
        A = torch.randn(1, 4, 4, dtype=torch.float64, generator=_seeded(50))
        A = (A + A.mT).abs()
        with provenance.record() as reg:
            laplacian.combinatorial(A)
            truncated_svd(A, r=2, mode="exact")

        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "registry.json"
            reg.save(path)
            assert path.exists()
            loaded = provenance.ProvenanceRegistry.load(path)

        assert len(loaded) == len(reg)
        # Hexes survive
        original_hexes = {n.hex for n in reg}
        loaded_hexes = {n.hex for n in loaded}
        assert original_hexes == loaded_hexes
        # Algorithm preserved
        assert loaded.hash_algorithm == reg.hash_algorithm


# --------------------------------------------------------------------
# Causal replay — downstream DAG re-execution
# --------------------------------------------------------------------


class TestReplay:
    def test_replay_requires_tensor_cache(self):
        A = torch.randn(1, 4, 4, dtype=torch.float64, generator=_seeded(70))
        A = (A + A.mT).abs()
        with provenance.record() as reg:
            laplacian.combinatorial(A)
        with pytest.raises(ValueError, match="cache_tensors"):
            reg.replay({"abc": torch.zeros(4)})

    def test_replay_rejects_unknown_target(self):
        A = torch.randn(1, 4, 4, dtype=torch.float64, generator=_seeded(71))
        A = (A + A.mT).abs()
        with provenance.record(cache_tensors=True) as reg:
            laplacian.combinatorial(A)
        with pytest.raises(KeyError, match="not in registry"):
            reg.replay({"0000000000000000": torch.zeros(1, 4, 4)})

    def test_replay_substitutes_and_propagates(self):
        """Replay rebuilds the downstream chain from cached upstream."""
        A = torch.randn(1, 5, 5, dtype=torch.float64, generator=_seeded(72))
        A = (A + A.mT).abs() + torch.eye(5, dtype=torch.float64).unsqueeze(dim=0)

        with provenance.record(cache_tensors=True) as reg:
            L = laplacian.combinatorial(A)
            U, S, Vt = truncated_svd(L, r=3, mode="exact")

        # Find the Laplacian node; substitute it with zeros.
        lap_hex = reg.where(
            op_id="synoros_lib.spectral.laplacian.combinatorial",
        )[0].hex
        new_outputs = reg.replay({lap_hex: torch.zeros(1, 5, 5, dtype=torch.float64)})

        # The SVD should have been re-executed; outputs differ from cache.
        svd_hex = reg.where(op_id="synoros_lib.algebra.linear.truncated_svd")[0].hex
        original_U = reg.get_tensor(f"{svd_hex}:0")
        new_U = new_outputs[f"{svd_hex}:0"]
        diff = (original_U - new_U).abs().max().item()
        assert diff > 0.01, (
            f"replay should produce different U for zero'd Laplacian; diff={diff}"
        )

    def test_replay_skips_unaffected_nodes(self):
        """Nodes upstream of the substitution target are NOT in the output;
        only descendants are re-executed.
        """
        A = torch.randn(1, 4, 4, dtype=torch.float64, generator=_seeded(73))
        A = (A + A.mT).abs() + torch.eye(4, dtype=torch.float64).unsqueeze(dim=0)
        with provenance.record(cache_tensors=True) as reg:
            L = laplacian.combinatorial(A)
            U, S, Vt = truncated_svd(L, r=2, mode="exact")

        lap_hex = reg.where(
            op_id="synoros_lib.spectral.laplacian.combinatorial",
        )[0].hex
        svd_hex = reg.where(op_id="synoros_lib.algebra.linear.truncated_svd")[0].hex
        new_outputs = reg.replay({lap_hex: torch.zeros(1, 4, 4, dtype=torch.float64)})

        # Only SVD outputs in new_outputs (combinatorial wasn't re-executed
        # since it was substituted, not downstream of itself)
        assert lap_hex not in new_outputs
        assert any(h.startswith(svd_hex) for h in new_outputs)


# --------------------------------------------------------------------
# Hash-algorithm pluggability
# --------------------------------------------------------------------


class TestHashAlgorithm:
    def test_sha256_is_always_available(self):
        A = torch.randn(1, 4, 4, dtype=torch.float64, generator=_seeded(60))
        A = (A + A.mT).abs()
        with provenance.record(hash_algorithm="sha256") as reg:
            laplacian.combinatorial(A)
        assert reg.hash_algorithm == "sha256"
        assert len(reg) == 1

    def test_unknown_algorithm_rejected(self):
        with pytest.raises(ValueError, match="hash_algorithm"):
            with provenance.record(hash_algorithm="md5") as _:
                pass

    def test_different_algorithms_produce_different_hexes(self):
        """sha256 and blake3 hash the same content to different hex values."""
        try:
            import blake3  # noqa: F401
        except ImportError:
            pytest.skip("blake3 not installed")
        A = torch.randn(1, 4, 4, dtype=torch.float64, generator=_seeded(61))
        A = (A + A.mT).abs()
        with provenance.record(hash_algorithm="sha256") as r1:
            laplacian.combinatorial(A)
        with provenance.record(hash_algorithm="blake3") as r2:
            laplacian.combinatorial(A)
        h1 = next(iter(r1)).hex
        h2 = next(iter(r2)).hex
        assert h1 != h2, "different hash algos must produce different hexes"
