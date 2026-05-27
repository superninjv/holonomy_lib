"""Tests for the agent-tool decorator + schema generators.

These cover the foundation layer (Phase 1). The actual inspection
tools (Phase 2+) have their own test classes.
"""

from __future__ import annotations

from typing import Optional

import pytest

from holonomy_lib.provenance import ProvenanceRegistry
from holonomy_lib.provenance import agent


def _wipe_registry():
    """Clear _AGENT_TOOLS so tests can register fresh tools without
    cross-contamination from the production tools imported elsewhere.
    """
    agent._AGENT_TOOLS.clear()


class TestAgentToolDecorator:
    def setup_method(self) -> None:
        _wipe_registry()

    def test_decoration_registers_tool(self) -> None:
        @agent.agent_tool(description="Echo the input.")
        def echo(x: str) -> str:
            """[unused docstring]"""
            return x
        names = [t.name for t in agent.list_tools()]
        assert "echo" in names

    def test_default_description_pulls_from_docstring_first_line(self) -> None:
        @agent.agent_tool()
        def add(a: int, b: int) -> int:
            """Add two integers.

            Detailed description that the LLM should not see.
            """
            return a + b
        spec = agent.get_tool("add")
        assert spec.description == "Add two integers."

    def test_explicit_name_overrides_function_name(self) -> None:
        @agent.agent_tool(description="Identity.", name="passthrough")
        def some_internal_name(x: str) -> str:
            return x
        names = [t.name for t in agent.list_tools()]
        assert "passthrough" in names
        assert "some_internal_name" not in names

    def test_unknown_tool_raises(self) -> None:
        with pytest.raises(KeyError):
            agent.get_tool("nonexistent")


class TestSchemaGeneration:
    def setup_method(self) -> None:
        _wipe_registry()

    def test_anthropic_schema_shape(self) -> None:
        @agent.agent_tool(description="Slice a cached tensor.")
        def tensor_slice(hex: str, expr: str = ":") -> dict:
            return {}
        schemas = agent.to_anthropic_schema()
        assert len(schemas) == 1
        s = schemas[0]
        assert s["name"] == "tensor_slice"
        assert s["description"] == "Slice a cached tensor."
        # Anthropic uses `input_schema`.
        assert "input_schema" in s
        params = s["input_schema"]
        assert params["type"] == "object"
        assert "hex" in params["properties"]
        assert params["properties"]["hex"] == {"type": "string"}
        # `expr` has a default, so it's optional (not in `required`).
        assert params["required"] == ["hex"]

    def test_openai_schema_shape(self) -> None:
        @agent.agent_tool(description="Slice a cached tensor.")
        def tensor_slice(hex: str, expr: str = ":") -> dict:
            return {}
        schemas = agent.to_openai_schema()
        assert len(schemas) == 1
        s = schemas[0]
        assert s["type"] == "function"
        # OpenAI nests under `function`.
        assert s["function"]["name"] == "tensor_slice"
        assert s["function"]["description"] == "Slice a cached tensor."
        assert "parameters" in s["function"]

    def test_registry_param_is_stripped_from_schema(self) -> None:
        """The transport adapters pre-bind the registry, so the
        LLM-facing schema must not include it. The agent should never
        be asked to provide a ProvenanceRegistry value.
        """
        @agent.agent_tool(description="Tool that needs the registry.")
        def some_tool(registry: ProvenanceRegistry, hex: str) -> dict:
            return {}
        schemas = agent.to_anthropic_schema()
        params = schemas[0]["input_schema"]["properties"]
        assert "registry" not in params
        assert "hex" in params

    def test_type_mapping_covers_common_python_types(self) -> None:
        @agent.agent_tool(description="Type-mapping smoke test.")
        def mixed(
            a_string: str,
            an_int: int,
            a_float: float,
            a_bool: bool,
            a_list: list[int],
            an_optional: Optional[str] = None,
        ) -> dict:
            return {}
        s = agent.to_anthropic_schema()[0]["input_schema"]
        p = s["properties"]
        assert p["a_string"] == {"type": "string"}
        assert p["an_int"] == {"type": "integer"}
        assert p["a_float"] == {"type": "number"}
        assert p["a_bool"] == {"type": "boolean"}
        assert p["a_list"] == {"type": "array", "items": {"type": "integer"}}
        # Optional[str] widens to string (single-arg union after None drop).
        assert p["an_optional"] == {"type": "string"}
        # Only the parameters without defaults are required.
        assert set(s["required"]) == {
            "a_string", "an_int", "a_float", "a_bool", "a_list",
        }


# --------------------------------------------------------------------
# Inspection tools (Phase 2)
# --------------------------------------------------------------------


import torch

from holonomy_lib.algebra import truncated_svd
from holonomy_lib.spectral import laplacian


def _seeded(seed: int) -> torch.Generator:
    g = torch.Generator()
    g.manual_seed(seed)
    return g


def _build_registry() -> tuple[ProvenanceRegistry, dict[str, str]]:
    """Build a small recording for the inspection-tool tests. Returns
    (registry, hex_by_op_id)."""
    A = torch.randn(4, 5, 5, dtype=torch.float64, generator=_seeded(0))
    A = (A + A.mT).abs() + torch.eye(5, dtype=torch.float64).unsqueeze(0)
    from holonomy_lib import provenance
    with provenance.record(cache_tensors=True) as reg:
        L = laplacian.combinatorial(A)
        truncated_svd(L, r=3, mode="exact")
    by_op = {n.op_id: n.hex for n in reg}
    return reg, by_op


class TestIndexExprParser:
    def test_integer(self) -> None:
        assert agent._parse_index_expr("0") == 0
        assert agent._parse_index_expr("-1") == -1

    def test_full_slice(self) -> None:
        # `:` parses as slice(None, None, None).
        result = agent._parse_index_expr(":")
        assert isinstance(result, slice)
        assert result == slice(None, None, None)

    def test_partial_slice(self) -> None:
        assert agent._parse_index_expr("2:5") == slice(2, 5, None)
        assert agent._parse_index_expr("::2") == slice(None, None, 2)

    def test_multi_dim(self) -> None:
        result = agent._parse_index_expr(":, 0")
        assert result == (slice(None, None, None), 0)

    def test_rejects_disallowed_chars(self) -> None:
        with pytest.raises(ValueError, match="disallowed"):
            agent._parse_index_expr("0; print('hi')")
        with pytest.raises(ValueError, match="disallowed"):
            agent._parse_index_expr("__import__")


class TestTensorSlice:
    def test_returns_inline_values_for_small_slice(self) -> None:
        reg, by_op = _build_registry()
        lap_hex = by_op["holonomy_lib.spectral.laplacian.combinatorial"]
        # 1D slice = 5 elements, well below the inline limit.
        out = agent.tensor_slice(reg, lap_hex, "0, 0, :")
        assert "values" in out
        assert len(out["values"]) == 5

    def test_returns_truncated_for_large_slice(self) -> None:
        reg, by_op = _build_registry()
        lap_hex = by_op["holonomy_lib.spectral.laplacian.combinatorial"]
        # 4*5*5 = 100 elements; still below 256, so this should be inline.
        # Build a bigger registry for the truncation test.
        big = torch.zeros(1, 32, 32, dtype=torch.float64)
        from holonomy_lib import provenance
        with provenance.record(cache_tensors=True) as reg2:
            laplacian.combinatorial(big.abs() + torch.eye(32).unsqueeze(0))
        hex_id = next(iter(reg2)).hex
        out = agent.tensor_slice(reg2, hex_id, ":")
        # 1*32*32 = 1024 elements > 256, should truncate.
        assert out.get("truncated") is True
        assert "stats" in out
        assert "values" not in out

    def test_missing_tensor_returns_error(self) -> None:
        reg, _ = _build_registry()
        out = agent.tensor_slice(reg, "0000000000000000", ":")
        assert "error" in out

    def test_bad_index_returns_error(self) -> None:
        reg, by_op = _build_registry()
        lap_hex = by_op["holonomy_lib.spectral.laplacian.combinatorial"]
        out = agent.tensor_slice(reg, lap_hex, "0; rm -rf /")
        assert "error" in out
        assert "disallowed" in out["error"]


class TestTensorPerBatchSummary:
    def test_returns_one_row_per_batch_element(self) -> None:
        """The headline fix: per-batch stats, not lumped averages."""
        reg, by_op = _build_registry()
        lap_hex = by_op["holonomy_lib.spectral.laplacian.combinatorial"]
        out = agent.tensor_per_batch_summary(reg, lap_hex)
        assert "per_batch" in out
        assert len(out["per_batch"]) == 4  # batch dim
        # Each row has the four stats and an index.
        for i, row in enumerate(out["per_batch"]):
            assert row["index"] == i
            assert {"mean", "std", "min", "max"} <= row.keys()

    def test_validates_anomaly_visibility(self) -> None:
        """Anomalies that v0.3.0's get_tensor_summary lumped together
        are visible per-batch. Inject a clear outlier and confirm the
        per-batch max picks it out.
        """
        torch.manual_seed(0)
        A = torch.zeros(3, 4, 4, dtype=torch.float64)
        # Batches 0 and 1 are "normal"; batch 2 has a 100x edge.
        A[0] = torch.eye(4, dtype=torch.float64)
        A[1] = torch.eye(4, dtype=torch.float64)
        A[2] = torch.eye(4, dtype=torch.float64)
        A[2, 0, 1] = 100.0
        A[2, 1, 0] = 100.0
        from holonomy_lib import provenance
        with provenance.record(cache_tensors=True) as reg:
            laplacian.combinatorial(A)
        hex_id = next(iter(reg)).hex
        out = agent.tensor_per_batch_summary(reg, hex_id)
        per_batch_maxes = [row["max"] for row in out["per_batch"]]
        # Batch 2 should have the largest max by a wide margin.
        anomaly_idx = max(range(3), key=lambda i: per_batch_maxes[i])
        assert anomaly_idx == 2
        assert per_batch_maxes[2] > per_batch_maxes[0] * 10


class TestTensorEigenvalues:
    def test_top_k_eigvals_for_identity_plus_perturbation(self) -> None:
        reg, by_op = _build_registry()
        lap_hex = by_op["holonomy_lib.spectral.laplacian.combinatorial"]
        out = agent.tensor_eigenvalues(reg, lap_hex, k=2)
        assert out["k"] == 2
        # Batched (4 batches) → values is list-of-lists of length 2 each.
        assert len(out["values"]) == 4
        for batch_vals in out["values"]:
            assert len(batch_vals) == 2
            # Eigenvalues of a Laplacian are non-negative; descending order.
            assert batch_vals[0] >= batch_vals[1] >= -1e-9

    def test_non_square_returns_error(self) -> None:
        reg, by_op = _build_registry()
        svd_hex = by_op["holonomy_lib.algebra.linear.truncated_svd"]
        # svd outputs U: (B, m, r), S: (B, r), Vt: (B, r, n) — none square.
        out = agent.tensor_eigenvalues(reg, f"{svd_hex}:0")
        assert "error" in out


class TestTensorSingularValues:
    def test_top_k_svd(self) -> None:
        reg, by_op = _build_registry()
        lap_hex = by_op["holonomy_lib.spectral.laplacian.combinatorial"]
        out = agent.tensor_singular_values(reg, lap_hex, k=3)
        assert out["k"] == 3
        assert len(out["values"]) == 4  # batched
        # SVs are non-negative and descending.
        for vals in out["values"]:
            assert vals[0] >= vals[1] >= vals[2] >= 0


class TestTensorNorm:
    def test_frobenius_default(self) -> None:
        reg, by_op = _build_registry()
        lap_hex = by_op["holonomy_lib.spectral.laplacian.combinatorial"]
        out = agent.tensor_norm(reg, lap_hex)
        assert out["order"] == "frobenius"
        assert "per_batch" in out
        assert len(out["per_batch"]) == 4

    def test_spectral_norm(self) -> None:
        reg, by_op = _build_registry()
        lap_hex = by_op["holonomy_lib.spectral.laplacian.combinatorial"]
        out = agent.tensor_norm(reg, lap_hex, order="spectral")
        assert out["order"] == "spectral"
        assert "per_batch" in out
        for v in out["per_batch"]:
            assert v >= 0

    def test_unknown_order_errors(self) -> None:
        reg, by_op = _build_registry()
        lap_hex = by_op["holonomy_lib.spectral.laplacian.combinatorial"]
        out = agent.tensor_norm(reg, lap_hex, order="L1.5")
        assert "error" in out


class TestTensorCompare:
    def test_max_abs_diff_finds_anomaly(self) -> None:
        """If A and B are identical tensors, max_abs is 0. If they
        differ by a constant, max_abs equals that constant.
        """
        from holonomy_lib import provenance
        A = torch.ones(1, 3, 3, dtype=torch.float64)
        B = A * 1.0
        with provenance.record(cache_tensors=True) as reg:
            L_a = laplacian.combinatorial(A)
            L_b = laplacian.combinatorial(B + 0.01)
        nodes = list(reg)
        out = agent.tensor_compare(reg, nodes[0].hex, nodes[1].hex)
        # The two laplacians differ; diff should be > 0.
        assert out["value"] > 0

    def test_shape_mismatch_errors(self) -> None:
        reg, by_op = _build_registry()
        lap_hex = by_op["holonomy_lib.spectral.laplacian.combinatorial"]
        svd_u_hex = f"{by_op['holonomy_lib.algebra.linear.truncated_svd']}:0"
        out = agent.tensor_compare(reg, lap_hex, svd_u_hex)
        assert "error" in out
        assert "shape mismatch" in out["error"]

    def test_cosine_metric(self) -> None:
        from holonomy_lib import provenance
        # A non-degenerate adjacency: random off-diagonal entries so
        # the Laplacian isn't the zero matrix (which would make
        # cosine 0/0 undefined).
        A = torch.rand(1, 4, 4, dtype=torch.float64, generator=_seeded(99))
        A = (A + A.mT).abs()
        A.diagonal(dim1=-2, dim2=-1).zero_()
        with provenance.record(cache_tensors=True) as reg:
            laplacian.combinatorial(A)
        hex_id = next(iter(reg)).hex
        out = agent.tensor_compare(reg, hex_id, hex_id, metric="cosine")
        # Cosine of a non-zero vector with itself is 1.
        assert abs(out["value"] - 1.0) < 1e-6


class TestOpDocstring:
    def test_returns_signature_and_docstring(self) -> None:
        out = agent.op_docstring("holonomy_lib.spectral.laplacian.combinatorial")
        assert out["op_id"] == "holonomy_lib.spectral.laplacian.combinatorial"
        assert "signature" in out
        assert "docstring" in out
        # The combinatorial laplacian's docstring should mention "Laplacian".
        assert "laplacian" in out["docstring"].lower()

    def test_unknown_op_id_errors(self) -> None:
        out = agent.op_docstring("not.an.op")
        assert "error" in out
