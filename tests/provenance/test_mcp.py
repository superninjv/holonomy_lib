"""Smoke tests for the MCP server module.

Skipped when the `mcp` SDK isn't installed (the typical dev venv).
Run with `pip install 'holonomy-lib[mcp]'` to enable.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import pytest
import torch


def _seeded(seed: int) -> torch.Generator:
    g = torch.Generator()
    g.manual_seed(seed)
    return g


pytest.importorskip("mcp")  # skip the whole module if MCP SDK absent


from holonomy_lib import provenance
from holonomy_lib.algebra import truncated_svd
from holonomy_lib.spectral import laplacian


def _build_fixture_registry() -> provenance.ProvenanceRegistry:
    A = torch.randn(1, 4, 4, dtype=torch.float64, generator=_seeded(0))
    A = (A + A.mT).abs() + torch.eye(4, dtype=torch.float64).unsqueeze(dim=0)
    with provenance.record(cache_tensors=True) as reg:
        L = laplacian.combinatorial(A)
        truncated_svd(L, r=2, mode="exact")
    return reg


class TestBuildServer:
    def test_build_server_creates_six_tools(self):
        """build_server returns a FastMCP whose tool surface includes
        all six tools the module advertises."""
        from holonomy_lib.provenance.mcp import build_server
        reg = _build_fixture_registry()
        server = build_server(reg)
        # FastMCP keeps tools on `_tool_manager._tools` (private but
        # stable across recent SDK versions). Use the documented
        # `.list_tools()` async API if `_tool_manager` is missing.
        names = set()
        if hasattr(server, "_tool_manager"):
            names = set(server._tool_manager._tools.keys())
        else:
            import asyncio
            tools = asyncio.run(server.list_tools())
            names = {t.name for t in tools}
        for expected in [
            "list_ops", "where", "node_info", "ancestors",
            "get_tensor_summary", "replay",
        ]:
            assert expected in names, f"missing tool: {expected}"

    def test_get_tensor_summary_returns_stats(self):
        """get_tensor_summary returns shape/dtype/stats for a cached node."""
        from holonomy_lib.provenance.mcp import build_server
        reg = _build_fixture_registry()
        server = build_server(reg)
        # Pull the underlying function out of the tool wrapper.
        # FastMCP wraps callables, but the underlying fn is accessible.
        lap_node = reg.where(
            op_id="holonomy_lib.spectral.laplacian.combinatorial",
        )[0]
        # Find the registered function. SDK shape varies; use
        # _tool_manager when present.
        if hasattr(server, "_tool_manager"):
            tool = server._tool_manager._tools["get_tensor_summary"]
            fn = tool.fn if hasattr(tool, "fn") else tool.callable
        else:
            pytest.skip("MCP SDK version doesn't expose tool internals")
        result = fn(hex_id=lap_node.hex)
        assert result["hex"] == lap_node.hex
        assert "shape" in result
        assert "mean" in result
        assert "std" in result

    def test_list_ops_returns_distinct_op_ids(self):
        from holonomy_lib.provenance.mcp import build_server
        reg = _build_fixture_registry()
        server = build_server(reg)
        if not hasattr(server, "_tool_manager"):
            pytest.skip("MCP SDK version doesn't expose tool internals")
        tool = server._tool_manager._tools["list_ops"]
        fn = tool.fn if hasattr(tool, "fn") else tool.callable
        ops = fn()
        assert "holonomy_lib.spectral.laplacian.combinatorial" in ops
        assert "holonomy_lib.algebra.linear.truncated_svd" in ops


class TestLoadRegistryFromEnv:
    def test_env_unset_raises(self):
        from holonomy_lib.provenance.mcp import _load_registry_from_env
        # Make sure the env var isn't set.
        old = os.environ.pop("HOLONOMY_PROVENANCE_REGISTRY", None)
        try:
            with pytest.raises(RuntimeError, match="HOLONOMY_PROVENANCE_REGISTRY"):
                _load_registry_from_env()
        finally:
            if old is not None:
                os.environ["HOLONOMY_PROVENANCE_REGISTRY"] = old

    def test_env_missing_file_raises(self):
        from holonomy_lib.provenance.mcp import _load_registry_from_env
        with tempfile.TemporaryDirectory() as d:
            os.environ["HOLONOMY_PROVENANCE_REGISTRY"] = str(
                Path(d) / "nonexistent.json"
            )
            try:
                with pytest.raises(FileNotFoundError):
                    _load_registry_from_env()
            finally:
                os.environ.pop("HOLONOMY_PROVENANCE_REGISTRY", None)

    def test_env_pointing_at_saved_registry_loads(self):
        from holonomy_lib.provenance.mcp import _load_registry_from_env
        reg = _build_fixture_registry()
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "reg.json"
            reg.save(path)
            os.environ["HOLONOMY_PROVENANCE_REGISTRY"] = str(path)
            try:
                loaded = _load_registry_from_env()
                assert len(loaded) == len(reg)
            finally:
                os.environ.pop("HOLONOMY_PROVENANCE_REGISTRY", None)
