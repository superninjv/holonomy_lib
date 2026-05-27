"""MCP server: expose a saved ProvenanceRegistry as agent tools.

Lets an LLM agent (Claude / GPT / any MCP client) inspect a recorded
provenance DAG from outside the Python process that produced it.
The registry is loaded from JSON (with optional disk-cache attached)
and the following tools are exposed:

  list_ops()                       — distinct op_ids in the registry
  where(op_id, op_version)         — nodes matching the filter
  node_info(hex)                   — full node metadata
  ancestors(hex)                   — transitive upstream hex set
  get_tensor_summary(hex)          — shape + dtype + mean/std/min/max
                                       (not the bytes — keeps the
                                       MCP message small)
  replay(target_hex, substitutions) — full re-execution from
                                       substitutions; returns a
                                       summary of new outputs

Entry point: `python -m holonomy_lib.provenance.mcp` reads the
registry path from the `HOLONOMY_PROVENANCE_REGISTRY` env var, loads
it, and starts the server on stdio (the MCP standard transport).

The MCP SDK (`mcp` package) is an optional dependency declared in
the `[mcp]` extras group. Importing this module without `mcp`
installed raises a clear ImportError that points to the install.

Limitations (v0.3):
  - File-loaded registries only — no live attachment to a running
    Python process. Socket-attached / live mode is planned for v0.4.
  - The registry is loaded once at server start; subsequent on-disk
    changes won't be visible until the server restarts.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Optional

import torch

from holonomy_lib.provenance.protocol import ProvenanceRegistry

try:
    from mcp.server.fastmcp import FastMCP
except ImportError as e:
    raise ImportError(
        "holonomy_lib.provenance.mcp requires the `mcp` package. "
        "Install it via `pip install 'holonomy-lib[mcp]'`."
    ) from e


def build_server(registry: ProvenanceRegistry) -> FastMCP:
    """Construct an MCP server bound to the given registry.

    Exposes the six query tools listed in the module docstring. The
    registry is captured by closure; reloading requires building a
    new server.
    """
    server = FastMCP("holonomy-lib-provenance")

    @server.tool()
    def list_ops() -> list[str]:
        """Return the sorted distinct op_ids present in the registry."""
        return sorted({n.op_id for n in registry})

    @server.tool()
    def where(
        op_id: Optional[str] = None,
        op_version: Optional[str] = None,
    ) -> list[dict[str, Any]]:
        """Return nodes matching op_id and/or op_version filters."""
        return [
            n.to_dict()
            for n in registry.where(op_id=op_id, op_version=op_version)
        ]

    @server.tool()
    def node_info(hex_id: str) -> dict[str, Any]:
        """Return the full ProvenanceNode dict for a given hex.

        Raises ValueError if the hex isn't in the registry.
        """
        if hex_id not in registry:
            raise ValueError(f"hex {hex_id!r} not in registry")
        return registry[hex_id].to_dict()

    @server.tool()
    def ancestors(hex_id: str) -> list[str]:
        """Return the sorted transitive upstream hex set."""
        if hex_id not in registry:
            raise ValueError(f"hex {hex_id!r} not in registry")
        return sorted(registry.ancestors(hex_id))

    @server.tool()
    def get_tensor_summary(hex_id: str) -> dict[str, Any]:
        """Return shape + dtype + per-batch summary stats for a cached tensor.

        Does NOT return the tensor bytes (those would blow up the
        MCP message size). Stats are mean/std/min/max as floats.
        Raises ValueError if the tensor isn't cached.
        """
        t = registry.get_tensor(hex_id)
        if t is None:
            raise ValueError(
                f"hex {hex_id!r} has no cached tensor "
                f"(recording may not have used cache_tensors=True)"
            )
        return {
            "hex": hex_id,
            "shape": list(t.shape),
            "dtype": str(t.dtype),
            "device": str(t.device),
            "numel": t.numel(),
            "mean": float(t.to(torch.float64).mean().item()),
            "std": float(t.to(torch.float64).std().item()) if t.numel() > 1 else 0.0,
            "min": float(t.to(torch.float64).min().item()),
            "max": float(t.to(torch.float64).max().item()),
        }

    @server.tool()
    def replay(
        substitutions_summary: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Re-execute downstream of one or more substituted hexes.

        The MCP wire format can't carry torch tensors directly, so
        substitutions are described as filled-with-zeros tensors of
        the requested shape/dtype. For richer substitutions, drive
        replay() from a local Python process via load() + replay().

        Args:
          substitutions_summary: list of dicts with keys
            {"hex": str, "shape": list[int], "dtype": str (optional)}.
            Defaults dtype to torch.float64.

        Returns:
          {"replayed_count": int, "new_outputs": [{hex, shape, mean,
            std}, ...]} — one summary entry per re-executed node.
        """
        substitutions: dict[str, torch.Tensor] = {}
        for entry in substitutions_summary:
            hex_id = entry["hex"]
            shape = entry["shape"]
            dtype_name = entry.get("dtype", "torch.float64").split(".")[-1]
            dtype = getattr(torch, dtype_name)
            substitutions[hex_id] = torch.zeros(*shape, dtype=dtype)
        new_outputs = registry.replay(substitutions)
        summary_entries = []
        for h, t in new_outputs.items():
            summary_entries.append({
                "hex": h,
                "shape": list(t.shape),
                "mean": float(t.to(torch.float64).mean().item()),
                "std": (
                    float(t.to(torch.float64).std().item())
                    if t.numel() > 1 else 0.0
                ),
            })
        return {
            "replayed_count": len(new_outputs),
            "new_outputs": summary_entries,
        }

    return server


def _load_registry_from_env() -> ProvenanceRegistry:
    """Locate and load the registry pointed to by HOLONOMY_PROVENANCE_REGISTRY."""
    path = os.environ.get("HOLONOMY_PROVENANCE_REGISTRY")
    if not path:
        raise RuntimeError(
            "HOLONOMY_PROVENANCE_REGISTRY environment variable is not set. "
            "Point it at a JSON file written by ProvenanceRegistry.save() "
            "before starting the MCP server."
        )
    p = Path(path).expanduser().resolve()
    if not p.exists():
        raise FileNotFoundError(f"Registry file not found: {p}")
    return ProvenanceRegistry.load(p)


def main() -> None:
    """Entry point: load registry from env and start the stdio server."""
    registry = _load_registry_from_env()
    server = build_server(registry)
    server.run()


if __name__ == "__main__":
    main()
