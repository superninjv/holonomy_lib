"""Content-addressable provenance protocol for math primitives.

Core types:
  ProvenanceNode    — one operation in the Merkle DAG.
  ProvenanceRegistry — content-addressed store of nodes + tensor cache.
  with_provenance   — decorator that wraps a primitive so it emits a node
                      when called inside a `record()` context.
  record            — context manager that activates recording.

Hex computation (see `_op_hex` and `_tensor_content_hex`):
  - For a user-supplied tensor with no upstream op:
      hex = sha256(shape + dtype + tensor.bytes)[:HEX_PREFIX_LEN]
  - For the output of an op:
      hex = sha256(op_id + op_version + canonical(params)
                   + ":".join(input_hexes))[:HEX_PREFIX_LEN]
  - Multi-output ops: each output i carries the hex `op_hex:i`.

The protocol is opt-in: outside a `record()` context the decorator is
transparent. Performance impact when not recording is one Python
attribute lookup per call.

Thread-safety: the recording context is thread-local. Sub-threads do
not inherit the parent's recording state.
"""

from __future__ import annotations

import hashlib
import json
import threading
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from functools import wraps
from typing import Any, Callable, Iterator, Optional

import torch


# How many hex chars to keep from the sha256 prefix. 16 = 64 bits ≈ 2^32
# expected ops before collision, more than enough for interactive
# sessions. Bump to 32 (= 128 bits) if you're recording huge runs.
HEX_PREFIX_LEN: int = 16

# Global registry mapping op_id → (callable, op_version). Used for replay
# / lookup. Populated by the @with_provenance decorator at import time.
OP_REGISTRY: dict[str, tuple[Callable, str]] = {}

# Thread-local active recording context.
_local = threading.local()


# ============================================================
# Data model
# ============================================================


@dataclass(frozen=True)
class ProvenanceNode:
    """One operation in the provenance DAG.

    Frozen so nodes are hashable and immutable once registered.

    Attributes:
      hex: content-addressable identifier for this op call.
      op_id: dotted-name identifier, e.g. "synoros_lib.algebra.linear.truncated_svd".
      op_version: version string; bump when the implementation changes.
      params: canonicalized non-tensor kwargs (sorted JSON-serializable).
      input_hexes: hex codes of input tensors, in argument order.
      output_shape: tuple of int tuples — one shape per output tensor.
        Length 1 for single-output ops; len > 1 for tuple-returning ops.
      output_dtype: tuple of dtype names, one per output.
    """

    hex: str
    op_id: str
    op_version: str
    params: str  # canonical JSON; dict would make node unhashable
    input_hexes: tuple[str, ...]
    output_shape: tuple[tuple[int, ...], ...]
    output_dtype: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def parsed_params(self) -> dict[str, Any]:
        """JSON-decode `params` back into a dict."""
        return json.loads(self.params) if self.params else {}


# ============================================================
# Registry
# ============================================================


class ProvenanceRegistry:
    """Content-addressed registry of provenance nodes.

    Lookup by hex, query by op_id, export to networkx / dataframe / dict,
    and substitute (activation patching) on subsequent calls.
    """

    def __init__(self, cache_tensors: bool = False):
        self._nodes: dict[str, ProvenanceNode] = {}
        self._tensor_cache: dict[str, torch.Tensor] = {}
        self._cache_tensors: bool = cache_tensors
        # tensor_id (Python id) → hex, for chaining inputs to upstream outputs.
        self._tensor_id_to_hex: dict[int, str] = {}
        # User-set substitutions: hex → tensor (or tuple of tensors).
        self._substitutions: dict[str, Any] = {}

    # ----- low-level registration -----

    def _register(
        self,
        node: ProvenanceNode,
        output: Any,
    ) -> None:
        self._nodes[node.hex] = node
        if self._cache_tensors:
            if isinstance(output, torch.Tensor):
                self._tensor_cache[node.hex] = output
            elif isinstance(output, tuple):
                for i, o in enumerate(output):
                    if isinstance(o, torch.Tensor):
                        self._tensor_cache[f"{node.hex}:{i}"] = o

    def _register_tensor_hex(self, t: torch.Tensor, hex_id: str) -> None:
        """Tag a tensor with its hex so downstream ops can chain."""
        self._tensor_id_to_hex[id(t)] = hex_id

    def _lookup_tensor_hex(self, t: torch.Tensor) -> Optional[str]:
        return self._tensor_id_to_hex.get(id(t))

    # ----- public query API -----

    def __getitem__(self, hex_id: str) -> ProvenanceNode:
        return self._nodes[hex_id]

    def __contains__(self, hex_id: str) -> bool:
        return hex_id in self._nodes

    def __iter__(self) -> Iterator[ProvenanceNode]:
        return iter(self._nodes.values())

    def __len__(self) -> int:
        return len(self._nodes)

    def where(
        self, op_id: Optional[str] = None, op_version: Optional[str] = None,
    ) -> list[ProvenanceNode]:
        """Filter nodes by op_id and/or op_version."""
        out = list(self)
        if op_id is not None:
            out = [n for n in out if n.op_id == op_id]
        if op_version is not None:
            out = [n for n in out if n.op_version == op_version]
        return out

    def get_tensor(self, hex_id: str) -> Optional[torch.Tensor]:
        """Retrieve a cached output tensor by hex (requires cache_tensors=True)."""
        return self._tensor_cache.get(hex_id)

    def parents(self, hex_id: str) -> list[ProvenanceNode]:
        """Direct upstream nodes (input_hexes of `hex_id`)."""
        node = self._nodes[hex_id]
        # Strip multi-output index suffix when looking up parents.
        parent_ids = [h.split(":")[0] for h in node.input_hexes]
        return [self._nodes[p] for p in parent_ids if p in self._nodes]

    def ancestors(self, hex_id: str) -> set[str]:
        """All transitive upstream hexes — the Merkle DAG above `hex_id`."""
        seen: set[str] = set()
        stack = [hex_id]
        while stack:
            h = stack.pop()
            base = h.split(":")[0]
            if base in seen or base not in self._nodes:
                continue
            seen.add(base)
            stack.extend(self._nodes[base].input_hexes)
        return seen

    # ----- substitution / activation patching -----

    @contextmanager
    def substitute(self, mapping: dict[str, Any]) -> Iterator["ProvenanceRegistry"]:
        """Within this context, decorated primitives that *would* produce a
        hex in `mapping` instead return the mapped value.

        Use case: activation patching à la TransformerLens, applied to
        math primitives. The caller re-runs the chain; downstream ops
        compute on the substituted tensor.

        Example:
          with provenance.record() as reg:
              result = some_pipeline(x)
              target = reg.where(op_id="my.op")[0].hex
              with reg.substitute({target: zeros}):
                  ablated = some_pipeline(x)  # `target` returns zeros
        """
        prev = dict(self._substitutions)
        self._substitutions.update(mapping)
        try:
            yield self
        finally:
            self._substitutions = prev

    # ----- export / interop -----

    def to_networkx(self):
        """Export the DAG as a NetworkX DiGraph (requires networkx)."""
        import networkx as nx  # optional dep — imported lazily

        G = nx.DiGraph()
        for node in self:
            G.add_node(node.hex, **{
                "op_id": node.op_id,
                "op_version": node.op_version,
                "params": node.params,
                "output_shape": node.output_shape,
                "output_dtype": node.output_dtype,
            })
            for in_hex in node.input_hexes:
                base = in_hex.split(":")[0]
                G.add_edge(base, node.hex)
        return G

    def to_dataframe(self):
        """Export node table as a pandas DataFrame (requires pandas)."""
        import pandas as pd  # optional dep
        return pd.DataFrame([n.to_dict() for n in self])

    def to_dict(self) -> dict[str, Any]:
        """Plain-Python JSON-friendly export."""
        return {"nodes": [n.to_dict() for n in self]}


# ============================================================
# Recording context
# ============================================================


@contextmanager
def record(cache_tensors: bool = False) -> Iterator[ProvenanceRegistry]:
    """Activate provenance recording for decorated primitives.

    Args:
      cache_tensors: if True, store output tensors in the registry so
        `registry.get_tensor(hex)` returns them. Costs memory; off by
        default to keep the overhead near zero.

    Yields:
      The active ProvenanceRegistry.
    """
    registry = ProvenanceRegistry(cache_tensors=cache_tensors)
    prev = getattr(_local, "context", None)
    _local.context = registry
    try:
        yield registry
    finally:
        _local.context = prev


def _current_context() -> Optional[ProvenanceRegistry]:
    return getattr(_local, "context", None)


# ============================================================
# Hashing helpers
# ============================================================


def _canonical_params(kwargs: dict[str, Any]) -> str:
    """JSON-canonicalize non-tensor kwargs (sorted keys, str-defaulted)."""
    cleaned = {
        k: v for k, v in kwargs.items()
        if not isinstance(v, torch.Tensor)
    }
    return json.dumps(cleaned, sort_keys=True, default=str)


def _tensor_content_hex(t: torch.Tensor) -> str:
    """Content hash of a tensor: shape + dtype + bytes (sha256 prefix)."""
    h = hashlib.sha256()
    h.update(str(tuple(t.shape)).encode())
    h.update(str(t.dtype).encode())
    # Hash bytes — move to CPU and contiguous first.
    h.update(t.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()[:HEX_PREFIX_LEN]


def _op_hex(
    op_id: str, op_version: str, params_json: str, input_hexes: tuple[str, ...],
) -> str:
    """Deterministic hex for an op call from its identity + inputs."""
    h = hashlib.sha256()
    h.update(op_id.encode())
    h.update(b"\0")
    h.update(op_version.encode())
    h.update(b"\0")
    h.update(params_json.encode())
    h.update(b"\0")
    for in_hex in input_hexes:
        h.update(in_hex.encode())
        h.update(b"\0")
    return h.hexdigest()[:HEX_PREFIX_LEN]


def _resolve_tensor_hex(t: torch.Tensor, ctx: ProvenanceRegistry) -> str:
    """Get hex for an input tensor: look up in context, else content-hash it."""
    upstream = ctx._lookup_tensor_hex(t)
    if upstream is not None:
        return upstream
    h = _tensor_content_hex(t)
    ctx._register_tensor_hex(t, h)
    return h


# ============================================================
# Decorator
# ============================================================


def with_provenance(op_id: str, op_version: str = "0.1") -> Callable:
    """Decorator: emit a provenance node when the wrapped op runs inside
    a `record()` context. Outside of recording, the decorator is
    transparent — single attribute lookup per call.

    Args:
      op_id: stable dotted-name identifier (e.g.
        "synoros_lib.algebra.linear.truncated_svd"). Used as the op
        identity in the hex computation; changing it invalidates all
        downstream hexes that depend on it.
      op_version: version string, bumped when the implementation changes
        in a way that should invalidate cached hashes. Defaults to "0.1".

    The decorated function may take any positional/keyword args. Tensor
    positional args contribute their hex to the op's hex; all other
    arguments are serialized into `params` via JSON.
    """
    def decorator(fn: Callable) -> Callable:
        if op_id in OP_REGISTRY and OP_REGISTRY[op_id][0] is not fn:
            raise ValueError(
                f"op_id {op_id!r} already registered to a different function"
            )
        OP_REGISTRY[op_id] = (fn, op_version)

        @wraps(fn)
        def wrapper(*args, **kwargs):
            ctx = _current_context()
            if ctx is None:
                return fn(*args, **kwargs)

            # Resolve input hexes for tensor positional args, in order.
            input_hexes: list[str] = []
            for arg in args:
                if isinstance(arg, torch.Tensor):
                    input_hexes.append(_resolve_tensor_hex(arg, ctx))
            # Tensor kwargs also contribute, sorted by key for determinism.
            for k in sorted(kwargs):
                v = kwargs[k]
                if isinstance(v, torch.Tensor):
                    input_hexes.append(f"{k}={_resolve_tensor_hex(v, ctx)}")

            params_json = _canonical_params(kwargs)
            this_hex = _op_hex(
                op_id, op_version, params_json, tuple(input_hexes),
            )

            # Substitution: if user has overridden this op-call's hex,
            # return their value instead of computing.
            if this_hex in ctx._substitutions:
                output = ctx._substitutions[this_hex]
            else:
                output = fn(*args, **kwargs)

            # Extract output shape/dtype metadata.
            if isinstance(output, torch.Tensor):
                shapes = (tuple(output.shape),)
                dtypes = (str(output.dtype),)
            elif isinstance(output, tuple):
                shapes = tuple(
                    tuple(o.shape) if isinstance(o, torch.Tensor) else ()
                    for o in output
                )
                dtypes = tuple(
                    str(o.dtype) if isinstance(o, torch.Tensor) else type(o).__name__
                    for o in output
                )
            else:
                shapes = ()
                dtypes = (type(output).__name__,)

            node = ProvenanceNode(
                hex=this_hex,
                op_id=op_id,
                op_version=op_version,
                params=params_json,
                input_hexes=tuple(input_hexes),
                output_shape=shapes,
                output_dtype=dtypes,
            )
            ctx._register(node, output)

            # Tag each output tensor with its hex so downstream ops chain.
            if isinstance(output, torch.Tensor):
                ctx._register_tensor_hex(output, this_hex)
            elif isinstance(output, tuple):
                for i, o in enumerate(output):
                    if isinstance(o, torch.Tensor):
                        ctx._register_tensor_hex(o, f"{this_hex}:{i}")

            return output

        return wrapper

    return decorator
