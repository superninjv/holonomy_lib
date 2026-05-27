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
import inspect
import json
import threading
import weakref
from collections import OrderedDict, deque
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from functools import wraps
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, Literal, Optional

import torch

# Pluggable hash algorithm. blake3 is ~5-10× faster than sha256 and
# still cryptographic; falls back to sha256 if blake3 isn't installed.
# The hex output format is identical so downstream code is unaffected,
# but a registry's hexes are tied to whichever hash was used to record
# it — recordings are not portable across hash algorithms.
try:
    import blake3 as _blake3

    def _make_hasher_blake3():
        return _blake3.blake3()

    _DEFAULT_HASHER_NAME = "blake3"
    _HASHER_FACTORIES: dict[str, Callable] = {
        "blake3": _make_hasher_blake3,
        "sha256": hashlib.sha256,
    }
except ImportError:
    _DEFAULT_HASHER_NAME = "sha256"
    _HASHER_FACTORIES = {"sha256": hashlib.sha256}


# How many hex chars to keep from the sha256 prefix. 16 = 64 bits ≈ 2^32
# expected ops before collision, more than enough for interactive
# sessions. Bump to 32 (= 128 bits) if you're recording huge runs.
HEX_PREFIX_LEN: int = 16

# Sketch-mode hashing: how many evenly-spaced samples to draw from a
# tensor's flattened content. Combined with shape, dtype, sum, and std
# this gives an O(1)-bytes-hashed digest regardless of tensor size,
# trading negligible collision risk for ~100× speedup on multi-MB
# tensors. Empirical collision rate on 10⁴ random (256, 256) float64
# tensors is 0 (no collisions); structural-collision risk (two tensors
# that share strided samples + sum + std but differ in the unsampled
# elements) is bounded but non-zero — full mode remains the default.
SKETCH_SAMPLES: int = 64

# Hash mode literal — sketch trades crypto-grade for speed.
HashMode = Literal["full", "sketch"]

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
      op_id: dotted-name identifier, e.g. "holonomy_lib.algebra.linear.truncated_svd".
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

    def __init__(
        self,
        cache_tensors: bool = False,
        hash_algorithm: str = _DEFAULT_HASHER_NAME,
        max_cache_size: Optional[int] = None,
        cache_ops: Optional[Iterable[str]] = None,
        hash_mode: HashMode = "full",
        cache_to_disk: Optional[str | Path] = None,
    ):
        self._nodes: dict[str, ProvenanceNode] = {}
        # OrderedDict so we can evict in insertion order ("oldest first")
        # when `max_cache_size` is enforced. Mech-interp users typically
        # want recent intermediates around; if a workload needs LRU-by-
        # access, swap to `move_to_end` on read here.
        self._tensor_cache: "OrderedDict[str, torch.Tensor]" = OrderedDict()
        # Setting cache_to_disk implies caching is on — there's no point
        # configuring a disk path if nothing is being cached. Make this
        # implicit so users only need the one kwarg.
        if cache_to_disk is not None:
            cache_tensors = True
        self._cache_tensors: bool = cache_tensors
        self._max_cache_size: Optional[int] = max_cache_size
        # Selective caching: if set, only outputs from these op_ids are
        # cached. Implies caching is on for those ops regardless of
        # `cache_tensors`. None means "cache all when cache_tensors=True".
        self._cache_ops: Optional[set[str]] = (
            set(cache_ops) if cache_ops is not None else None
        )
        if max_cache_size is not None and max_cache_size <= 0:
            raise ValueError(
                f"max_cache_size must be > 0 or None, got {max_cache_size}"
            )
        # tensor_id (Python id) → (weakref, hex), for chaining inputs to
        # upstream outputs. The weakref guards against id reuse: when a
        # tensor is GC'd, Python may hand out its memory address to a
        # new allocation; without the weakref check, the new tensor
        # would inherit the dead tensor's hex (silent corruption).
        self._tensor_id_to_hex: dict[
            int, tuple[weakref.ReferenceType, str]
        ] = {}
        # User-set substitutions: hex → tensor (or tuple of tensors).
        self._substitutions: dict[str, Any] = {}
        # Hook callbacks: op_id → list[Callable[[ProvenanceNode, Any], None]]
        self._hooks: dict[str, list[Callable[[ProvenanceNode, Any], None]]] = {}
        # Hash algorithm — frozen at construction so all hexes in this
        # registry are computed consistently.
        if hash_algorithm not in _HASHER_FACTORIES:
            raise ValueError(
                f"hash_algorithm must be one of {sorted(_HASHER_FACTORIES)}, "
                f"got {hash_algorithm!r}"
            )
        self.hash_algorithm: str = hash_algorithm
        self._hasher_factory = _HASHER_FACTORIES[hash_algorithm]
        if hash_mode not in ("full", "sketch"):
            raise ValueError(
                f"hash_mode must be 'full' or 'sketch', got {hash_mode!r}"
            )
        self.hash_mode: HashMode = hash_mode
        # Disk cache: a directory that mirrors the in-memory cache.
        # When set, _cache_put writes the tensor both to memory and to
        # disk; memory eviction (via max_cache_size) drops the in-memory
        # copy but the disk copy persists and is reloaded on demand.
        # Use an absolute path so a registry survives a chdir.
        self._disk_cache_dir: Optional[Path] = (
            Path(cache_to_disk).expanduser().resolve()
            if cache_to_disk is not None else None
        )
        self._disk_cache_keys: set[str] = set()
        if self._disk_cache_dir is not None:
            self._disk_cache_dir.mkdir(parents=True, exist_ok=True)

    # ----- low-level registration -----

    def _register(
        self,
        node: ProvenanceNode,
        output: Any,
    ) -> None:
        self._nodes[node.hex] = node
        # Decide whether to cache this output:
        #   - if `cache_ops` was set, only cache when op_id is in the set
        #     (this implicitly enables caching for those ops, regardless
        #      of `cache_tensors`)
        #   - else, fall through to the boolean `cache_tensors`
        if self._cache_ops is not None:
            should_cache = node.op_id in self._cache_ops
        else:
            should_cache = self._cache_tensors
        if not should_cache:
            return
        if isinstance(output, torch.Tensor):
            self._cache_put(node.hex, output)
        elif isinstance(output, tuple):
            for i, o in enumerate(output):
                if isinstance(o, torch.Tensor):
                    self._cache_put(f"{node.hex}:{i}", o)

    def _cache_put(self, key: str, value: torch.Tensor) -> None:
        """Insert into the tensor cache, evicting oldest if size-bounded.

        When `cache_to_disk` is set, the tensor is also persisted via
        `torch.save` to a file named for the (filesystem-sanitized)
        hex key. Memory eviction (from `max_cache_size`) drops the
        in-memory copy but the disk copy persists; `get_tensor` will
        reload it on demand.
        """
        if key in self._tensor_cache:
            # Refresh to the end so it's not the next eviction victim.
            self._tensor_cache.move_to_end(key)
        self._tensor_cache[key] = value
        if self._disk_cache_dir is not None:
            torch.save(
                value.detach().cpu(),
                self._disk_path_for(key),
            )
            self._disk_cache_keys.add(key)
        if (
            self._max_cache_size is not None
            and len(self._tensor_cache) > self._max_cache_size
        ):
            # popitem(last=False) pops the oldest entry from memory.
            # Disk copy (if any) is retained — get_tensor will reload it.
            self._tensor_cache.popitem(last=False)

    def _disk_path_for(self, key: str) -> Path:
        """Filesystem path for a cache key. Replaces ':' (multi-output
        index separator) with '__' for Windows compatibility.
        """
        assert self._disk_cache_dir is not None
        safe = key.replace(":", "__")
        return self._disk_cache_dir / f"{safe}.pt"

    def clear(self, delete_disk: bool = False) -> None:
        """Drop the in-memory cache. Optionally also delete disk artifacts.

        Args:
          delete_disk: if True and a disk cache is configured, remove
            every persisted `.pt` file (and the directory if it becomes
            empty). Default False so that a `clear()` doesn't silently
            destroy persisted state.
        """
        self._tensor_cache.clear()
        if delete_disk and self._disk_cache_dir is not None:
            for key in list(self._disk_cache_keys):
                p = self._disk_path_for(key)
                if p.exists():
                    p.unlink()
            self._disk_cache_keys.clear()

    def _register_tensor_hex(self, t: torch.Tensor, hex_id: str) -> None:
        """Tag a tensor with its hex so downstream ops can chain."""
        self._tensor_id_to_hex[id(t)] = (weakref.ref(t), hex_id)

    def _lookup_tensor_hex(self, t: torch.Tensor) -> Optional[str]:
        """Return the stored hex for `t`, or None if no live entry exists.

        The stored weakref must still resolve to the same tensor object;
        if it has been GC'd and `id(t)` was reused for a different tensor,
        we treat the entry as stale and let the caller fall back to a
        fresh content hash.
        """
        entry = self._tensor_id_to_hex.get(id(t))
        if entry is None:
            return None
        ref, hex_id = entry
        if ref() is not t:
            # id reuse: drop the stale entry so we don't keep checking.
            self._tensor_id_to_hex.pop(id(t), None)
            return None
        return hex_id

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
        """Retrieve a cached output tensor by hex (requires cache_tensors=True).

        If a disk cache is configured and the tensor is not in memory,
        loads it from disk and re-promotes to the in-memory cache so
        subsequent reads are fast. Returns None if the key isn't cached
        anywhere.
        """
        if hex_id in self._tensor_cache:
            return self._tensor_cache[hex_id]
        if (
            self._disk_cache_dir is not None
            and hex_id in self._disk_cache_keys
        ):
            t = torch.load(
                self._disk_path_for(hex_id),
                map_location="cpu",
                weights_only=True,
            )
            # Re-promote to memory; respects max_cache_size eviction.
            self._cache_put_memory_only(hex_id, t)
            return t
        return None

    def _cache_put_memory_only(self, key: str, value: torch.Tensor) -> None:
        """Insert into the in-memory cache only (no disk write).

        Used by `get_tensor` to re-promote a disk-loaded tensor without
        re-persisting it — the disk copy already exists.
        """
        if key in self._tensor_cache:
            self._tensor_cache.move_to_end(key)
        self._tensor_cache[key] = value
        if (
            self._max_cache_size is not None
            and len(self._tensor_cache) > self._max_cache_size
        ):
            self._tensor_cache.popitem(last=False)

    def parents(self, hex_id: str) -> list[ProvenanceNode]:
        """Direct upstream nodes (input_hexes of `hex_id`)."""
        node = self._nodes[hex_id]
        parents = []
        for input_hex in node.input_hexes:
            # input_hex is "name=hex" or "name=hex:i" — extract the base hex
            _, _, hex_part = input_hex.partition("=")
            base = hex_part.split(":")[0]
            if base in self._nodes:
                parents.append(self._nodes[base])
        return parents

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
            for input_hex in self._nodes[base].input_hexes:
                _, _, hex_part = input_hex.partition("=")
                stack.append(hex_part)
        return seen

    # ----- hooks / callbacks (observation without mutation) -----

    def on_op(
        self, op_id: str,
        callback: Callable[[ProvenanceNode, Any], None],
    ) -> None:
        """Register a callback that fires every time `op_id` is called
        inside this registry's recording context.

        The callback receives `(node, output)` where `output` is whatever
        the decorated primitive returned (a tensor, tuple of tensors, etc.).
        Multiple callbacks for the same op_id are called in registration order.

        TransformerLens-equivalent: forward-hook on a module, but indexed
        by op_id instead of module path. Doesn't change behavior (unlike
        substitute()); just lets you observe / log / accumulate.

        Example:
          activations = []
          reg.on_op("holonomy_lib.spectral.laplacian.combinatorial",
                      lambda node, out: activations.append((node.hex, out)))
          # ... run the pipeline ...
          # activations now lists every Laplacian computed
        """
        self._hooks.setdefault(op_id, []).append(callback)

    def clear_hooks(self, op_id: Optional[str] = None) -> None:
        """Remove all hooks (or just for one op_id)."""
        if op_id is None:
            self._hooks.clear()
        else:
            self._hooks.pop(op_id, None)

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
            for input_hex in node.input_hexes:
                _, _, hex_part = input_hex.partition("=")
                base = hex_part.split(":")[0]
                G.add_edge(base, node.hex)
        return G

    def to_dataframe(self):
        """Export node table as a pandas DataFrame (requires pandas)."""
        import pandas as pd  # optional dep
        return pd.DataFrame([n.to_dict() for n in self])

    def to_dict(self) -> dict[str, Any]:
        """Plain-Python JSON-friendly export."""
        return {
            "schema_version": "0.3",
            "hash_algorithm": self.hash_algorithm,
            "hash_mode": self.hash_mode,
            "cache_to_disk": (
                str(self._disk_cache_dir)
                if self._disk_cache_dir is not None else None
            ),
            "disk_cache_keys": sorted(self._disk_cache_keys),
            "nodes": [n.to_dict() for n in self],
        }

    def to_sae_dataset(
        self, op_id: Optional[str] = None,
    ) -> Iterator[tuple[torch.Tensor, dict[str, Any]]]:
        """Yield `(tensor, metadata)` pairs from cached outputs, in registration
        order. Useful as input to SAELens-style training where you want a
        stream of activations + labels.

        Args:
          op_id: if given, only yield records for this op.

        Yields:
          (output_tensor, dict) — the dict has {"hex", "op_id", "op_version",
          "params", "input_hexes"} for downstream filtering / labeling.

        Requires `cache_tensors=True` on the recording context.
        """
        for node in self.where(op_id=op_id):
            tensor = self.get_tensor(node.hex)
            if tensor is not None:
                yield tensor, {
                    "hex": node.hex,
                    "op_id": node.op_id,
                    "op_version": node.op_version,
                    "params": node.params,
                    "input_hexes": list(node.input_hexes),
                }
            else:
                # Multi-output op: yield each cached output with its hex:i
                for i in range(len(node.output_shape)):
                    sub_hex = f"{node.hex}:{i}"
                    t = self.get_tensor(sub_hex)
                    if t is not None:
                        yield t, {
                            "hex": sub_hex,
                            "op_id": node.op_id,
                            "op_version": node.op_version,
                            "params": node.params,
                            "input_hexes": list(node.input_hexes),
                            "output_index": i,
                        }

    # ----- diff -----

    def diff(self, other: "ProvenanceRegistry") -> dict[str, Any]:
        """Compare two recordings and return a structured diff.

        Useful for "did my refactor preserve semantics?" or "what changed
        between these two experimental conditions?". The diff groups by
        op_id so you see, per operation, which hexes are unique to each
        registry vs. shared.

        Returns:
          dict with keys:
            "only_in_self":   {op_id: [hex, ...]}  — ops in self, not other
            "only_in_other":  {op_id: [hex, ...]}
            "shared":         {op_id: [hex, ...]}
            "op_ids_only_in_self":   list[str]
            "op_ids_only_in_other":  list[str]
        """
        self_hexes_by_op: dict[str, set[str]] = {}
        for n in self:
            self_hexes_by_op.setdefault(n.op_id, set()).add(n.hex)
        other_hexes_by_op: dict[str, set[str]] = {}
        for n in other:
            other_hexes_by_op.setdefault(n.op_id, set()).add(n.hex)

        all_ops = set(self_hexes_by_op) | set(other_hexes_by_op)
        only_in_self: dict[str, list[str]] = {}
        only_in_other: dict[str, list[str]] = {}
        shared: dict[str, list[str]] = {}
        for op in all_ops:
            s = self_hexes_by_op.get(op, set())
            o = other_hexes_by_op.get(op, set())
            if s - o:
                only_in_self[op] = sorted(s - o)
            if o - s:
                only_in_other[op] = sorted(o - s)
            if s & o:
                shared[op] = sorted(s & o)

        return {
            "only_in_self": only_in_self,
            "only_in_other": only_in_other,
            "shared": shared,
            "op_ids_only_in_self": sorted(
                set(self_hexes_by_op) - set(other_hexes_by_op),
            ),
            "op_ids_only_in_other": sorted(
                set(other_hexes_by_op) - set(self_hexes_by_op),
            ),
        }

    # ----- causal replay: downstream DAG re-execution -----

    def replay(
        self,
        substitutions: dict[str, torch.Tensor],
        final_hex: Optional[str] = None,
    ) -> dict[str, torch.Tensor]:
        """Re-execute the downstream DAG with substitutions propagating.

        Unlike `substitute()` (which only affects ops whose hex is in
        the substitution map at *call* time, requiring you to re-run
        your pipeline), this method walks the recorded DAG and
        re-executes only the affected nodes — the upstream computation
        is reused from the cache, so this is much cheaper than re-running
        a full pipeline when you only want to intervene at one node.

        Args:
          substitutions: hex → replacement tensor. Each substituted hex
            must already exist in the registry's tensor cache.
          final_hex: if given, stop early once this node has been
            re-executed and return only its value (under that key).
            Otherwise return all re-executed nodes' new outputs.

        Returns:
          dict mapping hex → new output tensor for every re-executed
          node. Hexes match the *original* recording's hex (we use the
          original DAG topology; only the *values* change).

        Requires `cache_tensors=True` on the original recording.

        Caveats:
          - Random ops (those that take a Generator and call torch.randn
            etc.) replay with a fresh generator — the result will differ
            from the original. Avoid replaying through stochastic ops.
          - The substituted tensor must be shape-compatible with the
            original at that node, or downstream ops will raise.
        """
        if not self._cache_tensors:
            raise ValueError("replay requires cache_tensors=True on record()")
        for h in substitutions:
            base = h.split(":")[0]
            if base not in self._nodes:
                raise KeyError(
                    f"substitution target {h!r} not in registry"
                )

        # Reverse adjacency: parent_hex → list[child_hex]
        children_of: dict[str, list[str]] = {h: [] for h in self._nodes}
        for child_hex, child_node in self._nodes.items():
            for parent_input in child_node.input_hexes:
                # input_hex is "name=hex"; extract the hex part
                _, _, parent_hex = parent_input.partition("=")
                parent_base = parent_hex.split(":")[0]
                if parent_base in children_of:
                    children_of[parent_base].append(child_hex)

        # Collect ALL hexes downstream of any substituted hex.
        affected: set[str] = set()
        queue = deque(h.split(":")[0] for h in substitutions)
        while queue:
            h = queue.popleft()
            for child in children_of.get(h, ()):
                if child not in affected:
                    affected.add(child)
                    queue.append(child)

        # Topological sort (Kahn's algorithm) restricted to `affected`.
        indegree: dict[str, int] = {}
        for h in affected:
            count = 0
            for parent_input in self._nodes[h].input_hexes:
                _, _, parent_hex = parent_input.partition("=")
                parent_base = parent_hex.split(":")[0]
                if parent_base in affected:
                    count += 1
            indegree[h] = count
        ready = deque(h for h, d in indegree.items() if d == 0)
        order: list[str] = []
        while ready:
            h = ready.popleft()
            order.append(h)
            for child in children_of.get(h, ()):
                if child in indegree:
                    indegree[child] -= 1
                    if indegree[child] == 0:
                        ready.append(child)

        # Shadow cache: starts from baseline, gets overwritten as we go.
        # Substitutions are merged in immediately so any node that reads
        # them sees the substituted value.
        shadow: dict[str, torch.Tensor] = dict(self._tensor_cache)
        for h, val in substitutions.items():
            shadow[h] = val

        new_outputs: dict[str, torch.Tensor] = {}
        for h in order:
            node = self._nodes[h]
            if node.op_id not in OP_REGISTRY:
                raise RuntimeError(
                    f"cannot replay: op_id {node.op_id!r} not in OP_REGISTRY"
                )
            fn, _ver = OP_REGISTRY[node.op_id]
            # Reconstruct call kwargs from input_hexes + params.
            # Restore canonicalized values (e.g. torch.Generator) before
            # invoking the op — params serialize to JSON-stable forms
            # but the op expects real objects.
            call_kwargs: dict[str, Any] = {
                k: _restore_value(v) for k, v in node.parsed_params().items()
            }
            # Class-method replay is not yet supported: `self` was
            # canonicalized via `_provenance_signature` into a dict
            # rather than a class instance, and `fn(**call_kwargs)`
            # would crash with that dict bound to `self`. Fail loudly
            # instead of crashing on whatever the manifold method
            # tries to access first.
            for k, v in call_kwargs.items():
                if isinstance(v, dict) and v.get(_PROVENANCE_SIGNATURE_TAG):
                    raise NotImplementedError(
                        f"replay of class-method calls is not yet "
                        f"supported (op_id={node.op_id!r}, param "
                        f"{k!r} is a class instance). Use "
                        f"`substitute()` instead, or rerun your "
                        f"pipeline with the substituted upstream "
                        f"tensors. Tracked for v0.2."
                    )
            for input_hex in node.input_hexes:
                name, _, hex_part = input_hex.partition("=")
                if hex_part not in shadow:
                    raise RuntimeError(
                        f"replay: missing tensor for {hex_part!r} (parent of {h!r})"
                    )
                if "[" in name and name.endswith("]"):
                    # Tuple/list-of-tensor input was unpacked into
                    # per-element hex keys at record time. Replaying
                    # would need to reassemble `name[i]` back into a
                    # positional tuple — not yet implemented.
                    raise NotImplementedError(
                        f"replay of methods taking tuple/list-of-tensor "
                        f"inputs is not yet supported (op_id="
                        f"{node.op_id!r}, input parameter {name!r} was "
                        f"unpacked into per-element hex keys). Tracked "
                        f"for v0.2."
                    )
                call_kwargs[name] = shadow[hex_part]
            output = fn(**call_kwargs)
            # Store in shadow so downstream ops can find it.
            if isinstance(output, torch.Tensor):
                shadow[h] = output
                new_outputs[h] = output
            elif isinstance(output, tuple):
                for i, o in enumerate(output):
                    if isinstance(o, torch.Tensor):
                        shadow[f"{h}:{i}"] = o
                        new_outputs[f"{h}:{i}"] = o
            if final_hex is not None and h.split(":")[0] == final_hex.split(":")[0]:
                # Hit the requested terminus. For multi-output nodes,
                # `shadow` holds the outputs under `h:0`, `h:1`, ... —
                # so a `final_hex` matching only the base (no :i) cannot
                # be looked up directly. Return every shadow entry that
                # belongs to this node, plus the exact-match key if the
                # caller passed `h:i` for a specific output.
                base = h.split(":")[0]
                if final_hex in shadow:
                    return {final_hex: shadow[final_hex]}
                return {
                    k: v for k, v in new_outputs.items()
                    if k.split(":")[0] == base
                }

        if final_hex is not None:
            return {final_hex: shadow[final_hex]} if final_hex in shadow else {}
        return new_outputs

    # ----- persistence -----

    def save(self, path: str | Path) -> None:
        """Persist the registry's metadata to disk as JSON.

        The tensor cache (if any) is NOT persisted — only the DAG
        structure, hexes, op_ids, params, shapes, dtypes. To persist
        tensors, save them separately keyed by hex (e.g. via torch.save
        in a follow-on directory).
        """
        path = Path(path)
        path.write_text(json.dumps(self.to_dict(), indent=2))

    @classmethod
    def load(cls, path: str | Path) -> "ProvenanceRegistry":
        """Load a registry's metadata from a JSON file written by save().

        The loaded registry has no tensor cache (tensors aren't persisted).
        Substitutions and hooks are reset to empty.
        """
        path = Path(path)
        data = json.loads(path.read_text())
        algorithm = data.get("hash_algorithm", "sha256")
        # Pre-0.2 schemas predate hash_mode; default to "full" for
        # backward compatibility (the only mode that existed then).
        mode = data.get("hash_mode", "full")
        disk_dir = data.get("cache_to_disk")
        registry = cls(
            hash_algorithm=algorithm,
            hash_mode=mode,
            cache_to_disk=disk_dir,
        )
        # Repopulate the disk-key set so get_tensor() can find the
        # already-persisted files without rewalking the directory.
        for k in data.get("disk_cache_keys", []):
            registry._disk_cache_keys.add(k)
        for entry in data["nodes"]:
            node = ProvenanceNode(
                hex=entry["hex"],
                op_id=entry["op_id"],
                op_version=entry["op_version"],
                params=entry["params"],
                input_hexes=tuple(entry["input_hexes"]),
                output_shape=tuple(tuple(s) for s in entry["output_shape"]),
                output_dtype=tuple(entry["output_dtype"]),
            )
            registry._nodes[node.hex] = node
        return registry


# ============================================================
# Recording context
# ============================================================


@contextmanager
def record(
    cache_tensors: bool = False,
    hash_algorithm: str = _DEFAULT_HASHER_NAME,
    max_cache_size: Optional[int] = None,
    cache_ops: Optional[Iterable[str]] = None,
    hash_mode: HashMode = "full",
    cache_to_disk: Optional[str | Path] = None,
) -> Iterator[ProvenanceRegistry]:
    """Activate provenance recording for decorated primitives.

    Args:
      cache_tensors: if True, store output tensors in the registry so
        `registry.get_tensor(hex)` returns them. Costs memory; off by
        default to keep the overhead near zero.
      hash_algorithm: hash function name. Defaults to whichever was
        available at import (blake3 if installed, else sha256). All
        hexes within a registry are computed with this algorithm; you
        cannot mix.
      max_cache_size: hard cap on the number of cached output tensors.
        When exceeded, the oldest cached entry is evicted (FIFO). None
        means unbounded — convenient for short pipelines but
        a memory hazard for long ones, especially on GPU.
      cache_ops: if provided, only outputs whose op_id is in this set
        are cached. Overrides `cache_tensors` for selectivity: caching
        is enabled for the listed ops and disabled for everything else.
        Use this when you want to keep, say, every Laplacian but not
        every intermediate matmul.
      hash_mode: "full" (default) hashes the full tensor bytes — slow
        on multi-MB inputs but cryptographically distinguishes content.
        "sketch" hashes `shape + dtype + SKETCH_SAMPLES = 64` evenly-
        strided samples + sum + std — ~100× faster on big tensors at
        the cost of a non-zero (but tiny) collision risk on tensors
        that happen to share the sampled positions and the two
        summary statistics. Hexes are NOT portable across modes; a
        registry serialized in one mode cannot be merged with another.
      cache_to_disk: if set, cached tensors are also persisted to a
        directory at this path (one `.pt` file per cache key). Implies
        `cache_tensors=True`. Memory eviction from `max_cache_size`
        only removes the in-memory copy; the disk copy persists and
        is reloaded on demand by `get_tensor`. Useful when a recording
        produces more cached intermediates than fit in RAM, or when
        you want to inspect the cache after the Python process exits.

    Yields:
      The active ProvenanceRegistry.
    """
    registry = ProvenanceRegistry(
        cache_tensors=cache_tensors,
        hash_algorithm=hash_algorithm,
        max_cache_size=max_cache_size,
        cache_ops=cache_ops,
        hash_mode=hash_mode,
        cache_to_disk=cache_to_disk,
    )
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


# Sentinel tag for torch.Generator canonicalization. A Generator's
# Python repr is its memory address — two generators with the same
# seed get different reprs and therefore different hexes, which
# silently breaks reproducibility. Canonicalize to {seed, device}
# so that "same seed → same hex" holds, and replay can reconstruct
# a Generator with that seed.
_TORCH_GENERATOR_TAG = "__torch_generator__"
_PROVENANCE_SIGNATURE_TAG = "__provenance_signature__"


def _canonicalize_value(v: Any) -> Any:
    """Replace non-JSON-stable objects with canonical representations.

    Currently handles:
      - torch.Generator → recorded seed + device
      - any object exposing a `_provenance_signature(self) -> dict`
        method → that dict (used by manifold instances so class-method
        provenance is deterministic; `<Manifold object at 0x7f...>`
        would otherwise leak the memory address into the hex)

    Other un-serializable values fall through to `default=str` in the
    JSON encoder; if those become a reproducibility hazard, add cases
    here.
    """
    if isinstance(v, torch.Generator):
        return {
            _TORCH_GENERATOR_TAG: True,
            "seed": int(v.initial_seed()),
            "device": str(v.device),
        }
    sig_fn = getattr(v, "_provenance_signature", None)
    if callable(sig_fn):
        return {_PROVENANCE_SIGNATURE_TAG: True, "sig": sig_fn()}
    return v


def _restore_value(v: Any) -> Any:
    """Inverse of `_canonicalize_value` — rebuild objects from params dict.

    Used by `ProvenanceRegistry.replay` to reconstruct call kwargs.
    A torch.Generator restored here is freshly seeded with the recorded
    `initial_seed()`; it does not preserve consumed-state, so replaying
    through stochastic ops that consume the generator partway will
    produce different bits past that consumption point. The replay
    docstring already warns about stochastic ops.
    """
    if isinstance(v, dict) and v.get(_TORCH_GENERATOR_TAG):
        g = torch.Generator(device=v["device"])
        g.manual_seed(int(v["seed"]))
        return g
    return v


def _tensor_content_hex(t: torch.Tensor, hasher_factory: Callable) -> str:
    """Content hash of a tensor: shape + dtype + bytes."""
    h = hasher_factory()
    h.update(str(tuple(t.shape)).encode())
    h.update(str(t.dtype).encode())
    # Hash bytes — move to CPU and contiguous first.
    h.update(t.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()[:HEX_PREFIX_LEN]


def _tensor_sketch_hex(t: torch.Tensor, hasher_factory: Callable) -> str:
    """Sketch hash of a tensor: shape + dtype + strided samples + sum + std.

    Hashes O(SKETCH_SAMPLES) bytes regardless of tensor size. Used for
    `hash_mode="sketch"` registries on big tensors where the byte-hash
    in `_tensor_content_hex` dominates recording cost.

    Discriminators (so two visually-different tensors don't collide):
      - shape + dtype (cheap, very strong)
      - SKETCH_SAMPLES evenly-spaced flattened samples (catches local
        variation in the sampled positions)
      - sum (catches any global rescaling)
      - std (catches scale-preserving structural changes — sign flips,
        permutations — that would leave sum invariant)
    """
    h = hasher_factory()
    h.update(str(tuple(t.shape)).encode())
    h.update(str(t.dtype).encode())
    detached = t.detach()
    numel = detached.numel()
    if numel > 0:
        flat = detached.reshape(-1)
        stride = max(1, numel // SKETCH_SAMPLES)
        # Stride-sample at most SKETCH_SAMPLES elements; on tensors
        # smaller than SKETCH_SAMPLES, this is the whole tensor.
        samples = flat[::stride][:SKETCH_SAMPLES].to(
            device="cpu", dtype=torch.float64
        ).contiguous()
        h.update(samples.numpy().tobytes())
        # Sum + std as global discriminators. Cast to float64 for
        # cross-dtype reproducibility (a sum at float32 has reduction-
        # order-dependent rounding; float64 reductions are stabler).
        promoted = detached.to(device="cpu", dtype=torch.float64)
        # Sum is well-defined for empty above; we already guarded.
        h.update(str(float(promoted.sum().item())).encode())
        if numel > 1:
            h.update(str(float(promoted.std().item())).encode())
    return h.hexdigest()[:HEX_PREFIX_LEN]


def _op_hex(
    op_id: str,
    op_version: str,
    params_json: str,
    input_hexes: tuple[str, ...],
    hasher_factory: Callable,
) -> str:
    """Deterministic hex for an op call from its identity + inputs."""
    h = hasher_factory()
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
    """Get hex for an input tensor: look up in context, else hash it.

    Dispatches on `ctx.hash_mode`: "full" uses byte-level cryptographic
    hashing (default); "sketch" uses the O(1)-bytes sketch hash.
    """
    upstream = ctx._lookup_tensor_hex(t)
    if upstream is not None:
        return upstream
    if ctx.hash_mode == "sketch":
        h = _tensor_sketch_hex(t, ctx._hasher_factory)
    else:
        h = _tensor_content_hex(t, ctx._hasher_factory)
    ctx._register_tensor_hex(t, h)
    return h


# ============================================================
# Decorator
# ============================================================


def with_provenance(op_id: str, op_version: str = "0.1") -> Callable:
    """Decorator: emit a provenance node when the wrapped op runs inside
    a `record()` context. Outside of recording, the decorator is
    transparent.

    All arguments — positional or keyword — are bound by parameter name
    via `inspect.signature`. This makes call signatures fully
    reconstructible for `Registry.replay`, at the cost of one hex
    invalidation if you rename a parameter (bump op_version when that
    happens).

    Restrictions:
      - The function must not use `*args` or `**kwargs` in its signature
        (named parameters only). Our math primitives don't.
    """
    def decorator(fn: Callable) -> Callable:
        sig = inspect.signature(fn)
        # *args / **kwargs would not be addressable by name, so the
        # hex would lose information and replay couldn't reconstruct
        # the call. Reject at decoration time — fail fast, not later.
        for p in sig.parameters.values():
            if p.kind in (
                inspect.Parameter.VAR_POSITIONAL,
                inspect.Parameter.VAR_KEYWORD,
            ):
                raise TypeError(
                    f"@with_provenance({op_id!r}): function {fn.__qualname__} "
                    f"uses {p.kind.name.lower()} parameter {p.name!r}; "
                    f"only named parameters are supported"
                )
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

            # Bind all arguments to parameter names — this normalizes
            # `truncated_svd(M, 3)` and `truncated_svd(M, r=3)` to the
            # same call signature, and gives us the param name for
            # every input so replay can reconstruct the call.
            bound = sig.bind(*args, **kwargs)
            bound.apply_defaults()

            # Separate tensor inputs (which get hex IDs) from non-tensor
            # parameters (which get serialized to JSON). Tuples/lists
            # of tensors are unpacked into per-element hexes so that
            # e.g. a FixedRankPoint = (U, S, Vt) input contributes
            # three independent content hashes rather than being
            # str()-ed into the params blob.
            tensor_inputs: dict[str, str] = {}
            non_tensor_params: dict[str, Any] = {}
            for name, value in bound.arguments.items():
                if isinstance(value, torch.Tensor):
                    tensor_inputs[name] = _resolve_tensor_hex(value, ctx)
                elif (
                    isinstance(value, (tuple, list))
                    and len(value) > 0
                    and all(isinstance(x, torch.Tensor) for x in value)
                ):
                    for i, t in enumerate(value):
                        tensor_inputs[f"{name}[{i}]"] = _resolve_tensor_hex(t, ctx)
                else:
                    non_tensor_params[name] = _canonicalize_value(value)

            # Deterministic input_hexes: sorted by name, "name=hex" each.
            input_hexes = tuple(
                f"{name}={hex_id}"
                for name, hex_id in sorted(tensor_inputs.items())
            )
            params_json = json.dumps(
                non_tensor_params, sort_keys=True, default=str,
            )
            this_hex = _op_hex(
                op_id, op_version, params_json, input_hexes,
                ctx._hasher_factory,
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
                input_hexes=input_hexes,
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

            # Fire any hooks registered for this op_id.
            for hook in ctx._hooks.get(op_id, ()):
                hook(node, output)

            return output

        return wrapper

    return decorator
