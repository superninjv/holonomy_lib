"""Agent API: Python functions an LLM can call as native tools.

The pattern:
  1. Define a Python function for each inspection / intervention an
     agent might want to perform against a ProvenanceRegistry.
  2. Decorate with `@agent_tool` to register it.
  3. Call `to_anthropic_schema()` or `to_openai_schema()` to get the
     JSON-schema list the provider APIs expect.

The same registered functions are usable three ways:
  - Native LLM tool-use (Anthropic `tools=[...]`, OpenAI `functions=[...]`).
  - MCP transport (the `holonomy_lib.provenance.mcp` module wraps them).
  - Direct Python calls (just import and invoke).

Designing the schemas this way means the underlying tool surface is
provider-agnostic; if the MCP ecosystem fades or a new provider
appears, the same `@agent_tool` functions still work — only the
transport-specific schema dumper changes.

Most tools take an explicit `registry: ProvenanceRegistry` so callers
can bind the registry however they like (closure, global, dependency
injection). Transport adapters typically pre-bind the registry and
expose the tools with that argument stripped.
"""

from __future__ import annotations

import inspect
import typing
from dataclasses import dataclass
from typing import Any, Callable, Optional, Union, get_args, get_origin

import torch

from holonomy_lib.provenance.protocol import (
    OP_REGISTRY,
    ProvenanceRegistry,
)


# ============================================================
# Registry of decorated tools
# ============================================================


@dataclass
class ToolSpec:
    """Metadata about an @agent_tool-decorated function.

    Captured at decoration time so provider-specific schema dumpers can
    consume it without re-inspecting the function each time.

    `type_hints` carries the resolved type hints (after `__future__
    annotations` string un-stringification); use it for schema work
    rather than the raw `signature.parameters[*].annotation` so future-
    annotation modules work transparently.
    """
    name: str
    fn: Callable[..., Any]
    description: str
    signature: inspect.Signature
    docstring: str
    type_hints: dict[str, Any]


_AGENT_TOOLS: dict[str, ToolSpec] = {}


def agent_tool(
    *,
    description: str = "",
    name: Optional[str] = None,
) -> Callable[[Callable], Callable]:
    """Decorator: register a function as an agent-callable tool.

    Args:
      description: short (1-2 sentence) human-readable description that
        gets fed to the LLM as the tool's purpose. Defaults to the
        first line of the function's docstring.
      name: registry key. Defaults to the function's `__name__`.

    The function must have type-annotated parameters. Each parameter
    becomes a JSON-schema field in the emitted tool definition; the
    function's docstring becomes the schema's `description`.
    """
    def decorator(fn: Callable) -> Callable:
        tool_name = name if name is not None else fn.__name__
        doc = (fn.__doc__ or "").strip()
        desc = description.strip() or (doc.split("\n", 1)[0] if doc else "")
        try:
            # Resolve `from __future__ import annotations` strings.
            # include_extras=True keeps Annotated[] markers if we
            # ever add them.
            hints = typing.get_type_hints(fn, include_extras=True)
        except Exception:
            # Bad annotation references would normally fail loudly;
            # fall back to raw signature annotations so a buggy hint
            # doesn't block registration of an otherwise-valid tool.
            hints = {}
        spec = ToolSpec(
            name=tool_name,
            fn=fn,
            description=desc,
            signature=inspect.signature(fn),
            docstring=doc,
            type_hints=hints,
        )
        _AGENT_TOOLS[tool_name] = spec
        return fn
    return decorator


def list_tools() -> list[ToolSpec]:
    """Return all registered tools in insertion order."""
    return list(_AGENT_TOOLS.values())


def get_tool(name: str) -> ToolSpec:
    """Look up a tool by name. Raises KeyError if unknown."""
    return _AGENT_TOOLS[name]


# ============================================================
# Type → JSON-schema mapping
# ============================================================


def _python_type_to_json_schema(annotation: Any) -> dict[str, Any]:
    """Convert a Python type annotation to a JSON-schema fragment.

    Handles the common types our tool signatures use: str, int, float,
    bool, list[T], dict, Optional[T], Union[T, None]. Falls back to
    `{"type": "string"}` for anything unrecognized (the LLM will see
    a string and the tool will validate at call time).
    """
    if annotation is inspect.Parameter.empty:
        return {"type": "string"}
    if annotation is str:
        return {"type": "string"}
    if annotation is int:
        return {"type": "integer"}
    if annotation is float:
        return {"type": "number"}
    if annotation is bool:
        return {"type": "boolean"}
    origin = get_origin(annotation)
    if origin is list:
        args = get_args(annotation)
        item_schema = (
            _python_type_to_json_schema(args[0]) if args else {"type": "string"}
        )
        return {"type": "array", "items": item_schema}
    if origin is dict:
        return {"type": "object"}
    if origin is Union:
        args = [a for a in get_args(annotation) if a is not type(None)]
        if len(args) == 1:
            schema = _python_type_to_json_schema(args[0])
            return schema
        # Multi-type unions get widened to string for now; tool surface
        # should avoid these.
        return {"type": "string"}
    return {"type": "string"}


def _tool_to_json_schema_params(spec: ToolSpec) -> dict[str, Any]:
    """Build the JSON-schema object for a tool's parameters.

    Skips the `registry: ProvenanceRegistry` parameter when present;
    transport adapters pre-bind it, so the LLM-facing schema must not
    include it.
    """
    properties: dict[str, Any] = {}
    required: list[str] = []
    for name, param in spec.signature.parameters.items():
        if name == "registry":
            continue
        # Skip parameters that the LLM has no business setting (the
        # `registry` arg is the canonical example; in the future we
        # might add a marker for others).
        resolved = spec.type_hints.get(name, param.annotation)
        if resolved is ProvenanceRegistry:
            continue
        properties[name] = _python_type_to_json_schema(resolved)
        if param.default is inspect.Parameter.empty:
            required.append(name)
    schema: dict[str, Any] = {"type": "object", "properties": properties}
    if required:
        schema["required"] = required
    return schema


# ============================================================
# Provider-specific schema dumpers
# ============================================================


def to_anthropic_schema() -> list[dict[str, Any]]:
    """Emit the registered tools as an Anthropic-API `tools=[...]` list.

    Each entry has the shape:
      {"name": str, "description": str, "input_schema": {...}}
    """
    out: list[dict[str, Any]] = []
    for spec in _AGENT_TOOLS.values():
        out.append({
            "name": spec.name,
            "description": spec.description,
            "input_schema": _tool_to_json_schema_params(spec),
        })
    return out


def to_openai_schema() -> list[dict[str, Any]]:
    """Emit the registered tools as an OpenAI-API `tools=[...]` list.

    Each entry has the shape:
      {"type": "function", "function": {"name": str, "description": str,
                                          "parameters": {...}}}
    """
    out: list[dict[str, Any]] = []
    for spec in _AGENT_TOOLS.values():
        out.append({
            "type": "function",
            "function": {
                "name": spec.name,
                "description": spec.description,
                "parameters": _tool_to_json_schema_params(spec),
            },
        })
    return out
