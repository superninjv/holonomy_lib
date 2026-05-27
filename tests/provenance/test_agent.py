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
