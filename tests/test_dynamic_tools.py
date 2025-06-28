"""
Tests for dynamic tool functionality.
"""

import json
from typing import Any, cast

import pytest

from funcall import Funcall


def test_add_dynamic_tool():
    """Test adding a dynamic tool."""
    funcall = Funcall()

    # Add a simple dynamic tool
    funcall.add_dynamic_tool(
        name="test_tool",
        description="A test tool",
        parameters={
            "param1": {
                "type": "string",
                "description": "First parameter",
            },
            "param2": {
                "type": "number",
                "description": "Second parameter",
            },
        },
        required=["param1"],
    )

    # Check that the tool was added
    assert "test_tool" in funcall.function_registry
    assert "test_tool" in funcall.dynamic_tools


def test_dynamic_tool_with_handler():
    """Test dynamic tool with custom handler."""
    funcall = Funcall()

    def custom_handler(name: str, value: int) -> dict[str, Any]:
        return {"greeting": f"Hello {name}", "doubled": value * 2}

    funcall.add_dynamic_tool(
        name="greeting_tool",
        description="A greeting tool",
        parameters={
            "name": {
                "type": "string",
                "description": "Name to greet",
            },
            "value": {
                "type": "number",
                "description": "Value to double",
            },
        },
        required=["name", "value"],
        handler=custom_handler,
    )

    # Test calling the tool
    result = funcall.call_function(
        "greeting_tool",
        json.dumps({"name": "Alice", "value": 5}),
    )

    expected = {"greeting": "Hello Alice", "doubled": 10}
    assert result == expected


def test_dynamic_tool_without_handler():
    """Test dynamic tool without custom handler (default behavior)."""
    funcall = Funcall()

    funcall.add_dynamic_tool(
        name="default_tool",
        description="A tool with default behavior",
        parameters={
            "input": {
                "type": "string",
                "description": "Input parameter",
            },
        },
        required=["input"],
    )

    # Test calling the tool
    result = funcall.call_function(
        "default_tool",
        json.dumps({"input": "test"}),
    )

    result_dict = cast("dict[str, Any]", result)
    assert result_dict["tool"] == "default_tool"
    assert result_dict["arguments"]["input"] == "test"
    assert "message" in result_dict


def test_get_tools_with_dynamic_tools():
    """Test get_tools includes dynamic tools."""
    funcall = Funcall()

    funcall.add_dynamic_tool(
        name="dynamic_test",
        description="Dynamic test tool",
        parameters={
            "param": {
                "type": "string",
                "description": "Test parameter",
            },
        },
        required=["param"],
    )

    # Test response format
    tools = funcall.get_tools(target="response")
    response_tools = [cast("dict[str, Any]", t) for t in tools]
    dynamic_tool = next((t for t in response_tools if t.get("name") == "dynamic_test"), None)
    assert dynamic_tool is not None
    assert dynamic_tool["description"] == "Dynamic test tool"
    assert dynamic_tool["parameters"]["properties"]["param"]["type"] == "string"

    # Test completion format
    tools = funcall.get_tools(target="completion")
    completion_tools = [cast("dict[str, Any]", t) for t in tools]
    dynamic_tool = next((t for t in completion_tools if t.get("function", {}).get("name") == "dynamic_test"), None)
    assert dynamic_tool is not None
    assert dynamic_tool["function"]["description"] == "Dynamic test tool"


def test_remove_dynamic_tool():
    """Test removing a dynamic tool."""
    funcall = Funcall()

    funcall.add_dynamic_tool(
        name="temp_tool",
        description="Temporary tool",
        parameters={
            "param": {
                "type": "string",
                "description": "Test parameter",
            },
        },
        required=["param"],
    )

    # Verify tool exists
    assert "temp_tool" in funcall.function_registry
    assert "temp_tool" in funcall.dynamic_tools

    # Remove the tool
    funcall.remove_dynamic_tool("temp_tool")

    # Verify tool is removed
    assert "temp_tool" not in funcall.function_registry
    assert "temp_tool" not in funcall.dynamic_tools

    # Verify calling the removed tool raises an error
    with pytest.raises(ValueError, match="Function temp_tool not found"):
        funcall.call_function("temp_tool", json.dumps({"param": "test"}))


def test_get_tool_meta_for_dynamic_tool():
    """Test get_tool_meta for dynamic tools."""
    funcall = Funcall()

    funcall.add_dynamic_tool(
        name="meta_test",
        description="Test tool metadata",
        parameters={
            "param": {
                "type": "string",
                "description": "Test parameter",
            },
        },
        required=["param"],
    )

    meta = funcall.get_tool_meta("meta_test")
    assert meta["require_confirm"] is False
    assert meta["return_direct"] is False


if __name__ == "__main__":
    pytest.main([__file__])
