"""
Tests for function management APIs.
"""

import pytest

from funcall import Funcall


def sample_function(x: int, y: int) -> int:
    """Add two numbers."""
    return x + y


def another_function(message: str) -> str:
    """Return a greeting message."""
    return f"Hello, {message}!"


async def async_function(value: str) -> str:
    """An async function."""
    return f"Async: {value}"


def test_add_function():
    """Test adding a function after initialization."""
    funcall = Funcall()

    # Initially empty
    assert len(funcall.list_functions()) == 0

    # Add a function
    funcall.add_function(sample_function)

    # Check it was added
    assert "sample_function" in funcall.list_functions()
    assert funcall.has_function("sample_function")
    assert "sample_function" in funcall.list_regular_functions()
    assert "sample_function" not in funcall.list_dynamic_tools()

    # Test calling the function
    result = funcall.call_function("sample_function", '{"x": 5, "y": 3}')
    assert result == 8


def test_add_function_duplicate():
    """Test adding a function with duplicate name raises error."""
    funcall = Funcall([sample_function])

    # Try to add the same function again
    with pytest.raises(ValueError, match="Function 'sample_function' already exists"):
        funcall.add_function(sample_function)


def test_remove_function():
    """Test removing a function by name."""
    funcall = Funcall([sample_function, another_function])

    # Initially has both functions
    assert len(funcall.list_functions()) == 2
    assert "sample_function" in funcall.list_functions()
    assert "another_function" in funcall.list_functions()

    # Remove one function
    funcall.remove_function("sample_function")

    # Check it was removed
    assert len(funcall.list_functions()) == 1
    assert "sample_function" not in funcall.list_functions()
    assert "another_function" in funcall.list_functions()

    # Test that the removed function can't be called
    with pytest.raises(ValueError, match="Function sample_function not found"):
        funcall.call_function("sample_function", '{"x": 1, "y": 2}')


def test_remove_function_not_found():
    """Test removing a non-existent function raises error."""
    funcall = Funcall()

    with pytest.raises(ValueError, match="Function 'nonexistent' not found"):
        funcall.remove_function("nonexistent")


def test_remove_function_by_callable():
    """Test removing a function by its callable reference."""
    funcall = Funcall([sample_function, another_function])

    # Remove by callable
    funcall.remove_function_by_callable(sample_function)

    # Check it was removed
    assert "sample_function" not in funcall.list_functions()
    assert "another_function" in funcall.list_functions()


def test_remove_function_by_callable_not_found():
    """Test removing a function by callable that's not registered."""
    funcall = Funcall()

    with pytest.raises(ValueError, match="Function 'sample_function' not found"):
        funcall.remove_function_by_callable(sample_function)


def test_remove_function_by_callable_different_object():
    """Test removing a function with same name but different object."""

    # Create a function with the same name as sample_function
    def sample_function(a: int, b: int) -> int:
        return a * b

    funcall = Funcall()

    # Add the original sample_function from module level
    original_sample_function = globals()["sample_function"]  # Get the original function
    funcall.function_registry["sample_function"] = original_sample_function
    funcall.functions.append(original_sample_function)

    # Try to remove by the local callable (different object)
    with pytest.raises(ValueError, match="Function 'sample_function' found but is not the same object"):
        funcall.remove_function_by_callable(sample_function)


def test_mixed_function_and_dynamic_tool_management():
    """Test managing both regular functions and dynamic tools."""
    funcall = Funcall([sample_function])

    # Add a dynamic tool
    funcall.add_dynamic_tool(
        name="dynamic_test",
        description="A test dynamic tool",
        parameters={"input": {"type": "string", "description": "Input"}},
        required=["input"],
    )

    # Add another function
    funcall.add_function(another_function)

    # Check lists
    all_functions = funcall.list_functions()
    regular_functions = funcall.list_regular_functions()
    dynamic_tools = funcall.list_dynamic_tools()

    assert len(all_functions) == 3
    assert set(all_functions) == {"sample_function", "dynamic_test", "another_function"}
    assert set(regular_functions) == {"sample_function", "another_function"}
    assert set(dynamic_tools) == {"dynamic_test"}


def test_cannot_remove_dynamic_tool_with_remove_function():
    """Test that dynamic tools can't be removed with remove_function."""
    funcall = Funcall()

    # Add a dynamic tool
    funcall.add_dynamic_tool(
        name="dynamic_test",
        description="A test dynamic tool",
        parameters={"input": {"type": "string", "description": "Input"}},
        required=["input"],
    )

    # Try to remove it with remove_function
    with pytest.raises(ValueError, match="'dynamic_test' is a dynamic tool. Use remove_dynamic_tool"):
        funcall.remove_function("dynamic_test")


def test_get_tools_includes_added_functions():
    """Test that get_tools includes dynamically added functions."""
    funcall = Funcall()

    # Add a function
    funcall.add_function(sample_function)

    # Get tools
    tools = funcall.get_tools()

    # Should include the added function
    assert len(tools) == 1
    tool = tools[0]
    assert tool["name"] == "sample_function"  # type: ignore
    assert tool["description"] == "Add two numbers."  # type: ignore


def test_async_function_handling():
    """Test adding and calling async functions."""
    funcall = Funcall()

    # Add async function
    funcall.add_function(async_function)

    # Test sync call (should work with warning)
    result = funcall.call_function("async_function", '{"value": "test"}')
    assert result == "Async: test"


if __name__ == "__main__":
    pytest.main([__file__])
