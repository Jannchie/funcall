"""
Example: Function Management APIs

This example demonstrates how to use the Funcall class's function management APIs
to dynamically add and remove function tools after initialization.
"""

import json

from funcall import Funcall


def add_numbers(a: int, b: int) -> int:
    """Add two numbers together."""
    return a + b


def multiply_numbers(x: float, y: float) -> float:
    """Multiply two numbers."""
    return x * y


def greet_user(name: str, greeting: str = "Hello") -> str:
    """Greet a user with a custom greeting."""
    return f"{greeting}, {name}!"


async def async_operation(data: str) -> str:
    """An async operation that processes data."""
    return f"Processed: {data}"


def demonstrate_initial_state(funcall: Funcall):
    """Demonstrate the initial state of Funcall."""
    print("=== Initial State ===")
    print(f"Functions: {funcall.list_functions()}")
    print(f"Regular functions: {funcall.list_regular_functions()}")
    print(f"Dynamic tools: {funcall.list_dynamic_tools()}")
    print()


def demonstrate_adding_functions(funcall: Funcall):
    """Demonstrate adding functions and dynamic tools."""
    print("=== Adding Functions ===")
    funcall.add_function(add_numbers)
    funcall.add_function(multiply_numbers)
    print(f"Added functions. Total: {funcall.list_functions()}")

    # Add a dynamic tool
    funcall.add_dynamic_tool(
        name="subtract",
        description="Subtract two numbers",
        parameters={
            "a": {"type": "number", "description": "First number"},
            "b": {"type": "number", "description": "Second number"},
        },
        required=["a", "b"],
        handler=lambda a, b: a - b,
    )
    print(f"Added dynamic tool. All tools: {funcall.list_functions()}")
    print(f"Regular functions: {funcall.list_regular_functions()}")
    print(f"Dynamic tools: {funcall.list_dynamic_tools()}")
    print()


def demonstrate_function_calls(funcall: Funcall):
    """Demonstrate calling various functions."""
    print("=== Testing Function Calls ===")

    # Test regular function
    result = funcall.call_function("add_numbers", json.dumps({"a": 10, "b": 5}))
    print(f"add_numbers(10, 5) = {result}")

    result = funcall.call_function("multiply_numbers", json.dumps({"x": 3.5, "y": 2.0}))
    print(f"multiply_numbers(3.5, 2.0) = {result}")

    # Test dynamic tool
    result = funcall.call_function("subtract", json.dumps({"a": 15, "b": 7}))
    print(f"subtract(15, 7) = {result}")
    print()


def demonstrate_more_functions(funcall: Funcall):
    """Demonstrate adding and testing more functions."""
    print("=== Adding More Functions ===")
    funcall.add_function(greet_user)
    funcall.add_function(async_operation)

    print(f"All functions: {funcall.list_functions()}")

    # Test the new functions
    result = funcall.call_function("greet_user", json.dumps({"name": "Alice"}))
    print(f"greet_user(name='Alice') = {result}")

    result = funcall.call_function("greet_user", json.dumps({"name": "Bob", "greeting": "Hi"}))
    print(f"greet_user(name='Bob', greeting='Hi') = {result}")

    # Test async function (called synchronously)
    result = funcall.call_function("async_operation", json.dumps({"data": "test data"}))
    print(f"async_operation(data='test data') = {result}")
    print()


def demonstrate_tool_definitions(funcall: Funcall):
    """Show tool definitions."""
    print("=== Tool Definitions ===")
    tools = funcall.get_tools()
    for tool in tools:
        print(f"Tool: {tool['name']} - {tool['description']}")  # type: ignore
    print()


def demonstrate_removing_functions(funcall: Funcall):
    """Demonstrate removing functions."""
    print("=== Removing Functions ===")
    funcall.remove_function("multiply_numbers")
    print(f"Removed multiply_numbers. Remaining: {funcall.list_functions()}")

    # Remove by callable reference
    funcall.remove_function_by_callable(greet_user)
    print(f"Removed greet_user by callable. Remaining: {funcall.list_functions()}")

    # Remove dynamic tool
    funcall.remove_dynamic_tool("subtract")
    print(f"Removed dynamic tool. Remaining: {funcall.list_functions()}")
    print(f"Regular functions: {funcall.list_regular_functions()}")
    print(f"Dynamic tools: {funcall.list_dynamic_tools()}")
    print()


def demonstrate_error_handling(funcall: Funcall):
    """Demonstrate error handling."""
    print("=== Error Handling ===")

    # Try to call removed function
    try:
        funcall.call_function("multiply_numbers", json.dumps({"x": 1, "y": 2}))
    except ValueError as e:
        print(f"Expected error: {e}")

    # Try to add duplicate function
    try:
        funcall.add_function(add_numbers)  # Already exists
    except ValueError as e:
        print(f"Expected error: {e}")

    # Try to remove non-existent function
    try:
        funcall.remove_function("nonexistent")
    except ValueError as e:
        print(f"Expected error: {e}")
    print()


def demonstrate_final_state(funcall: Funcall):
    """Show final state."""
    print("=== Final State ===")
    print(f"Functions: {funcall.list_functions()}")
    print(f"Regular functions: {funcall.list_regular_functions()}")
    print(f"Dynamic tools: {funcall.list_dynamic_tools()}")


def main():
    # Start with an empty Funcall instance
    funcall = Funcall()

    demonstrate_initial_state(funcall)
    demonstrate_adding_functions(funcall)
    demonstrate_function_calls(funcall)
    demonstrate_more_functions(funcall)
    demonstrate_tool_definitions(funcall)
    demonstrate_removing_functions(funcall)
    demonstrate_error_handling(funcall)
    demonstrate_final_state(funcall)


if __name__ == "__main__":
    main()
