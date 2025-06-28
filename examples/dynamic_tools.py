"""
Example: Using Dynamic Tools

This example demonstrates how to use the Funcall class's add_dynamic_tool method
to add tools without defining actual functions.
"""

import json

from funcall import Funcall


def main():
    # Create Funcall instance
    funcall = Funcall()

    # Add a calculator tool
    funcall.add_dynamic_tool(
        name="calculator",
        description="Perform basic mathematical operations",
        parameters={
            "operation": {
                "type": "string",
                "description": "The operation to perform",
                "enum": ["add", "subtract", "multiply", "divide"],
            },
            "a": {
                "type": "number",
                "description": "The first number",
            },
            "b": {
                "type": "number",
                "description": "The second number",
            },
        },
        required=["operation", "a", "b"],
        handler=lambda operation, a, b: {
            "add": a + b,
            "subtract": a - b,
            "multiply": a * b,
            "divide": a / b if b != 0 else "Cannot divide by zero",
        }[operation],
    )

    # Add a weather query tool (without custom handler)
    funcall.add_dynamic_tool(
        name="get_weather",
        description="Get weather information for a specified city",
        parameters={
            "city": {
                "type": "string",
                "description": "City name",
            },
            "units": {
                "type": "string",
                "description": "Temperature units",
                "enum": ["celsius", "fahrenheit"],
                "default": "celsius",
            },
        },
        required=["city"],
    )

    # Add a file operations tool
    funcall.add_dynamic_tool(
        name="file_operations",
        description="Perform file operations",
        parameters={
            "action": {
                "type": "string",
                "description": "The operation to perform",
                "enum": ["create", "read", "delete", "list"],
            },
            "path": {
                "type": "string",
                "description": "File path",
            },
            "content": {
                "type": "string",
                "description": "File content",
            },
        },
        required=["action", "path"],
        handler=lambda action, path, content=None: {
            "action": action,
            "path": path,
            "content": content,
            "result": f"Simulated {action} operation on {path}",
        },
    )

    # Get tool definitions
    print("=== OpenAI Response Format Tool Definitions ===")
    tools = funcall.get_tools(target="response")
    for tool in tools:
        print(json.dumps(tool, indent=2, ensure_ascii=False))
        print()

    print("=== LiteLLM Completion Format Tool Definitions ===")
    tools = funcall.get_tools(target="completion")
    for tool in tools:
        print(json.dumps(tool, indent=2, ensure_ascii=False))
        print()

    # Test tool calls
    print("=== Testing Tool Calls ===")

    # Test calculator
    result = funcall.call_function(
        "calculator",
        json.dumps({"operation": "add", "a": 10, "b": 5}),
    )
    print(f"Calculator result: {result}")

    # Test weather query (without custom handler)
    result = funcall.call_function(
        "get_weather",
        json.dumps({"city": "Beijing", "units": "celsius"}),
    )
    print(f"Weather query result: {result}")

    # Test file operations
    result = funcall.call_function(
        "file_operations",
        json.dumps({"action": "create", "path": "/test.txt", "content": "Hello World"}),
    )
    print(f"File operation result: {result}")

    # Demonstrate how to remove dynamic tools
    print("\n=== Removing Dynamic Tools ===")
    funcall.remove_dynamic_tool("calculator")
    print("Calculator tool removed")

    try:
        funcall.call_function("calculator", json.dumps({"operation": "add", "a": 1, "b": 1}))
    except ValueError as e:
        print(f"Expected error: {e}")


if __name__ == "__main__":
    main()
