# Funcall

**Don't repeat yourself!**

Funcall is a Python library that simplifies the use of function calling, especially with OpenAI and Pydantic. Its core goal is to eliminate repetitive schema definitions and boilerplate, letting you focus on your logic instead of writing the same code again and again.

## Motivation

If you use only the OpenAI SDK, enabling function call support requires you to:

- Write your function logic
- Manually define a schema for each function
- Pass the schema through the `tools` parameter
- Parse and handle the function call results yourself

This process is repetitive and error-prone. Funcall automates schema generation and function call handling, so you only need to define your function once.

## Features

- Automatically generate schemas from your function signatures and Pydantic models
- Integrate easily with OpenAI's function calling API
- No more manual, repetitive schema definitions—just define your function once
- **Dynamic tool registration** - Add tools without defining actual functions
- Easy to extend and use in your own projects

## Installation

```bash
pip install funcall
```

## Usage Example

```python
import openai
from openai.types.responses import ResponseFunctionToolCall
from pydantic import BaseModel, Field
from funcall import Funcall

# Define your data model once
class AddForm(BaseModel):
    a: float = Field(description="The first number")
    b: float = Field(description="The second number")

# Define your function once, no need to repeat schema elsewhere
def add(data: AddForm) -> float:
    """Calculate the sum of two numbers"""
    return data.a + data.b

fc = Funcall([add])

resp = openai.responses.create(
    model="gpt-4.1",
    input="Use function call to calculate the sum of 114 and 514",
    tools=fc.get_tools(), # Automatically generates the schema
)

for o in resp.output:
    if isinstance(o, ResponseFunctionToolCall):
        result = fc.handle_function_call(o) # Automatically handles the function call
        print(result) # 628.0
```

Funcall can read the type hints and docstrings to generate the schema automatically:

```json
[
    {
        "type": "function",
        "name": "add",
        "description": "Calculate the sum of two numbers",
        "parameters": {
            "type": "object",
            "properties": {
                "a": {
                    "type": "number",
                    "description": "The first number"
                },
                "b": {
                    "type": "number",
                    "description": "The second number"
                }
            },
            "required": ["a", "b"],
            "additionalProperties": false
        },
        "strict": true
    }
]
```

## Dynamic Tools

Funcall now supports dynamic tool registration, allowing you to add tools without defining actual functions. This provides more flexibility in tool management.

### Adding Dynamic Tools

Use the `add_dynamic_tool` method to register tools directly through metadata:

```python
from funcall import Funcall

funcall = Funcall()

# Add a basic tool
funcall.add_dynamic_tool(
    name="calculator",
    description="Perform basic mathematical operations",
    parameters={
        "operation": {
            "type": "string",
            "description": "The operation to perform",
            "enum": ["add", "subtract", "multiply", "divide"]
        },
        "a": {
            "type": "number", 
            "description": "The first number"
        },
        "b": {
            "type": "number",
            "description": "The second number"
        }
    },
    required=["operation", "a", "b"],
    handler=lambda operation, a, b: {
        "add": a + b,
        "subtract": a - b, 
        "multiply": a * b,
        "divide": a / b if b != 0 else "Cannot divide by zero"
    }[operation]
)
```

### Parameters

- `name`: Tool name
- `description`: Tool description  
- `parameters`: Parameter definitions (JSON Schema format)
- `required`: List of required parameter names (optional)
- `handler`: Custom handler function (optional)

### Handler Options

#### With Custom Handler

```python
def custom_handler(city: str, units: str = "celsius") -> dict:
    return {"city": city, "temperature": "25°C", "units": units}

funcall.add_dynamic_tool(
    name="get_weather",
    description="Get weather information",
    parameters={
        "city": {"type": "string", "description": "City name"},
        "units": {"type": "string", "description": "Temperature units", "default": "celsius"}
    },
    required=["city"],
    handler=custom_handler
)
```

#### Without Handler (Default Behavior)

If no handler is provided, the tool returns call information:

```python
funcall.add_dynamic_tool(
    name="simple_tool",
    description="A simple tool",
    parameters={
        "input": {"type": "string", "description": "Input parameter"}
    },
    required=["input"]
)

# When called, returns:
# {
#     "tool": "simple_tool",
#     "arguments": {"input": "value"},
#     "message": "Tool 'simple_tool' called with arguments: {'input': 'value'}"
# }
```

### Removing Dynamic Tools

```python
funcall.remove_dynamic_tool("tool_name")
```

### Integration with Existing Features

Dynamic tools are fully integrated with existing functionality:

- `get_tools()` - Includes dynamic tool definitions
- `call_function()` / `call_function_async()` - Call dynamic tools
- `handle_function_call()` - Handle LLM tool calls
- `get_tool_meta()` - Get tool metadata

### Use Cases

1. **Rapid Prototyping** - Test tools without defining functions
2. **Configuration-Driven Tools** - Create tools dynamically from config files
3. **API Proxy Tools** - Create tools that call external APIs
4. **Mocking and Testing** - Create mock tools for testing

See the `examples/` directory for more usage examples.

## License

MIT
