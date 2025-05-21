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

See the `examples/` directory for more usage examples.

## License

MIT
