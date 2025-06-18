from dataclasses import dataclass, field

import openai
from openai.types.responses import ResponseFunctionToolCall

from funcall import Funcall


# Use dataclasses to define the schema
@dataclass
class AddForm:
    a: float = field(metadata={"description": "The first number"})
    b: float = field(metadata={"description": "The second number"})


# Define the function to be called
def add(data: AddForm) -> float:
    """Calculate the sum of two numbers"""
    return data.a + data.b


# Use Funcall to manage function
fc = Funcall([add])

resp = openai.responses.create(
    model="gpt-4.1",
    input="Use function call to calculate the sum of 114 and 514",
    tools=fc.get_tools(),  # Get the function metadata
)

for o in resp.output:
    if isinstance(o, ResponseFunctionToolCall):
        result = fc.handle_function_call(o)  # Call the function
        print(result)
