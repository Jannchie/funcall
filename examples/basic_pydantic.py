import openai
from openai.types.responses import ResponseFunctionToolCall
from pydantic import BaseModel, Field

from funcall import Funcall


# Use Pydantic to define the schema
class AddForm(BaseModel):
    c: float = Field(description="The first number")


# Define the function to be called
def add(form: AddForm, b: float) -> float:
    """Calculate the sum of two numbers"""
    return form.c + b


# Use Funcall to manage function
fc = Funcall([add])
print(fc.get_tools())
resp = openai.responses.create(
    model="gpt-4.1",
    input="Use function call to calculate the sum of 114 and 514",
    tools=fc.get_tools(),  # Get the function metadata
)

for o in resp.output:
    if isinstance(o, ResponseFunctionToolCall):
        result = fc.handle_function_call(o)  # Call the function
        print(result)
