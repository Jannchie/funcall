from dataclasses import dataclass

import openai
from openai.types.responses import ResponseFunctionToolCall

from funcall import Context, Funcall


@dataclass
class ContextState:
    user_id: str


# Define the function to be called
def get_user_id(ctx: Context[ContextState]) -> str:
    """Check if the user ID in the context matches a specific value"""
    return ctx.value.user_id


# Use Funcall to manage function
fc = Funcall([get_user_id])

resp = openai.responses.create(
    model="gpt-4.1-nano",
    input="Use function call to check if the user ID in the context.",
    tools=fc.get_tools(),
)

ctx = ContextState(user_id="12345")

for o in resp.output:
    if isinstance(o, ResponseFunctionToolCall):
        result = fc.handle_function_call(o, ctx)  # Call the function
        print(result)
