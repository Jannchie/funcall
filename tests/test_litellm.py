import litellm
from pydantic import BaseModel, Field

from funcall import Funcall


class AddForm(BaseModel):
    a: float = Field(description="The first number")
    b: float = Field(description="The second number")


def add(data: AddForm) -> float:
    """Calculate the sum of two numbers"""
    return data.a + data.b


def test_litellm_funcall_sum():
    fc = Funcall([add])
    tools = fc.get_tools(target="litellm")
    resp = litellm.completion(
        model="gpt-4.1",
        messages=[{"role": "user", "content": "Use function call to calculate the sum of 114 and 514"}],
        tools=tools,
    )
    results = []
    choice = resp.choices[0]
    for tool_call in choice.message.tool_calls:
        if isinstance(tool_call, litellm.ChatCompletionMessageToolCall):
            result = fc.handle_function_call(tool_call)
            results.append(result)
    assert 628 in results
