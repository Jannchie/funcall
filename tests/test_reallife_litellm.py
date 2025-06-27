import litellm
import pytest
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
    tools = fc.get_tools(target="completion")
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


def test_litellm_funcall_sum_simple():
    def add(a: int, b: int) -> int:
        """Calculate the sum of two numbers"""
        return a + b

    fc = Funcall([add])
    tools = fc.get_tools(target="completion")
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


@pytest.mark.asyncio
async def test_litellm_funcall_async():
    fc = Funcall([add])
    tools = fc.get_tools(target="completion")
    resp = await litellm.acompletion(
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


@pytest.mark.asyncio
async def test_litellm_funcall_with_async_tool():
    async def async_add(data: AddForm) -> float:
        """Calculate the sum of two numbers asynchronously"""
        return data.a + data.b

    fc = Funcall([async_add])
    tools = fc.get_tools(target="completion")
    resp = await litellm.acompletion(
        model="gpt-4.1",
        messages=[{"role": "user", "content": "Use function call to calculate the sum of 114 and 514"}],
        tools=tools,
        tool_choice={"type": "function", "function": {"name": "async_add"}},  # type: ignore
    )
    results = []
    choice = resp.choices[0]
    for tool_call in choice.message.tool_calls:
        if isinstance(tool_call, litellm.ChatCompletionMessageToolCall):
            result = await fc.handle_function_call_async(tool_call)
            results.append(result)
    assert 628 in results
