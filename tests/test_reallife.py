from dataclasses import dataclass, field
from typing import Optional, Union

import openai
import pytest
from openai.types.responses import ResponseFunctionToolCall
from pydantic import BaseModel, Field

from funcall import Context, Funcall


def test_openai_funcall_sum():
    def add(a: float, b: float) -> float:
        """Calculate the sum of two numbers"""
        return a + b

    fc = Funcall([add])
    resp = openai.responses.create(
        model="gpt-4.1-nano",
        input="Use function call to calculate the sum of 114 and 514",
        tools=fc.get_tools(),
    )
    results = []
    for o in resp.output:
        if isinstance(o, ResponseFunctionToolCall):
            result = fc.handle_function_call(o)
            results.append(result)
    assert 628 in results


@pytest.mark.asyncio
async def test_openai_funcall_sum_tool_choice():
    async def add(a: float, b: float) -> float:
        """Calculate the sum of two numbers"""
        return a + b

    fc = Funcall([add])
    resp = openai.responses.create(
        model="gpt-4.1-nano",
        input="Use function call to calculate the sum of 114 and 514",
        tools=fc.get_tools(),
        tool_choice={"type": "function", "name": "add"},
    )
    results = []
    for o in resp.output:
        if isinstance(o, ResponseFunctionToolCall):
            result = await fc.handle_function_call_async(o)
            results.append(result)
    assert 628 in results


# Pydantic 参数测试
class AddForm(BaseModel):
    c: float = Field(description="The first number")


def test_openai_funcall_pydantic():
    def add(form: AddForm, b: float) -> float:
        return form.c + b

    fc = Funcall([add])
    resp = openai.responses.create(
        model="gpt-4.1-nano",
        input="Use function call to calculate the sum of 100 and 200",
        tools=fc.get_tools(),
    )
    results = []
    for o in resp.output:
        if isinstance(o, ResponseFunctionToolCall):
            result = fc.handle_function_call(o)
            results.append(result)
    assert 300 in results


# dataclass 参数测试
@dataclass
class AddData:
    a: float = field(metadata={"description": "The first number"})
    b: float = field(metadata={"description": "The second number"})


def test_openai_funcall_dataclass():
    def add(data: AddData) -> float:
        return data.a + data.b

    fc = Funcall([add])
    resp = openai.responses.create(
        model="gpt-4.1-nano",
        input="Use function call to calculate the sum of 1 and 2",
        tools=fc.get_tools(),
    )
    results = []
    for o in resp.output:
        if isinstance(o, ResponseFunctionToolCall):
            result = fc.handle_function_call(o)
            results.append(result)
    assert 3 in results


# list 参数测试
class Item(BaseModel):
    name: str = Field(description="Item name")
    value: int = Field(description="Item value")


def test_openai_funcall_list():
    def sum_values(items: list[Item]) -> int:
        return sum(item.value for item in items)

    fc = Funcall([sum_values])
    resp = openai.responses.create(
        model="gpt-4.1-nano",
        input="Sum the values of items: apple=1, banana=2, cherry=3",
        tools=fc.get_tools(),
    )
    results = []
    for o in resp.output:
        if isinstance(o, ResponseFunctionToolCall):
            result = fc.handle_function_call(o)
            results.append(result)
    assert 6 in results


def test_openai_funcall_nested_list():
    class Item(BaseModel):
        name: str
        value: int

    def sum_nested_values(items: list[list[Item]]) -> int:
        return sum(item.value for sublist in items for item in sublist)

    fc = Funcall([sum_nested_values])
    resp = openai.responses.create(
        model="gpt-4.1-nano",
        input="Sum the values of items: [[apple=1, banana=2], [cherry=3]], You should only call the function once.",
        tools=fc.get_tools(),
    )
    results = []
    for o in resp.output:
        if isinstance(o, ResponseFunctionToolCall):
            result = fc.handle_function_call(o)
            results.append(result)
    assert 6 in results


# context 参数测试
@dataclass
class ContextState:
    user_id: int


def test_openai_funcall_context():
    def get_user_id(ctx: Context[ContextState]) -> int:
        return ctx.value.user_id

    fc = Funcall([get_user_id])
    resp = openai.responses.create(
        model="gpt-4.1-nano",
        input="Use function call to check if the user ID in the context.",
        tools=fc.get_tools(),
    )
    ctx = Context(ContextState(user_id=12345))
    results = []
    for o in resp.output:
        if isinstance(o, ResponseFunctionToolCall):
            result = fc.handle_function_call(o, ctx)
            results.append(result)
    assert 12345 in results


def test_openai_funcall_union():
    def parse_number(n: Union[int, str]) -> int:  # noqa: UP007
        """If n is int, return as is; if str, try to parse as int."""
        if isinstance(n, int):
            return n
        return int(n)

    fc = Funcall([parse_number])
    resp = openai.responses.create(
        model="gpt-4.1-nano",
        input="Use function call to parse the number '42'",
        tools=fc.get_tools(),
    )
    results = []
    for o in resp.output:
        if isinstance(o, ResponseFunctionToolCall):
            result = fc.handle_function_call(o)
            results.append(result)
    assert 42 in results


def test_openai_funcall_optional():
    def greet(name: Optional[str] = None) -> str:  # noqa: UP045
        if name:
            return f"Hello, {name}!"
        return "Hello, guest!"

    fc = Funcall([greet])
    resp = openai.responses.create(
        model="gpt-4.1-nano",
        input="Use function call to greet without a name",
        tools=fc.get_tools(),
    )
    results = []
    for o in resp.output:
        if isinstance(o, ResponseFunctionToolCall):
            result = fc.handle_function_call(o)
            results.append(result)
    assert "guest" in results[0]


def test_openai_funcall_optional_2():
    def greet(name: str | None = None) -> str:
        if name:
            return f"Hello, {name}!"
        return "Hello, guest!"

    fc = Funcall([greet])
    resp = openai.responses.create(
        model="gpt-4.1-nano",
        input="Use function call to greet without a name",
        tools=fc.get_tools(),
    )
    results = []
    for o in resp.output:
        if isinstance(o, ResponseFunctionToolCall):
            result = fc.handle_function_call(o)
            results.append(result)
    assert "guest" in results[0]


def test_openai_funcall_empty():
    def ping() -> str:
        return "pong"

    fc = Funcall([ping])
    resp = openai.responses.create(
        model="gpt-4.1-nano",
        input="Use function call to ping",
        tools=fc.get_tools(),
    )
    results = []
    for o in resp.output:
        if isinstance(o, ResponseFunctionToolCall):
            result = fc.handle_function_call(o)
            results.append(result)
    assert "pong" in results


def test_openai_funcall_union_310():
    def parse_number(n: int | str) -> int:
        """If n is int, return as is; if str, try to parse as int."""
        if isinstance(n, int):
            return n
        return int(n)

    fc = Funcall([parse_number])
    resp = openai.responses.create(
        model="gpt-4.1-nano",
        input="Use function call to parse the number '123'",
        tools=fc.get_tools(),
    )
    results = []
    for o in resp.output:
        if isinstance(o, ResponseFunctionToolCall):
            result = fc.handle_function_call(o)
            results.append(result)
    assert 123 in results


def test_openai_funcall_optional_310():
    def greet(name: str | None = None) -> str:
        if name:
            return f"Hello, {name}!"
        return "Hello, guest!"

    fc = Funcall([greet])
    resp = openai.responses.create(
        model="gpt-4.1-nano",
        input="Use function call to greet without a name",
        tools=fc.get_tools(),
    )
    results = []
    for o in resp.output:
        if isinstance(o, ResponseFunctionToolCall):
            result = fc.handle_function_call(o)
            results.append(result)
    assert "guest" in results[0]


# dataclass 可选参数测试
@dataclass
class UserInfo:
    name: str
    age: int | None = None


def test_openai_funcall_dataclass_optional():
    def get_user_age(user: UserInfo) -> str:
        if user.age is not None:
            return f"{user.name} is {user.age} years old."
        return f"{user.name}'s age is unknown."

    fc = Funcall([get_user_age])
    print(fc.get_tools())
    resp = openai.responses.create(
        model="gpt-4.1-nano",
        input="Get the age of user Alice (age unknown)",
        tools=fc.get_tools(),
    )
    results = []
    for o in resp.output:
        if isinstance(o, ResponseFunctionToolCall):
            result = fc.handle_function_call(o)
            results.append(result)
    assert "unknown" in results[0]


# pydantic 可选参数测试
class Product(BaseModel):
    name: str
    price: float
    description: str | None = None


def test_openai_funcall_pydantic_optional():
    def get_product_desc(product: Product) -> str:
        if product.description:
            return f"{product.name}: {product.description}"
        return f"{product.name} has no description."

    fc = Funcall([get_product_desc])
    resp = openai.responses.create(
        model="gpt-4.1-nano",
        input="Get the description of product 'Widget' (no description)",
        tools=fc.get_tools(),
    )
    results = []
    for o in resp.output:
        if isinstance(o, ResponseFunctionToolCall):
            result = fc.handle_function_call(o)
            results.append(result)
    assert "no description" in results[0]


def test_openai_funcall_union_param():
    def echo(x: int | str) -> str:
        return str(x)

    fc = Funcall([echo])
    resp = openai.responses.create(
        model="gpt-4.1-nano",
        input="Use function call to echo 123",
        tools=fc.get_tools(),
    )
    results = []
    for o in resp.output:
        if isinstance(o, ResponseFunctionToolCall):
            result = fc.handle_function_call(o)
            results.append(result)
    assert "123" in results


def test_openai_funcall_union_type():
    def echo(x: Union[int, str]) -> str:  # noqa: UP007
        return str(x)

    fc = Funcall([echo])
    resp = openai.responses.create(
        model="gpt-4.1-nano",
        input="Use function call to echo 123",
        tools=fc.get_tools(),
    )
    results = []
    for o in resp.output:
        if isinstance(o, ResponseFunctionToolCall):
            result = fc.handle_function_call(o)
            results.append(result)
    assert "123" in results


def test_openai_funcall_optional_param():
    def greet(name: str | None = None) -> str:
        if name:
            return f"Hello, {name}!"
        return "Hello, guest!"

    fc = Funcall([greet])
    resp = openai.responses.create(
        model="gpt-4.1-nano",
        input="Use function call to greet without a name",
        tools=fc.get_tools(),
    )
    results = []
    for o in resp.output:
        if isinstance(o, ResponseFunctionToolCall):
            result = fc.handle_function_call(o)
            results.append(result)
    assert "guest" in results[0]


def test_greet_function_with_optional_type():
    def greet(name: Optional[str] = None) -> str:  # noqa: UP045
        if name:
            return f"Hello, {name}!"
        return "Hello, guest!"

    fc = Funcall([greet])
    resp = openai.responses.create(
        model="gpt-4.1-nano",
        input="Use function call to greet without a name",
        tools=fc.get_tools(),
    )
    results = []
    for o in resp.output:
        if isinstance(o, ResponseFunctionToolCall):
            result = fc.handle_function_call(o)
            results.append(result)
    assert "guest" in results[0]


def test_openai_funcall_pydantic_nested():
    class Inner(BaseModel):
        value: int

    class Middle(BaseModel):
        inners: list[Inner]

    class Outer(BaseModel):
        middles: list[Middle]

    def sum_all(outer: Outer) -> int:
        return sum(inner.value for middle in outer.middles for inner in middle.inners)

    fc = Funcall([sum_all])
    resp = openai.responses.create(
        model="gpt-4.1-nano",
        input="Sum all values in: [[1,2],[3]]",
        tools=fc.get_tools(),
    )
    results = []
    for o in resp.output:
        if isinstance(o, ResponseFunctionToolCall):
            result = fc.handle_function_call(o)
            results.append(result)


def test_openai_funcall_dataclass_nested():
    @dataclass
    class Inner:
        value: int

    @dataclass
    class Middle:
        inners: list[Inner]

    @dataclass
    class Outer:
        middles: list[Middle]

    def sum_all(outer: Outer) -> int:
        return sum(inner.value for middle in outer.middles for inner in middle.inners)

    fc = Funcall([sum_all])
    resp = openai.responses.create(
        model="gpt-4.1-nano",
        input="Sum all values in: [[4,5],[6]], only call the function once.",
        tools=fc.get_tools(),
    )
    results = []
    for o in resp.output:
        if isinstance(o, ResponseFunctionToolCall):
            result = fc.handle_function_call(o)
            results.append(result)
