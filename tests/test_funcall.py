import dataclasses
import json
from unittest.mock import patch

import pytest
from litellm import ResponseFunctionToolCall
from pydantic import BaseModel, Field

from funcall import Context, Funcall, generate_meta


class UnsupportedRefError(ValueError):
    """Exception raised for unsupported references."""


def resolve_ref(schema: dict, ref: str) -> dict:
    if not ref.startswith("#/$defs/"):
        raise UnsupportedRefError("Unsupported ref: " + ref)
    def_name = ref.split("/", 2)[-1]
    return schema["$defs"][def_name]


def get_dummy_response_function_tool_call(name: str, arguments: str) -> ResponseFunctionToolCall:
    return ResponseFunctionToolCall(name=name, arguments=arguments, call_id="dummy_call_id", type="function_call")


def test_generate_meta_normal_func():
    def add(a: int, b: int) -> int:
        """Add two numbers"""
        return a + b

    meta = generate_meta(add)
    assert meta["name"] == "add"
    assert meta["description"] == "Add two numbers"
    props = meta["parameters"]["properties"]
    assert props["a"]["type"] == "integer"
    assert props["b"]["type"] == "integer"
    assert "a" in meta["parameters"]["required"]
    assert "b" in meta["parameters"]["required"]


def test_generate_meta_pydantic():
    class FooModel(BaseModel):
        x: int
        y: int | None = None
        z: int = Field(default=3)

    def foo(data: FooModel) -> int:
        """foo doc"""
        return data.x

    meta = generate_meta(foo)
    assert meta["name"] == "foo"
    props = meta["parameters"]["properties"]
    # 字段直接在顶层
    assert "x" in props
    assert "y" in props
    assert "z" in props
    assert "x" in meta["parameters"]["required"]
    # openai schema 中，即使可选，类型的字段也需要出现在 required 中
    assert "y" in meta["parameters"]["required"]


def test_generate_meta_dataclass():
    @dataclasses.dataclass
    class D:
        x: int
        y: int = 2

    def bar(data: D) -> int:
        """bar doc"""
        return data.x + data.y

    meta = generate_meta(bar)
    assert meta["name"] == "bar"
    props = meta["parameters"]["properties"]
    assert "x" in props
    assert "y" in props
    assert "x" in meta["parameters"]["required"]
    # openai schema 中，即使可选，类型的字段也需要出现在 required 中
    assert "y" in meta["parameters"]["required"]


def test_generate_meta_param_type_builtin_types():
    def foo(*, a: int, b: float, c: str, d: bool) -> None:
        pass

    meta = generate_meta(foo)
    props = meta["parameters"]["properties"]
    assert props["a"]["type"] == "integer"
    assert props["b"]["type"] == "number"
    assert props["c"]["type"] == "string"
    assert props["d"]["type"] == "boolean"
    assert set(meta["parameters"]["required"]) == {"a", "b", "c", "d"}


def test_generate_meta_param_type_list_and_dict():
    def foo(a: list[int], b: list[str]) -> None:
        pass

    meta = generate_meta(foo)
    props = meta["parameters"]["properties"]
    assert props["a"]["type"] == "array"
    assert props["b"]["type"] == "array"


def test_dict_raise_error():
    def foo(a: dict[str, int]) -> None:
        pass

    with pytest.raises(TypeError, match="is not supported directly, use pydantic BaseModel or dataclass instead."):
        generate_meta(foo)


def test_get_tools():
    def f1(a: int) -> int:
        return a

    def f2(b: str) -> str:
        return b

    fc = Funcall([f1, f2])
    tools = fc.get_tools()
    assert len(tools) == 2
    assert tools[0]["name"] == "f1"
    assert tools[1]["name"] == "f2"


def test_handle_function_call_normal():
    def add(a: int, b: int) -> int:
        return a + b

    fc = Funcall([add])
    item = get_dummy_response_function_tool_call("add", json.dumps({"a": 1, "b": 2}))
    with patch("litellm.ResponseFunctionToolCall", get_dummy_response_function_tool_call):
        result = fc.handle_function_call(item)
    assert result == 3


def test_handle_function_call_normal_async():
    async def add(a: int, b: int) -> int:
        return a + b

    fc = Funcall([add])
    item = get_dummy_response_function_tool_call("add", json.dumps({"a": 1, "b": 2}))
    with patch("litellm.ResponseFunctionToolCall", get_dummy_response_function_tool_call):
        result = fc.handle_function_call(item)
    assert result == 3


def test_no_function_call():
    fc = Funcall()
    assert fc.get_tools() == []


def test_handle_function_call_basemodel():
    # 直接用真实的 pydantic BaseModel
    class MyModel(BaseModel):
        x: int
        y: int | None = None

    def foo(data: MyModel) -> int:
        return data.x * 2

    foo.__annotations__ = {"data": MyModel}
    fc = Funcall([foo])
    item = get_dummy_response_function_tool_call("foo", json.dumps({"x": 5}))
    result = fc.handle_function_call(item)
    assert result == 10


def test_handle_function_call_not_found():
    fc = Funcall([])
    item = get_dummy_response_function_tool_call("not_exist", "{}")
    with pytest.raises(ValueError, match="Function not_exist not found"):
        fc.handle_function_call(item)


def test_handle_function_call_invalid_json():
    def add(a: int, b: int) -> int:
        return a + b

    fc = Funcall([add])
    item = get_dummy_response_function_tool_call("add", "not a json")
    with pytest.raises(json.JSONDecodeError):
        fc.handle_function_call(item)


def test_generate_meta_param_type_dataclass():
    @dataclasses.dataclass
    class MyData:
        a: int
        b: str = "default"

    def foo(data: MyData) -> str:
        return f"{data.a}-{data.b}"

    meta = generate_meta(foo)
    fc = Funcall([foo])
    item = get_dummy_response_function_tool_call("foo", json.dumps({"a": 1, "b": "test"}))
    with patch("litellm.ResponseFunctionToolCall", get_dummy_response_function_tool_call):
        fc.handle_function_call(item)
    props = meta["parameters"]["properties"]
    assert "a" in props
    assert "b" in props
    assert props["a"]["type"] == "integer"
    assert props["b"]["type"] == "string"
    assert "a" in meta["parameters"]["required"]
    # openai schema 中，即使可选，类型的字段也需要出现在 required 中
    assert "b" in meta["parameters"]["required"]


def test_handle_function_call_with_context_dataclass():
    @dataclasses.dataclass
    class MyData:
        a: int

    @dataclasses.dataclass
    class MyCtx:
        user: str

    def foo(data: MyData, ctx: Context[MyCtx]) -> str:
        return f"{data.a}-{ctx.value.user}"

    meta = generate_meta(foo)
    # context 参数不应出现在 schema
    props = meta["parameters"]["properties"]
    assert "ctx" not in props
    assert "a" in props
    fc = Funcall([foo])
    item = get_dummy_response_function_tool_call("foo", json.dumps({"a": 42}))
    ctx = Context(MyCtx(user="alice"))
    result = fc.handle_function_call(item, context=ctx)
    assert result == "42-alice"


def test_handle_function_call_with_context_pydantic():
    class MyData(BaseModel):
        a: int

    class MyCtx(BaseModel):
        user: str

    def foo(data: MyData, whatever: Context[MyCtx]) -> str:
        return f"{data.a}-{whatever.value.user}"

    meta = generate_meta(foo)
    # context 参数不应出现在 schema
    props = meta["parameters"]["properties"]
    assert "whatever" not in props
    assert "a" in props
    fc = Funcall([foo])
    item = get_dummy_response_function_tool_call("foo", json.dumps({"a": 7}))
    ctx = Context(MyCtx(user="bob"))
    result = fc.handle_function_call(item, context=ctx)
    assert result == "7-bob"


def test_generate_meta_multiple_contexts_warns_and_injects(caplog: pytest.LogCaptureFixture):
    @dataclasses.dataclass
    class MyData:
        a: int

    @dataclasses.dataclass
    class MyCtx:
        user: str

    def foo(data: MyData, ctx1: Context[MyCtx], ctx2: Context[MyCtx]) -> str:
        return f"{data.a}-{ctx1.value.user}-{ctx2.value.user}"

    with caplog.at_level("WARNING"):
        meta = generate_meta(foo)
    # 两个 context 参数都不应出现在 schema
    props = meta["parameters"]["properties"]
    assert "ctx1" not in props
    assert "ctx2" not in props
    assert "a" in props
    # warning 被触发
    assert any("Multiple Context-type parameters detected" in r.message for r in caplog.records)
    fc = Funcall([foo])
    item = get_dummy_response_function_tool_call("foo", json.dumps({"a": 1}))
    ctx = Context(MyCtx(user="alice"))
    result = fc.handle_function_call(item, context=ctx)
    # 两个 context 参数都注入同一个实例
    assert result == "1-alice-alice"


def test_handle_function_call_array_of_pydantic():
    class Item(BaseModel):
        name: str
        value: int

    def sum_values(items: list[Item]) -> int:
        """Sum the 'value' field of all items in the list."""
        return sum(item.value for item in items)

    meta = generate_meta(sum_values)
    # 检查 schema 结构
    props = meta["parameters"]["properties"]
    assert props["items"]["type"] == "array"
