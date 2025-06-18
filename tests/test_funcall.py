import dataclasses
import json
from unittest.mock import patch

import pytest
from pydantic import Field

from funcall import Funcall, generate_meta


class DummyResponseFunctionToolCall:
    def __init__(self, name: str, arguments: str) -> None:
        self.name = name
        self.arguments = arguments


def test_generate_meta_normal_func():
    def add(a: int, b: int) -> int:
        """Add two numbers"""
        return a + b

    meta = generate_meta(add)
    assert meta["name"] == "add"
    assert meta["description"] == "Add two numbers"
    assert meta["parameters"]["properties"]["a"]["type"] == "number"
    assert "a" in meta["parameters"]["required"]
    assert "b" in meta["parameters"]["required"]


def test_generate_meta_pydantic():
    from pydantic import BaseModel

    class FooModel(BaseModel):
        x: int
        y: int | None = None
        z: int = Field(default=3)

    def foo(data: FooModel) -> int:
        """foo doc"""
        return data.x

    meta = generate_meta(foo)
    assert meta["name"] == "foo"
    assert "x" in meta["parameters"]["properties"]
    assert "x" in meta["parameters"]["required"]
    assert "y" in meta["parameters"]["properties"]
    assert "y" not in meta["parameters"]["required"]


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
    assert "x" in meta["parameters"]["properties"]
    assert "x" in meta["parameters"]["required"]
    assert "y" in meta["parameters"]["properties"]
    assert "y" not in meta["parameters"]["required"]


def test_generate_meta_param_type_builtin_types():
    def foo(*, a: int, b: float, c: str, d: bool) -> None:
        pass

    meta = generate_meta(foo)
    props = meta["parameters"]["properties"]
    assert props["a"]["type"] == "number"
    assert props["b"]["type"] == "number"
    assert props["c"]["type"] == "string"
    assert props["d"]["type"] == "boolean"
    assert set(meta["parameters"]["required"]) == {"a", "b", "c", "d"}


def test_generate_meta_param_type_list_and_dict():
    def foo(a: list[int], b: list[str], c: dict, d: dict[str, int]) -> None:
        pass

    meta = generate_meta(foo)
    props = meta["parameters"]["properties"]
    assert props["a"]["type"] == "array"
    assert props["b"]["type"] == "array"
    assert props["c"]["type"] == "object"
    assert props["d"]["type"] == "object"


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
    item = DummyResponseFunctionToolCall("add", json.dumps({"a": 1, "b": 2}))
    with patch("funcall.__init__.ResponseFunctionToolCall", DummyResponseFunctionToolCall):
        result = fc.handle_function_call(item)
    assert result == 3


def test_no_function_call():
    fc = Funcall()
    assert fc.get_tools() == []


def test_handle_function_call_basemodel():
    from pydantic import BaseModel

    # 直接用真实的 pydantic BaseModel
    class MyModel(BaseModel):
        x: int
        y: int | None = None

    def foo(data: MyModel) -> int:
        return data.x * 2

    foo.__annotations__ = {"data": MyModel}
    fc = Funcall([foo])
    item = DummyResponseFunctionToolCall("foo", json.dumps({"x": 5}))
    result = fc.handle_function_call(item)
    assert result == 10


def test_handle_function_call_not_found():
    fc = Funcall([])
    item = DummyResponseFunctionToolCall("not_exist", "{}")
    with pytest.raises(ValueError, match="Function not_exist not found"):
        fc.handle_function_call(item)


def test_handle_function_call_invalid_json():
    def add(a: int, b: int) -> int:
        return a + b

    fc = Funcall([add])
    item = DummyResponseFunctionToolCall("add", "not a json")
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
    item = DummyResponseFunctionToolCall("foo", json.dumps({"a": 1, "b": "test"}))
    with patch("funcall.__init__.ResponseFunctionToolCall", DummyResponseFunctionToolCall):
        fc.handle_function_call(item)
    props = meta["parameters"]["properties"]
    assert "a" in props
    assert "b" in props
    assert props["a"]["type"] == "number"
    assert props["b"]["type"] == "string"
    assert "a" in meta["parameters"]["required"]
    assert "b" not in meta["parameters"]["required"]
