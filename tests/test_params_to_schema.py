import dataclasses

import pytest
from pydantic import BaseModel

from funcall.params_to_schema import params_to_schema


def test_params_to_schema_basic():
    schema = params_to_schema([int, str, float])
    props = schema["properties"]
    assert props["param_0"]["type"] == "integer"
    assert props["param_1"]["type"] == "string"
    assert props["param_2"]["type"] == "number"


def test_params_to_schema_dataclass():
    @dataclasses.dataclass
    class Foo:
        x: int
        y: str

    schema = params_to_schema([Foo])
    props = schema["properties"]["param_0"]
    assert "$ref" in props
    assert props["$ref"].startswith("#/$defs/Foo")
    assert "$defs" in schema
    assert "Foo" in schema["$defs"]


def test_params_to_schema_pydantic():
    class Bar(BaseModel):
        a: int
        b: float

    schema = params_to_schema([Bar])
    props = schema["properties"]["param_0"]
    assert "$ref" in props
    assert props["$ref"].startswith("#/$defs/Bar")
    assert "$defs" in schema
    assert "Bar" in schema["$defs"]


def test_params_to_schema_nested():
    class Bar(BaseModel):
        a: int

    schema = params_to_schema([list[Bar]])
    items = schema["properties"]["param_0"]["items"]
    assert "$ref" in items
    assert items["$ref"].startswith("#/$defs/Bar")


def test_params_to_schema_dict():
    with pytest.raises(TypeError, match="is not supported directly, use pydantic BaseModel or dataclass instead."):
        params_to_schema([dict[str, int]])
