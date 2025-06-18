import pytest

from funcall.params_to_schema import params_to_schema


def test_params_to_schema_basic():
    schema = params_to_schema([int, str, float])
    props = schema["properties"]
    assert props["param_0"]["type"] == "integer"
    assert props["param_1"]["type"] == "string"
    assert props["param_2"]["type"] == "number"


def test_params_to_schema_dict():
    with pytest.raises(TypeError, match="is not supported directly, use pydantic BaseModel or dataclass instead."):
        params_to_schema([dict[str, int]])
