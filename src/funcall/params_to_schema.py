import dataclasses
import types
from dataclasses import fields, is_dataclass
from typing import Any, get_args, get_origin
from typing import Any as TypingAny
from typing import Union as TypingUnion

from pydantic import BaseModel, create_model


def _create_union_type(union_types: tuple) -> type:
    """Create a Union type, handling compatibility issues"""
    try:
        return TypingUnion[union_types]  # noqa: UP007
    except TypeError:
        return TypingUnion.__getitem__(union_types)


def _handle_tuple_type(args: tuple) -> type:
    """Handle Tuple type conversion"""
    if not args:
        return list[TypingAny]

    # Tuple[T, ...] -> List[T]
    if len(args) == 2 and args[1] is Ellipsis:
        item_type = to_field_type(args[0])
        return list[item_type]

    # Tuple[T1, T2, ...] -> List[Union[T1, T2, ...]]
    item_types = tuple(to_field_type(a) for a in args)
    if len(item_types) == 1:
        return list[item_types[0]]

    union_type = _create_union_type(item_types)
    return list[union_type]


def _dataclass_to_pydantic_model(dataclass_type: type) -> type:
    """Convert a dataclass to a Pydantic Model"""
    model_fields = {}

    for field in fields(dataclass_type):
        # Determine the default value of the field
        if field.default is not dataclasses.MISSING:
            default_value = field.default
        elif field.default_factory is not dataclasses.MISSING:
            # 修复：不要立即调用工厂函数，而是传递工厂函数本身
            default_value = field.default_factory
        else:
            default_value = ...

        model_fields[field.name] = (field.type, default_value)

    # Create Pydantic Model
    model = create_model(dataclass_type.__name__, **model_fields)

    # Add field descriptions
    _add_field_descriptions(model, dataclass_type)

    return model


def _add_field_descriptions(model: type, dataclass_type: type) -> None:
    """Add descriptions to Pydantic Model fields"""
    for field in fields(dataclass_type):
        if hasattr(field, "metadata") and "description" in field.metadata:
            description = field.metadata["description"]
            if hasattr(model, "model_fields") and field.name in model.model_fields:
                model.model_fields[field.name].description = description


def to_field_type(param: type) -> type:
    """
    Convert various type annotations to field types.
    """
    if param is None:
        return type(None)

    origin = get_origin(param)
    args = get_args(param)

    # 修复：更清晰的类型检查顺序
    # 首先检查是否是 Pydantic BaseModel
    if isinstance(param, type) and issubclass(param, BaseModel):
        return param

    # 检查是否是 dataclass
    if is_dataclass(param):
        return _dataclass_to_pydantic_model(param)

    # 处理泛型类型
    if origin is not None:
        # Union/Optional (compatible with 3.10+ X | Y)
        if origin is TypingUnion or (hasattr(types, "UnionType") and origin is types.UnionType):
            union_types = tuple(to_field_type(a) for a in args)
            return _create_union_type(union_types)

        # List
        if origin is list:
            item_type = to_field_type(args[0]) if args else TypingAny
            return list[item_type]

        # Dict - 提供更清晰的错误信息
        if origin is dict:
            msg = f"Dict type {param} is not supported directly, use pydantic BaseModel or dataclass instead."
            raise TypeError(msg)

        # Tuple
        if origin is tuple:
            return _handle_tuple_type(args)

    # 基本类型处理
    if isinstance(param, type):
        if param is dict:
            msg = "Dict type is not supported directly, use pydantic BaseModel or dataclass instead."
            raise TypeError(msg)
        return param

    # 如果都不匹配，抛出错误
    msg = f"Unsupported param type: {param} (type: {type(param)})"
    raise TypeError(msg)


def params_to_schema(params: list[Any]) -> dict[str, Any]:
    """
    Read a parameter list, which can contain various types, dataclasses, pydantic models, basic types, even nested or nested in lists.
    Output a jsonschema describing this set of parameters.
    """
    # 修复：添加输入验证
    if not isinstance(params, list):
        msg = "params must be a list"
        raise TypeError(msg)

    # Build parameter model
    if not params:
        # Handle the case of an empty parameter list
        model = create_model("ParamsModel")
    else:
        model_fields = {}
        for i, p in enumerate(params):
            field_type = to_field_type(p)
            model_fields[f"param_{i}"] = (field_type, ...)

        model = create_model("ParamsModel", **model_fields)

    schema = model.model_json_schema()
    _normalize_schema(schema)

    return schema


def _normalize_schema(schema: dict | list) -> None:
    """
    Normalize schema, add additionalProperties: false and fix required fields
    """
    if isinstance(schema, dict):
        if schema.get("type") == "object":
            schema.setdefault("additionalProperties", False)
            # OpenAI Function Calling: required must contain all properties
            if "properties" in schema:
                schema["required"] = list(schema["properties"].keys())

        # Recursively handle nested objects
        for value in schema.values():
            if isinstance(value, (dict, list)):  # 修复：添加类型检查
                _normalize_schema(value)

    elif isinstance(schema, list):
        for item in schema:
            if isinstance(item, (dict, list)):  # 修复：添加类型检查
                _normalize_schema(item)
