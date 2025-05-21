import dataclasses
import inspect
import json
from collections.abc import Callable
from logging import getLogger
from typing import get_type_hints

from openai.types.responses import (
    ResponseFunctionToolCall,
    ToolParam,
)

# 新增导入
from pydantic import BaseModel
from pydantic.fields import FieldInfo

logger = getLogger("funcall")


def param_type(py_type: str | type | FieldInfo | None) -> str:
    """Map Python types to JSON Schema types"""
    type_map = {
        int: "number",
        float: "number",
        str: "string",
        bool: "boolean",
        list: "array",
        dict: "object",
    }
    origin = getattr(py_type, "__origin__", None)
    if origin is not None:
        origin_map = {list: "array", dict: "object"}
        if origin in origin_map:
            return origin_map[origin]
    if py_type in type_map:
        return type_map[py_type]
    if BaseModel and isinstance(py_type, type) and issubclass(py_type, BaseModel):
        return "object"
    if dataclasses.is_dataclass(py_type):
        return "object"
    if isinstance(py_type, FieldInfo):
        return param_type(py_type.annotation)
    return "string"


def generate_meta(func: Callable) -> ToolParam:
    sig = inspect.signature(func)
    type_hints = get_type_hints(func)
    params = {}
    required = []
    doc = func.__doc__.strip() if func.__doc__ else ""

    for name in sig.parameters:
        hint = type_hints.get(name, str)
        if isinstance(hint, type) and issubclass(hint, BaseModel):
            model = hint
            for field_name, field in model.model_fields.items():
                desc = field.description if field.description else None
                params[field_name] = {
                    "type": param_type(field),
                    "description": desc or f"{name}.{field_name}",
                }
                if field.is_required:
                    required.append(field_name)

        elif dataclasses.is_dataclass(hint):
            # Python dataclass
            for field in dataclasses.fields(hint):
                desc = field.metadata.get("description") if "description" in field.metadata else None
                params[field.name] = {
                    "type": param_type(field.type),
                    "description": desc or f"{name}.{field.name}",
                }
                if field.default is dataclasses.MISSING and field.default_factory is dataclasses.MISSING:
                    required.append(field.name)
        else:
            # Normal parameter
            param_desc = f"The {list(sig.parameters.keys()).index(name) + 1}th parameter"
            params[name] = {"type": param_type(hint), "description": param_desc}
            required.append(name)

    meta: ToolParam = {
        "type": "function",
        "name": func.__name__,
        "description": doc,
        "parameters": {
            "type": "object",
            "properties": params,
            "required": required,
            "additionalProperties": False,
        },
        "strict": True,
    }
    return meta


class Funcall:
    def __init__(self, functions: list | None = None) -> None:
        if functions is None:
            functions = []
        self.functions = functions
        self.function_map = {func.__name__: func for func in functions}

    def get_tools(self) -> list[ToolParam]:
        return [generate_meta(func) for func in self.functions]

    def handle_function_call(self, item: ResponseFunctionToolCall):
        if item.name in self.function_map:
            func = self.function_map[item.name]
            args = item.arguments
            if BaseModel and issubclass(
                func.__annotations__.get("data", None),
                BaseModel,
            ):
                model = func.__annotations__["data"]
                data = model.model_validate_json(args)
                result = func(data)
            else:
                kwargs = json.loads(args)
                result = func(**kwargs)

            return result
        msg = f"Function {item.name} not found"
        raise ValueError(msg)
