import dataclasses
import inspect
import json
from collections.abc import Callable
from logging import getLogger
from typing import get_type_hints

from openai.types.responses import (
    FunctionToolParam,
    ResponseFunctionToolCall,
)
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
    if isinstance(py_type, FieldInfo):
        return param_type(py_type.annotation)
    return "string"


def generate_meta(func: Callable) -> FunctionToolParam:
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
                if field.is_required():
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
            params[name] = {"type": param_type(hint)}
            required.append(name)
    meta: FunctionToolParam = {
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

    def get_tools(self) -> list[FunctionToolParam]:
        return [generate_meta(func) for func in self.functions]

    def handle_function_call(self, item: ResponseFunctionToolCall):
        if item.name in self.function_map:
            func = self.function_map[item.name]
            args = item.arguments
            sig = inspect.signature(func)
            type_hints = get_type_hints(func)
            kwargs = json.loads(args)
            new_kwargs = {}
            for name in sig.parameters:
                hint = type_hints.get(name, str)
                if isinstance(hint, type) and BaseModel and issubclass(hint, BaseModel):
                    # 用 kwargs 构造 Pydantic 对象
                    model = hint
                    model_fields = {k: v for k, v in kwargs.items() if k in model.model_fields}
                    new_kwargs[name] = model(**model_fields)
                elif dataclasses.is_dataclass(hint):
                    # 用 kwargs 构造 dataclass 对象
                    model_fields = {k: v for k, v in kwargs.items() if k in [f.name for f in dataclasses.fields(hint)]}
                    new_kwargs[name] = hint(**model_fields)
                elif name in kwargs:
                    new_kwargs[name] = kwargs[name]
            return func(**new_kwargs)
        msg = f"Function {item.name} not found"
        raise ValueError(msg)


__all__ = ["Funcall", "generate_meta", "param_type"]
