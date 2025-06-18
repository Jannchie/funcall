import contextlib
import dataclasses
import inspect
import json
from collections.abc import Callable
from logging import getLogger
from typing import Generic, TypeVar, Union, get_args, get_type_hints

from openai.types.responses import (
    FunctionToolParam,
    ResponseFunctionToolCall,
)
from pydantic import BaseModel

from .params_to_schema import params_to_schema

logger = getLogger("funcall")

T = TypeVar("T")


class Context(Generic[T]):
    def __init__(self, value: T | None = None) -> None:
        self.value = value


def generate_meta(func: Callable) -> FunctionToolParam:
    sig = inspect.signature(func)
    type_hints = get_type_hints(func)
    doc = func.__doc__.strip() if func.__doc__ else ""
    param_names = []
    param_types = []
    context_param_count = 0
    for name in sig.parameters:
        hint = type_hints.get(name, str)
        # 跳过所有类型为 Context 的参数
        if getattr(hint, "__origin__", None) is Context or hint is Context:
            context_param_count += 1
            continue
        param_names.append(name)
        param_types.append(hint)
    if context_param_count > 1:
        logger.warning("Multiple Context-type parameters detected in function '%s'. Only one context instance will be injected at runtime.", func.__name__)
    schema = params_to_schema(param_types)
    # 单参数且为 dataclass 或 BaseModel，提升其字段为顶层
    if len(param_names) == 1:
        hint = param_types[0]
        if isinstance(hint, type) and (dataclasses.is_dataclass(hint) or (BaseModel and issubclass(hint, BaseModel))):
            prop = schema["properties"]["param_0"]
            # 跟进 $ref
            if "$ref" in prop:
                ref = prop["$ref"]
                def_name = ref.split("/", 2)[-1]
                def_schema = schema["$defs"][def_name]
                properties = def_schema["properties"]
                required = def_schema.get("required", [])
                additional = def_schema.get("additionalProperties", False)
            else:
                properties = prop["properties"]
                required = prop.get("required", [])
                additional = prop.get("additionalProperties", False)
            meta: FunctionToolParam = {
                "type": "function",
                "name": func.__name__,
                "description": doc,
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    # OpenAI Function Calling 要求 required 必须包含所有字段
                    "required": list(properties.keys()),
                    "additionalProperties": additional,
                },
                "strict": True,
            }
            # if "$defs" in schema:
            #     meta["parameters"]["$defs"] = schema["$defs"]
            return meta
    # 多参数或非 dataclass/BaseModel
    properties = {}
    required = []
    for i, name in enumerate(param_names):
        prop = schema["properties"][f"param_{i}"]
        properties[name] = prop
        required.append(name)
    meta: FunctionToolParam = {
        "type": "function",
        "name": func.__name__,
        "description": doc,
        "parameters": {
            "type": "object",
            "properties": properties,
            "required": required,
            "additionalProperties": False,
        },
        "strict": True,
    }
    if "$defs" in schema:
        meta["parameters"]["$defs"] = schema["$defs"]
    return meta


def _convert_arg(value: object, hint: type) -> object:
    result = value
    origin = getattr(hint, "__origin__", None)
    if origin is list or origin is set or origin is tuple:
        args = get_args(hint)
        item_type = args[0] if args else str
        result = [_convert_arg(v, item_type) for v in value]
    elif origin is dict:
        result = value
    elif getattr(hint, "__origin__", None) is Union:
        # 只处理 Optional[X]，否则直接返回
        args = get_args(hint)
        non_none = [a for a in args if a is not type(None)]
        result = _convert_arg(value, non_none[0]) if len(non_none) == 1 else value
    elif (isinstance(hint, type) and BaseModel and issubclass(hint, BaseModel)) or dataclasses.is_dataclass(hint):
        result = hint(**value) if isinstance(value, dict) else value
    else:
        result = value
    return result


class Funcall:
    def __init__(self, functions: list | None = None) -> None:
        if functions is None:
            functions = []
        self.functions = functions
        self.function_map = {func.__name__: func for func in functions}

    def get_tools(self) -> list[FunctionToolParam]:
        return [generate_meta(func) for func in self.functions]

    def handle_function_call(self, item: ResponseFunctionToolCall, context: object = None):
        if item.name in self.function_map:
            func = self.function_map[item.name]
            args = item.arguments
            sig = inspect.signature(func)
            type_hints = get_type_hints(func)
            kwargs = json.loads(args)
            # 兼容单参数为数组时直接传数组
            if isinstance(kwargs, list) and len(sig.parameters) == 1:
                only_param = next(iter(sig.parameters))
                kwargs = {only_param: kwargs}
            new_kwargs = {}
            for name in sig.parameters:
                hint = type_hints.get(name, str)
                if getattr(hint, "__origin__", None) is Context or hint is Context:
                    new_kwargs[name] = context
                elif name in kwargs:
                    new_kwargs[name] = _convert_arg(kwargs[name], hint)
                elif isinstance(hint, type) and BaseModel and issubclass(hint, BaseModel):
                    # 支持 pydantic 单对象参数
                    with contextlib.suppress(Exception):
                        new_kwargs[name] = hint(**kwargs)
                elif dataclasses.is_dataclass(hint):
                    # 支持 dataclass 单对象参数
                    with contextlib.suppress(Exception):
                        new_kwargs[name] = hint(**kwargs)
            return func(**new_kwargs)
        msg = f"Function {item.name} not found"
        raise ValueError(msg)


__all__ = ["Context", "Funcall", "generate_meta"]
