import asyncio
import concurrent.futures
import dataclasses
import inspect
import json
from collections.abc import Callable
from logging import getLogger
from typing import Generic, Literal, Required, TypedDict, TypeVar, Union, get_args, get_type_hints

import litellm
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


class LiteLLMFunctionToolParam(TypedDict):
    name: Required[str]
    parameters: Required[dict[str, object] | None]
    strict: Required[bool | None]
    type: Required[Literal["function"]]
    description: str | None


class LiteLLMFunctionToolParam(TypedDict):
    type: Literal["function"]
    function: Required[LiteLLMFunctionToolParam]


def generate_meta(func: Callable, target: Literal["openai", "litellm"] = "openai") -> FunctionToolParam:
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
            properties = prop["properties"]
            required = prop.get("required", [])
            additional = prop.get("additionalProperties", False)
            if target == "litellm":
                meta = {
                    "type": "function",
                    "function": {
                        "name": func.__name__,
                        "description": doc,
                        "parameters": {
                            "type": "object",
                            "properties": properties,
                            # litellm Function Calling 要求 required 必须包含所有字段
                            "required": list(properties.keys()) if required else [],
                            "additionalProperties": additional,
                        },
                    },
                }
            elif target == "openai":
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


def _convert_arg(value: object, hint: type) -> object:  # noqa: PLR0911
    origin = getattr(hint, "__origin__", None)
    if origin in (list, set, tuple):
        args = get_args(hint)
        item_type = args[0] if args else str
        return [_convert_arg(v, item_type) for v in value]
    if origin is dict:
        return value
    if getattr(hint, "__origin__", None) is Union:
        args = get_args(hint)
        non_none = [a for a in args if a is not type(None)]
        return _convert_arg(value, non_none[0]) if len(non_none) == 1 else value
    if isinstance(hint, type) and BaseModel and issubclass(hint, BaseModel):
        if isinstance(value, dict):
            fields = hint.model_fields
            return hint(**{k: _convert_arg(v, fields[k].annotation) if k in fields else v for k, v in value.items()})
        return value
    if dataclasses.is_dataclass(hint):
        if isinstance(value, dict):
            field_types = {f.name: f.type for f in dataclasses.fields(hint)}
            return hint(**{k: _convert_arg(v, field_types.get(k, type(v))) for k, v in value.items()})
        return value
    return value


def _is_async_function(func: Callable) -> bool:
    """检查函数是否为异步函数"""
    return inspect.iscoroutinefunction(func)


class Funcall:
    def __init__(self, functions: list | None = None) -> None:
        if functions is None:
            functions = []
        self.functions = functions
        self.function_map = {func.__name__: func for func in functions}

    # target 可以是 openai 或者 litellm
    def get_tools(self, target: Literal["openai", "litellm"] = "openai") -> list[FunctionToolParam]:
        return [generate_meta(func, target) for func in self.functions]

    def _prepare_function_call(self, func_name: str, args: str, context: object = None) -> tuple[Callable, dict]:
        """准备函数调用的通用逻辑"""
        if func_name not in self.function_map:
            msg = f"Function {func_name} not found"
            raise ValueError(msg)

        func = self.function_map[func_name]
        sig = inspect.signature(func)
        type_hints = get_type_hints(func)
        kwargs = json.loads(args)

        # 找出所有非 context 参数
        non_context_params = [name for name in sig.parameters if not (getattr(type_hints.get(name, str), "__origin__", None) is Context or type_hints.get(name, str) is Context)]

        # 如果只有一个非 context 参数，且 kwargs 不是以该参数名为 key 的 dict，则包裹
        if len(non_context_params) == 1 and (not isinstance(kwargs, dict) or set(kwargs.keys()) != set(non_context_params)):
            only_param = non_context_params[0]
            kwargs = {only_param: kwargs}

        new_kwargs = {}
        for name in sig.parameters:
            hint = type_hints.get(name, str)
            if getattr(hint, "__origin__", None) is Context or hint is Context:
                new_kwargs[name] = context
            elif name in kwargs:
                new_kwargs[name] = _convert_arg(kwargs[name], hint)

        return func, new_kwargs

    def handle_openai_function_call(self, item: ResponseFunctionToolCall, context: object = None):
        """同步处理 OpenAI 函数调用"""
        if not isinstance(item, ResponseFunctionToolCall):
            msg = "item must be an instance of ResponseFunctionToolCall"
            raise TypeError(msg)

        func, kwargs = self._prepare_function_call(item.name, item.arguments, context)

        if _is_async_function(func):
            logger.warning("Function %s is async but being called synchronously. Consider using handle_openai_function_call_async.", item.name)
            # 在同步上下文中运行异步函数
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    # 如果已经在事件循环中，创建新的任务

                    with concurrent.futures.ThreadPoolExecutor() as executor:
                        future = executor.submit(asyncio.run, func(**kwargs))
                        return future.result()
                else:
                    return loop.run_until_complete(func(**kwargs))
            except RuntimeError:
                # 没有事件循环，创建新的
                return asyncio.run(func(**kwargs))
        else:
            return func(**kwargs)

    async def handle_openai_function_call_async(self, item: ResponseFunctionToolCall, context: object = None):
        """异步处理 OpenAI 函数调用"""
        if not isinstance(item, ResponseFunctionToolCall):
            msg = "item must be an instance of ResponseFunctionToolCall"
            raise TypeError(msg)

        func, kwargs = self._prepare_function_call(item.name, item.arguments, context)

        if _is_async_function(func):
            return await func(**kwargs)
        # 在线程池中运行同步函数，避免阻塞事件循环
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, lambda: func(**kwargs))

    def handle_litellm_function_call(self, item: litellm.ChatCompletionMessageToolCall, context: object = None):
        """同步处理 LiteLLM 函数调用"""
        if not isinstance(item, litellm.ChatCompletionMessageToolCall):
            msg = "item must be an instance of litellm.ChatCompletionMessageToolCall"
            raise TypeError(msg)

        func, kwargs = self._prepare_function_call(item.function.name, item.function.arguments, context)

        if _is_async_function(func):
            logger.warning("Function %s is async but being called synchronously. Consider using handle_litellm_function_call_async.", item.function.name)
            # 在同步上下文中运行异步函数
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    # 如果已经在事件循环中，创建新的任务

                    with concurrent.futures.ThreadPoolExecutor() as executor:
                        future = executor.submit(asyncio.run, func(**kwargs))
                        return future.result()
                else:
                    return loop.run_until_complete(func(**kwargs))
            except RuntimeError:
                # 没有事件循环，创建新的
                return asyncio.run(func(**kwargs))
        else:
            return func(**kwargs)

    async def handle_litellm_function_call_async(self, item: litellm.ChatCompletionMessageToolCall, context: object = None):
        """异步处理 LiteLLM 函数调用"""
        if not isinstance(item, litellm.ChatCompletionMessageToolCall):
            msg = "item must be an instance of litellm.ChatCompletionMessageToolCall"
            raise TypeError(msg)

        func, kwargs = self._prepare_function_call(item.function.name, item.function.arguments, context)

        if _is_async_function(func):
            return await func(**kwargs)
        # 在线程池中运行同步函数，避免阻塞事件循环
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, lambda: func(**kwargs))

    def handle_function_call(self, item: ResponseFunctionToolCall | litellm.ChatCompletionMessageToolCall, context: object = None):
        """同步函数调用处理"""
        if isinstance(item, ResponseFunctionToolCall):
            return self.handle_openai_function_call(item, context)
        if isinstance(item, litellm.ChatCompletionMessageToolCall):
            return self.handle_litellm_function_call(item, context)
        msg = "item must be an instance of ResponseFunctionToolCall or litellm.ChatCompletionMessageToolCall"
        raise TypeError(msg)

    async def handle_function_call_async(self, item: ResponseFunctionToolCall | litellm.ChatCompletionMessageToolCall, context: object = None):
        """异步函数调用处理"""
        if isinstance(item, ResponseFunctionToolCall):
            return await self.handle_openai_function_call_async(item, context)
        if isinstance(item, litellm.ChatCompletionMessageToolCall):
            return await self.handle_litellm_function_call_async(item, context)
        msg = "item must be an instance of ResponseFunctionToolCall or litellm.ChatCompletionMessageToolCall"
        raise TypeError(msg)


__all__ = ["Context", "Funcall", "generate_meta"]
