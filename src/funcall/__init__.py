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

from funcall.params_to_schema import params_to_schema

logger = getLogger("funcall")

T = TypeVar("T")


class Context(Generic[T]):
    """Generic context container for dependency injection in function calls."""

    def __init__(self, value: T | None = None) -> None:
        self.value = value


class LiteLLMFunctionSpec(TypedDict):
    """Type definition for LiteLLM function specification."""

    name: Required[str]
    parameters: Required[dict[str, object] | None]
    strict: Required[bool | None]
    type: Required[Literal["function"]]
    description: str | None


class LiteLLMFunctionToolParam(TypedDict):
    """Type definition for LiteLLM function tool parameter."""

    type: Literal["function"]
    function: Required[LiteLLMFunctionSpec]


def generate_function_metadata(
    func: Callable,
    target: Literal["openai", "litellm"] = "openai",
) -> FunctionToolParam | LiteLLMFunctionToolParam:
    """
    Generate function metadata for OpenAI or LiteLLM function calling.

    Args:
        func: The function to generate metadata for
        target: Target platform ("openai" or "litellm")

    Returns:
        Function metadata in the appropriate format
    """
    signature = inspect.signature(func)
    type_hints = get_type_hints(func)
    description = func.__doc__.strip() if func.__doc__ else ""

    # Extract non-context parameters
    param_names, param_types, context_count = _extract_parameters(signature, type_hints)

    if context_count > 1:
        logger.warning(
            "Multiple Context-type parameters detected in function '%s'. Only one context instance will be injected at runtime.",
            func.__name__,
        )

    schema = params_to_schema(param_types)

    # Handle single parameter case (dataclass or BaseModel)
    if len(param_names) == 1:
        metadata = _generate_single_param_metadata(
            func,
            param_types[0],
            schema,
            description,
            target,
        )
        if metadata:
            return metadata

    # Handle multiple parameters case
    return _generate_multi_param_metadata(func, param_names, schema, description, target)


def _extract_parameters(signature: inspect.Signature, type_hints: dict) -> tuple[list[str], list[type], int]:
    """Extract parameter information from function signature."""
    param_names = []
    param_types = []
    context_count = 0

    for name in signature.parameters:
        hint = type_hints.get(name, str)

        # Skip Context-type parameters
        if _is_context_type(hint):
            context_count += 1
            continue

        param_names.append(name)
        param_types.append(hint)

    return param_names, param_types, context_count


def _is_context_type(hint: type) -> bool:
    """Check if a type hint is a Context type."""
    return getattr(hint, "__origin__", None) is Context or hint is Context


def _is_optional_type(hint: type) -> bool:
    """判断类型是否为 Optional/Union[..., None]"""
    origin = getattr(hint, "__origin__", None)
    if origin is Union:
        args = get_args(hint)
        return any(a is type(None) for a in args)
    return False


def _generate_single_param_metadata(
    func: Callable,
    param_type: type,
    schema: dict,
    description: str,
    target: str,
) -> FunctionToolParam | LiteLLMFunctionToolParam | None:
    """Generate metadata for functions with a single dataclass/BaseModel parameter."""
    if not (isinstance(param_type, type) and (dataclasses.is_dataclass(param_type) or (BaseModel and issubclass(param_type, BaseModel)))):
        return None

    prop = schema["properties"]["param_0"]
    properties = prop["properties"]
    required = prop.get("required", [])
    additional_properties = prop.get("additionalProperties", False)

    base_params = {
        "type": "object",
        "properties": properties,
        "additionalProperties": additional_properties,
    }

    if target == "litellm":
        model_fields = None
        if BaseModel and issubclass(param_type, BaseModel):
            model_fields = param_type.model_fields
        elif dataclasses.is_dataclass(param_type):
            model_fields = {f.name: f for f in dataclasses.fields(param_type)}
        litellm_required = []
        for k in properties:
            # 优先用 pydantic/dc 字段信息判断
            is_optional = False
            if model_fields and k in model_fields:
                if BaseModel and issubclass(param_type, BaseModel):
                    ann = model_fields[k].annotation
                    is_optional = _is_optional_type(ann) or model_fields[k].is_required is False
                else:
                    ann = model_fields[k].type
                    is_optional = _is_optional_type(ann) or (getattr(model_fields[k], "default", dataclasses.MISSING) is not dataclasses.MISSING)
            else:
                is_optional = k not in required
            if not is_optional:
                litellm_required.append(k)
        return {
            "type": "function",
            "function": {
                "name": func.__name__,
                "description": description,
                "parameters": {
                    **base_params,
                    "required": litellm_required,
                },
            },
        }

    # OpenAI format
    metadata: FunctionToolParam = {
        "type": "function",
        "name": func.__name__,
        "description": description,
        "parameters": {
            **base_params,
            "required": list(properties.keys()),
        },
        "strict": True,
    }
    return metadata


def _generate_multi_param_metadata(
    func: Callable,
    param_names: list[str],
    schema: dict,
    description: str,
    target: str,
) -> FunctionToolParam | LiteLLMFunctionToolParam:
    """Generate metadata for functions with multiple parameters."""
    properties = {}
    for i, name in enumerate(param_names):
        properties[name] = schema["properties"][f"param_{i}"]

    base_params = {
        "type": "object",
        "properties": properties,
        "additionalProperties": False,
    }

    if target == "litellm":
        sig = inspect.signature(func)
        type_hints = get_type_hints(func)
        litellm_required = []
        for name in param_names:
            hint = type_hints.get(name, str)
            param = sig.parameters[name]
            is_optional = _is_optional_type(hint) or (param.default != inspect.Parameter.empty)
            if not is_optional:
                litellm_required.append(name)
        return {
            "type": "function",
            "function": {
                "name": func.__name__,
                "description": description,
                "parameters": {
                    **base_params,
                    "required": litellm_required,
                },
            },
        }

    # OpenAI format
    metadata: FunctionToolParam = {
        "type": "function",
        "name": func.__name__,
        "description": description,
        "parameters": {
            **base_params,
            "required": list(param_names),
        },
        "strict": True,
    }

    return metadata


def _convert_argument_type(value: object, hint: type) -> object:
    """
    Convert argument values to match expected types.

    Args:
        value: The value to convert
        hint: The type hint to convert to

    Returns:
        Converted value
    """
    origin = getattr(hint, "__origin__", None)
    result = value

    # Handle collection types
    if origin in (list, set, tuple):
        args = get_args(hint)
        item_type = args[0] if args else str
        result = [_convert_argument_type(v, item_type) for v in value]
    elif origin is dict:
        result = value
    elif origin is Union:
        args = get_args(hint)
        non_none_types = [a for a in args if a is not type(None)]
        result = _convert_argument_type(value, non_none_types[0]) if len(non_none_types) == 1 else value
    elif isinstance(hint, type) and BaseModel and issubclass(hint, BaseModel):
        if isinstance(value, dict):
            fields = hint.model_fields
            converted_data = {k: _convert_argument_type(v, fields[k].annotation) if k in fields else v for k, v in value.items()}
            result = hint(**converted_data)
        else:
            result = value
    elif dataclasses.is_dataclass(hint):
        if isinstance(value, dict):
            field_types = {f.name: f.type for f in dataclasses.fields(hint)}
            converted_data = {k: _convert_argument_type(v, field_types.get(k, type(v))) for k, v in value.items()}
            result = hint(**converted_data)
        else:
            result = value

    return result


def _is_async_function(func: Callable) -> bool:
    """Check if a function is asynchronous."""
    return inspect.iscoroutinefunction(func)


class Funcall:
    """Handler for function calling in LLM interactions."""

    def __init__(self, functions: list[Callable] | None = None) -> None:
        """
        Initialize the function call handler.

        Args:
            functions: List of functions to register
        """
        self.functions = functions or []
        self.function_registry = {func.__name__: func for func in self.functions}

    def get_tools(self, target: Literal["openai", "litellm"] = "openai") -> list[FunctionToolParam]:
        """
        Get tool definitions for the specified target platform.

        Args:
            target: Target platform ("openai" or "litellm")

        Returns:
            List of function tool parameters
        """
        return [generate_function_metadata(func, target) for func in self.functions]

    def _prepare_function_execution(
        self,
        func_name: str,
        args: str,
        context: object = None,
    ) -> tuple[Callable, dict]:
        """
        Prepare function call arguments and context injection.

        Args:
            func_name: Name of the function to call
            args: JSON string of function arguments
            context: Context object to inject

        Returns:
            Tuple of (function, prepared_kwargs)
        """
        if func_name not in self.function_registry:
            msg = f"Function {func_name} not found"
            raise ValueError(msg)

        func = self.function_registry[func_name]
        signature = inspect.signature(func)
        type_hints = get_type_hints(func)
        arguments = json.loads(args)

        # Find non-context parameters
        non_context_params = [name for name in signature.parameters if not _is_context_type(type_hints.get(name, str))]

        # Handle single parameter case
        if len(non_context_params) == 1 and (not isinstance(arguments, dict) or set(arguments.keys()) != set(non_context_params)):
            arguments = {non_context_params[0]: arguments}

        # Prepare final kwargs with type conversion and context injection
        prepared_kwargs = {}
        for param_name in signature.parameters:
            hint = type_hints.get(param_name, str)

            if _is_context_type(hint):
                prepared_kwargs[param_name] = context
            elif param_name in arguments:
                prepared_kwargs[param_name] = _convert_argument_type(arguments[param_name], hint)

        return func, prepared_kwargs

    def _execute_sync_in_async_context(self, func: Callable, kwargs: dict) -> object:
        """Execute synchronous function in async context safely."""
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # If already in event loop, use thread pool
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(func, **kwargs)
                    return future.result()
            else:
                return loop.run_until_complete(func(**kwargs))
        except RuntimeError:
            # No event loop exists, create new one
            return asyncio.run(func(**kwargs))

    def call_function(
        self,
        name: str,
        arguments: str,
        context: object = None,
    ) -> object:
        """
        Call a function by name with JSON arguments synchronously.

        Args:
            name: Name of the function to call
            arguments: JSON string of function arguments
            context: Context object to inject (optional)

        Returns:
            Function execution result

        Raises:
            ValueError: If function is not found
            json.JSONDecodeError: If arguments are not valid JSON
        """
        func, kwargs = self._prepare_function_execution(name, arguments, context)

        if _is_async_function(func):
            logger.warning(
                "Function %s is async but being called synchronously. Consider using call_function_async.",
                name,
            )
            return self._execute_sync_in_async_context(func, kwargs)

        return func(**kwargs)

    async def call_function_async(
        self,
        name: str,
        arguments: str,
        context: object = None,
    ) -> object:
        """
        Call a function by name with JSON arguments asynchronously.

        Args:
            name: Name of the function to call
            arguments: JSON string of function arguments
            context: Context object to inject (optional)

        Returns:
            Function execution result

        Raises:
            ValueError: If function is not found
            json.JSONDecodeError: If arguments are not valid JSON
        """
        func, kwargs = self._prepare_function_execution(name, arguments, context)

        if _is_async_function(func):
            return await func(**kwargs)

        # Run sync function in thread pool to avoid blocking event loop
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, lambda: func(**kwargs))

    def handle_openai_function_call(
        self,
        call: ResponseFunctionToolCall,
        context: object = None,
    ) -> object:
        """
        Handle OpenAI function call synchronously.

        Args:
            call: OpenAI function tool call
            context: Context object to inject

        Returns:
            Function execution result
        """
        if not isinstance(call, ResponseFunctionToolCall):
            msg = "call must be an instance of ResponseFunctionToolCall"
            raise TypeError(msg)

        return self.call_function(call.name, call.arguments, context)

    async def handle_openai_function_call_async(
        self,
        call: ResponseFunctionToolCall,
        context: object = None,
    ) -> object:
        """
        Handle OpenAI function call asynchronously.

        Args:
            call: OpenAI function tool call
            context: Context object to inject

        Returns:
            Function execution result
        """
        if not isinstance(call, ResponseFunctionToolCall):
            msg = "call must be an instance of ResponseFunctionToolCall"
            raise TypeError(msg)

        return await self.call_function_async(call.name, call.arguments, context)

    def handle_litellm_function_call(
        self,
        call: litellm.ChatCompletionMessageToolCall,
        context: object = None,
    ) -> object:
        """
        Handle LiteLLM function call synchronously.

        Args:
            call: LiteLLM function tool call
            context: Context object to inject

        Returns:
            Function execution result
        """
        if not isinstance(call, litellm.ChatCompletionMessageToolCall):
            msg = "call must be an instance of litellm.ChatCompletionMessageToolCall"
            raise TypeError(msg)

        return self.call_function(
            call.function.name,
            call.function.arguments,
            context,
        )

    async def handle_litellm_function_call_async(
        self,
        call: litellm.ChatCompletionMessageToolCall,
        context: object = None,
    ) -> object:
        """
        Handle LiteLLM function call asynchronously.

        Args:
            call: LiteLLM function tool call
            context: Context object to inject

        Returns:
            Function execution result
        """
        if not isinstance(call, litellm.ChatCompletionMessageToolCall):
            msg = "call must be an instance of litellm.ChatCompletionMessageToolCall"
            raise TypeError(msg)

        return await self.call_function_async(
            call.function.name,
            call.function.arguments,
            context,
        )

    def handle_function_call(
        self,
        call: ResponseFunctionToolCall | litellm.ChatCompletionMessageToolCall,
        context: object = None,
    ) -> object:
        """
        Handle function call synchronously (unified interface).

        Args:
            call: Function tool call (OpenAI or LiteLLM)
            context: Context object to inject

        Returns:
            Function execution result
        """
        if isinstance(call, ResponseFunctionToolCall):
            return self.handle_openai_function_call(call, context)
        if isinstance(call, litellm.ChatCompletionMessageToolCall):
            return self.handle_litellm_function_call(call, context)
        msg = "call must be an instance of ResponseFunctionToolCall or litellm.ChatCompletionMessageToolCall"
        raise TypeError(msg)

    async def handle_function_call_async(
        self,
        call: ResponseFunctionToolCall | litellm.ChatCompletionMessageToolCall,
        context: object = None,
    ) -> object:
        """
        Handle function call asynchronously (unified interface).

        Args:
            call: Function tool call (OpenAI or LiteLLM)
            context: Context object to inject

        Returns:
            Function execution result
        """
        if isinstance(call, ResponseFunctionToolCall):
            return await self.handle_openai_function_call_async(call, context)
        if isinstance(call, litellm.ChatCompletionMessageToolCall):
            return await self.handle_litellm_function_call_async(call, context)
        msg = "call must be an instance of ResponseFunctionToolCall or litellm.ChatCompletionMessageToolCall"
        raise TypeError(msg)


# 兼容旧接口
generate_meta = generate_function_metadata


__all__ = ["Context", "Funcall", "generate_function_metadata"]
