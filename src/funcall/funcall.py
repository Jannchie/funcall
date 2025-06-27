import asyncio
import concurrent.futures
import inspect
import json
from collections.abc import Callable
from logging import getLogger
from typing import Literal, Union, get_type_hints

import litellm
from openai.types.responses import (
    FunctionToolParam,
    ResponseFunctionToolCall,
)
from pydantic import BaseModel

from funcall.decorators import ToolWrapper
from funcall.types import ToolMeta, is_context_type

from .metadata import generate_function_metadata


def _convert_argument_type(value: list, hint: type) -> object:
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
    if origin in (list, set, tuple):
        args = getattr(hint, "__args__", [])
        item_type = args[0] if args else str
        result = [_convert_argument_type(v, item_type) for v in value]
    elif origin is dict:
        result = value
    elif getattr(hint, "__origin__", None) is Union:
        args = getattr(hint, "__args__", [])
        non_none_types = [a for a in args if a is not type(None)]
        result = _convert_argument_type(value, non_none_types[0]) if len(non_none_types) == 1 else value
    elif isinstance(hint, type) and BaseModel and issubclass(hint, BaseModel):
        if isinstance(value, dict):
            fields = hint.model_fields
            converted_data = {k: _convert_argument_type(v, fields[k].annotation) if k in fields else v for k, v in value.items()}  # type: ignore
            result = hint(**converted_data)
        else:
            result = value
    elif hasattr(hint, "__dataclass_fields__"):
        if isinstance(value, dict):
            field_types = {f: t.type for f, t in hint.__dataclass_fields__.items()}
            converted_data = {k: _convert_argument_type(v, field_types.get(k, type(v))) for k, v in value.items()}
            result = hint(**converted_data)
        else:
            result = value
    return result


def _is_async_function(func: object) -> bool:
    """Check if a function is asynchronous."""
    return inspect.iscoroutinefunction(func)


logger = getLogger("funcall")


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

    def get_tools(self, target: Literal["response", "completion"] = "response") -> list[FunctionToolParam]:
        """
        Get tool definitions for the specified target platform.

        Args:
            target: Target api ("response" or "completion")

        Returns:
            List of function tool parameters
        """
        return [generate_function_metadata(func, target) for func in self.functions]  # type: ignore

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
        non_context_params = [name for name in signature.parameters if not is_context_type(type_hints.get(name, str))]

        # Handle single parameter case
        if len(non_context_params) == 1 and (not isinstance(arguments, dict) or set(arguments.keys()) != set(non_context_params)):
            arguments = {non_context_params[0]: arguments}

        # Prepare final kwargs with type conversion and context injection
        prepared_kwargs = {}
        for param_name in signature.parameters:
            hint = type_hints.get(param_name, str)

            if is_context_type(hint):
                prepared_kwargs[param_name] = context
            elif param_name in arguments:
                prepared_kwargs[param_name] = _convert_argument_type(arguments[param_name], hint)  # type: ignore

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

        if isinstance(func, ToolWrapper):
            if func.is_async:
                logger.warning(
                    "Function %s is async but being called synchronously. Consider using call_function_async.",
                    name,
                )
                return self._execute_sync_in_async_context(func, kwargs)
            return func(**kwargs)

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
        if isinstance(func, ToolWrapper):
            if func.is_async:
                return await func.acall(**kwargs)
            # Run sync function in thread pool to avoid blocking event loop
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(None, lambda: func(**kwargs))

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
        if not call.function:
            msg = "call.function must not be None"
            raise ValueError(msg)
        if not call.function.name:
            msg = "call.function.name must not be empty"
            raise ValueError(msg)
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
        if not call.function:
            msg = "call.function must not be None"
            raise ValueError(msg)
        if not call.function.name:
            msg = "call.function.name must not be empty"
            raise ValueError(msg)
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

    def get_tool_meta(self, name: str) -> ToolMeta:
        """
        Get metadata for a registered function by name.

        Args:
            name: Name of the function

        Returns:
            Function metadata dictionary
        """
        if name not in self.function_registry:
            msg = f"Function {name} not found"
            raise ValueError(msg)

        func = self.function_registry[name]
        if isinstance(func, ToolWrapper):
            return ToolMeta(
                require_confirm=func.require_confirm,
                return_direct=func.return_direct,
            )
        return ToolMeta(
            require_confirm=False,
            return_direct=False,
        )
