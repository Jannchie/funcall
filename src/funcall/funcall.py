import asyncio
import concurrent.futures
import inspect
import json
from collections.abc import Callable
from logging import getLogger
from typing import Any, Literal, Union, get_type_hints, overload

import litellm
from openai.types.responses import FunctionToolParam as ResponseFunctionToolParam
from openai.types.responses import ResponseFunctionToolCall
from openai.types.shared_params import FunctionDefinition
from pydantic import BaseModel

from funcall.decorators import ToolWrapper
from funcall.events import create_emitter
from funcall.types import CompletionFunctionToolParam, Context, ToolMeta, is_context_type

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
            converted_data = {k: _convert_argument_type(v, fields[k].annotation) if k in fields and fields[k].annotation is not None else v for k, v in value.items()}  # type: ignore
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
        self.dynamic_tools: dict[str, dict[str, Any]] = {}
        self._event_callbacks: list[Callable[[Any], None]] = []

    def add_dynamic_tool(
        self,
        name: str,
        description: str,
        parameters: dict[str, Any],
        *,
        required: list[str] | None = None,
        handler: Callable[..., Any] | None = None,
    ) -> None:
        """
        Add a dynamic tool by specifying its metadata directly.

        Args:
            name: Tool name
            description: Tool description
            parameters: Parameter schema (JSON Schema properties format)
            required: List of required parameter names
            handler: Optional function to handle the tool call
        """

        # Create a dynamic function that mimics a real function
        def dynamic_func(**kwargs: object) -> object:
            if handler:
                return handler(**kwargs)
            # Default behavior: return the call information
            return {
                "tool": name,
                "arguments": kwargs,
                "message": f"Tool '{name}' called with arguments: {kwargs}",
            }

        # Set function metadata for the dynamic function
        dynamic_func.__name__ = name
        dynamic_func.__doc__ = description

        # Store the tool metadata
        self.dynamic_tools[name] = {
            "name": name,
            "description": description,
            "parameters": parameters,
            "required": required or [],
            "handler": handler,
            "function": dynamic_func,
        }

        # Add to function registry
        self.function_registry[name] = dynamic_func

    def remove_dynamic_tool(self, name: str) -> None:
        """
        Remove a dynamic tool by name.

        Args:
            name: Name of the tool to remove
        """
        if name in self.dynamic_tools:
            del self.dynamic_tools[name]
            if name in self.function_registry:
                del self.function_registry[name]

    @overload
    def get_tools(self, target: Literal["response"] = "response") -> list[ResponseFunctionToolParam]: ...

    @overload
    def get_tools(self, target: Literal["completion"]) -> list[CompletionFunctionToolParam]: ...

    def get_tools(self, target: Literal["response", "completion"] = "response") -> list[ResponseFunctionToolParam] | list[CompletionFunctionToolParam]:
        """
        Get tool definitions for the specified target platform.

        Args:
            target: Target api ("response" or "completion")

        Returns:
            List of function tool parameters
        """
        # Add regular function tools
        if target == "completion":
            tools: list[CompletionFunctionToolParam] = [generate_function_metadata(func, target) for func in self.functions]  # type: ignore
            # Add dynamic tools
            for tool_info in self.dynamic_tools.values():
                dynamic_tool: CompletionFunctionToolParam = {
                    "type": "function",
                    "function": FunctionDefinition(
                        name=tool_info["name"],
                        description=tool_info["description"],
                        parameters={
                            "type": "object",
                            "properties": tool_info["parameters"],
                            "required": tool_info["required"],
                        },
                    ),
                }
                tools.append(dynamic_tool)
            return tools

        # response format
        tools_response: list[ResponseFunctionToolParam] = [generate_function_metadata(func, target) for func in self.functions]  # type: ignore
        # Add dynamic tools
        for tool_info in self.dynamic_tools.values():
            dynamic_tool_response: ResponseFunctionToolParam = {
                "type": "function",
                "name": tool_info["name"],
                "description": tool_info["description"],
                "parameters": {
                    "type": "object",
                    "properties": tool_info["parameters"],
                    "required": tool_info["required"],
                    "additionalProperties": False,
                },
                "strict": True,
            }
            tools_response.append(dynamic_tool_response)
        return tools_response

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
        arguments = json.loads(args)

        # Check if this is a dynamic tool
        if func_name in self.dynamic_tools:
            # For dynamic tools, we pass arguments as-is since they don't have type hints
            prepared_kwargs = arguments if isinstance(arguments, dict) else {"value": arguments}

            # Check if the handler accepts emit parameter
            tool_info = self.dynamic_tools[func_name]
            handler = tool_info.get("handler")
            if handler and "emit" in inspect.signature(handler).parameters:
                prepared_kwargs["emit"] = self._create_emit_function()

            return func, prepared_kwargs

        # Handle regular functions
        signature = inspect.signature(func)
        type_hints = get_type_hints(func)

        # Find parameters that are not Context or emit function
        emit_param_name = None
        non_context_non_emit_params = []

        for param_name in signature.parameters:
            hint = type_hints.get(param_name, str)
            if is_context_type(hint):
                continue
            if param_name == "emit" or (hasattr(hint, "__origin__") and str(hint).startswith("typing.Callable")):
                emit_param_name = param_name
            else:
                non_context_non_emit_params.append(param_name)

        # Handle single parameter case (excluding context and emit)
        if len(non_context_non_emit_params) == 1 and (not isinstance(arguments, dict) or set(arguments.keys()) != set(non_context_non_emit_params)):
            arguments = {non_context_non_emit_params[0]: arguments}

        # Prepare final kwargs with type conversion, context injection, and emit injection
        prepared_kwargs = {}
        for param_name in signature.parameters:
            hint = type_hints.get(param_name, str)

            if is_context_type(hint):
                prepared_kwargs[param_name] = context
            elif param_name == emit_param_name:
                # Inject emit function
                prepared_kwargs[param_name] = self._create_emit_function()
            elif param_name in arguments:  # type: ignore
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
        # if context is not Context, wrap it
        if context is not None and not is_context_type(type(context)):
            context = Context(context)

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
        # if context is not Context, wrap it
        if context is not None and not is_context_type(type(context)):
            context = Context(context)

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

        # if context is not Context, wrap it
        if context is not None and not is_context_type(type(context)):
            context = Context(context)

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
        # For dynamic tools, always return default metadata
        return ToolMeta(
            require_confirm=False,
            return_direct=False,
        )

    def add_function(self, func: Callable) -> None:
        """
        Add a function as a tool after initialization.

        Args:
            func: The function to add as a tool

        Raises:
            ValueError: If a function with the same name already exists
        """
        func_name = func.__name__
        if func_name in self.function_registry:
            msg = f"Function '{func_name}' already exists"
            raise ValueError(msg)

        self.functions.append(func)
        self.function_registry[func_name] = func

    def remove_function(self, name: str) -> None:
        """
        Remove a function tool by name.

        Args:
            name: Name of the function to remove

        Raises:
            ValueError: If the function is not found or is a dynamic tool
        """
        if name not in self.function_registry:
            msg = f"Function '{name}' not found"
            raise ValueError(msg)

        # Check if it's a dynamic tool (should use remove_dynamic_tool instead)
        if name in self.dynamic_tools:
            msg = f"'{name}' is a dynamic tool. Use remove_dynamic_tool() instead"
            raise ValueError(msg)

        # Find and remove from functions list
        func_to_remove = self.function_registry[name]
        self.functions = [f for f in self.functions if f is not func_to_remove]

        # Remove from registry
        del self.function_registry[name]

    def remove_function_by_callable(self, func: Callable) -> None:
        """
        Remove a function tool by its callable reference.

        Args:
            func: The function to remove

        Raises:
            ValueError: If the function is not found
        """
        func_name = func.__name__

        # Check if the function is registered and is the same object
        if func_name not in self.function_registry:
            msg = f"Function '{func_name}' not found"
            raise ValueError(msg)

        if self.function_registry[func_name] is not func:
            msg = f"Function '{func_name}' found but is not the same object"
            raise ValueError(msg)

        # Check if it's a dynamic tool
        if func_name in self.dynamic_tools:
            msg = f"'{func_name}' is a dynamic tool. Use remove_dynamic_tool() instead"
            raise ValueError(msg)

        # Remove from functions list
        self.functions = [f for f in self.functions if f is not func]

        # Remove from registry
        del self.function_registry[func_name]

    def has_function(self, name: str) -> bool:
        """
        Check if a function tool exists by name.

        Args:
            name: Name of the function to check

        Returns:
            True if the function exists, False otherwise
        """
        return name in self.function_registry

    def list_functions(self) -> list[str]:
        """
        Get a list of all registered function names.

        Returns:
            List of function names (both regular functions and dynamic tools)
        """
        return list(self.function_registry.keys())

    def list_regular_functions(self) -> list[str]:
        """
        Get a list of regular function names (excluding dynamic tools).

        Returns:
            List of regular function names
        """
        return [func.__name__ for func in self.functions]

    def list_dynamic_tools(self) -> list[str]:
        """
        Get a list of dynamic tool names.

        Returns:
            List of dynamic tool names
        """
        return list(self.dynamic_tools.keys())

    def on_event(self, callback: Callable[[Any], None]) -> None:
        """
        Register an event callback function.

        Args:
            callback: Function to call when events are emitted
        """
        self._event_callbacks.append(callback)

    def _create_emit_function(self) -> Callable[[Any], None]:
        """
        Create an emit function for event broadcasting.

        Returns:
            Emit function that broadcasts events to all registered callbacks
        """
        return create_emitter(self._event_callbacks)

    # ...existing code...
