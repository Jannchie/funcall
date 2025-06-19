import dataclasses
import inspect
from asyncio.log import logger
from collections.abc import Callable
from typing import Literal, get_type_hints

from openai.types.responses import FunctionToolParam
from pydantic import BaseModel

from funcall.params_to_schema import params_to_schema

from .types import LiteLLMFunctionToolParam, is_context_type, is_optional_type


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
        if is_context_type(hint):
            context_count += 1
            continue

        param_names.append(name)
        param_types.append(hint)

    return param_names, param_types, context_count


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
                    is_optional = is_optional_type(ann) or model_fields[k].is_required is False
                else:
                    ann = model_fields[k].type
                    is_optional = is_optional_type(ann) or (getattr(model_fields[k], "default", dataclasses.MISSING) is not dataclasses.MISSING)
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
            is_optional = is_optional_type(hint) or (param.default != inspect.Parameter.empty)
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
