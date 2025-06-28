from typing import Generic, TypedDict, TypeVar, Union, get_args

from openai.types.chat import ChatCompletionToolParam

T = TypeVar("T")


class Context(Generic[T]):
    """Generic context container for dependency injection in function calls."""

    def __init__(self, value: T | None = None) -> None:
        self.value = value


CompletionFunctionToolParam = ChatCompletionToolParam


class ToolMeta(TypedDict):
    require_confirm: bool
    return_direct: bool


def is_context_type(hint: type) -> bool:
    return getattr(hint, "__origin__", None) is Context or hint is Context


def is_optional_type(hint: type) -> bool:
    origin = getattr(hint, "__origin__", None)
    if origin is Union:
        args = get_args(hint)
        return any(a is type(None) for a in args)
    return False
