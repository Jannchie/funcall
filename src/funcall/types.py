from typing import Generic, Literal, Required, TypedDict, TypeVar, Union, get_args

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
