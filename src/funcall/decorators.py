"""
decorators.py

This module contains custom decorators for the funcall package.
"""

import asyncio
import functools
from collections.abc import Awaitable, Callable
from typing import Generic, ParamSpec, TypeVar, cast

P = ParamSpec("P")
R = TypeVar("R")


class ToolWrapper(Generic[P, R]):
    def __init__(self, func: Callable[P, R], *, require_confirm: bool = False, return_direct: bool = False) -> None:
        functools.update_wrapper(self, func)
        self._func = func
        self.require_confirm = require_confirm
        self.return_direct = return_direct
        self.is_async = asyncio.iscoroutinefunction(func)

    def __call__(self, *args: P.args, **kwargs: P.kwargs) -> R:
        if self.is_async:
            msg = "This tool function is async, use 'await' to call it."
            raise RuntimeError(msg)
        return self._func(*args, **kwargs)

    async def acall(self, *args: P.args, **kwargs: P.kwargs) -> R:
        if self.is_async:
            return await cast("Callable[P, Awaitable[R]]", self._func)(*args, **kwargs)
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, functools.partial(self._func, *args, **kwargs))


def tool(
    func: Callable[P, R] | None = None,
    *,
    return_immediately: bool = False,
    require_confirmation: bool = False,
) -> Callable[[Callable[P, R]], ToolWrapper[P, R]] | ToolWrapper[P, R]:
    """
    Decorator: Mark a function as a tool, specifying if it should return immediately and/or require human confirmation.

    Can be used as @tool or @tool(...)
    """

    def decorator(f: Callable[P, R]) -> ToolWrapper[P, R]:
        return ToolWrapper(f, return_direct=return_immediately, require_confirm=require_confirmation)

    if func is not None:
        return decorator(func)
    return decorator
