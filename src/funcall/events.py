"""
events.py

Event system for function calls with generic payload support.
"""

from collections.abc import Callable
from typing import Any, Generic, TypeVar

T = TypeVar("T")


class EventEmitter(Generic[T]):
    """Generic event emitter that accepts typed payloads."""

    def __init__(self, callbacks: list[Callable[[T], None]]) -> None:
        """
        Initialize the event emitter.

        Args:
            callbacks: List of callback functions to invoke when events are emitted
        """
        self._callbacks = callbacks

    def emit(self, payload: T) -> None:
        """
        Emit an event with the given payload.

        Args:
            payload: The event payload data
        """
        for callback in self._callbacks:
            callback(payload)


def create_emitter(callbacks: list[Callable[[Any], None]]) -> Callable[[Any], None]:
    """
    Create a simple emit function.

    Args:
        callbacks: List of callback functions

    Returns:
        An emit function that calls all callbacks with the payload
    """
    def emit(payload: Any) -> None:
        for callback in callbacks:
            callback(payload)
    return emit
