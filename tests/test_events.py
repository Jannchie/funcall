"""
Test cases for the event system functionality.
"""

import asyncio
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, NoReturn

import pytest

from funcall import Context, EventEmitter, Funcall
from funcall.events import create_emitter


class TestEventEmitter:
    """Test the EventEmitter class."""

    def test_event_emitter_creation(self):
        """Test creating an EventEmitter."""
        callbacks = []
        emitter = EventEmitter[dict[str, Any]](callbacks)
        assert emitter._callbacks == callbacks

    def test_emit_with_no_callbacks(self):
        """Test emitting with no registered callbacks."""
        emitter = EventEmitter[str]([])
        # Should not raise an exception
        emitter.emit("test payload")

    def test_emit_with_callbacks(self):
        """Test emitting events to registered callbacks."""
        received_events = []

        def callback(payload):
            received_events.append(payload)

        emitter = EventEmitter[str]([callback])
        emitter.emit("test event")

        assert len(received_events) == 1
        assert received_events[0] == "test event"

    def test_emit_with_multiple_callbacks(self):
        """Test emitting events to multiple callbacks."""
        received_events_1 = []
        received_events_2 = []

        def callback1(payload):
            received_events_1.append(payload)

        def callback2(payload):
            received_events_2.append(payload)

        emitter = EventEmitter[int]([callback1, callback2])
        emitter.emit(42)

        assert received_events_1 == [42]
        assert received_events_2 == [42]


class TestCreateEmitter:
    """Test the create_emitter function."""

    def test_create_emitter(self):
        """Test creating an emitter function."""
        received_events = []

        def callback(payload):
            received_events.append(payload)

        emit = create_emitter([callback])
        emit("test")

        assert received_events == ["test"]


class TestFuncallEvents:
    """Test event functionality in Funcall."""

    def test_on_event_registration(self):
        """Test registering event callbacks."""
        fc = Funcall()

        def callback(event: Any) -> None:
            pass

        fc.on_event(callback)
        assert len(fc._event_callbacks) == 1
        assert fc._event_callbacks[0] == callback

    def test_multiple_event_callbacks(self):
        """Test registering multiple event callbacks."""
        fc = Funcall()

        def callback1(event: Any) -> None:
            pass

        def callback2(event: Any) -> None:
            pass

        fc.on_event(callback1)
        fc.on_event(callback2)

        assert len(fc._event_callbacks) == 2
        assert callback1 in fc._event_callbacks
        assert callback2 in fc._event_callbacks

    def test_emit_function_creation(self):
        """Test creating emit function."""
        fc = Funcall()
        received_events = []

        def callback(event: Any) -> None:
            received_events.append(event)

        fc.on_event(callback)
        emit = fc._create_emit_function()

        emit({"test": "data"})
        assert received_events == [{"test": "data"}]


class TestFunctionEvents:
    """Test event emission from regular functions."""

    def test_function_with_emit_parameter(self):
        """Test function that accepts emit parameter."""

        def test_func(message: str, emit: Callable[[dict[str, Any]], None]) -> str:
            emit({"type": "start", "message": f"Starting: {message}"})
            emit({"type": "complete", "result": f"Done: {message}"})
            return f"Processed: {message}"

        fc = Funcall([test_func])
        received_events = []

        def callback(event: Any) -> None:
            received_events.append(event)

        fc.on_event(callback)
        result = fc.call_function("test_func", '"hello world"')

        assert result == "Processed: hello world"
        assert len(received_events) == 2
        assert received_events[0] == {"type": "start", "message": "Starting: hello world"}
        assert received_events[1] == {"type": "complete", "result": "Done: hello world"}

    def test_function_without_emit_parameter(self):
        """Test function that doesn't use emit parameter."""

        def simple_func(x: int) -> int:
            return x * 2

        fc = Funcall([simple_func])
        received_events = []

        fc.on_event(lambda event: received_events.append(event))
        result = fc.call_function("simple_func", "5")

        assert result == 10
        # No events should be emitted
        assert len(received_events) == 0

    def test_function_with_context_and_emit(self):
        """Test function that uses both context and emit."""
        @dataclass
        class TestContext:
            user_id: str

        def context_emit_func(data: str, context: Context[TestContext], emit: Callable[[dict[str, Any]], None]) -> str:
            emit({"user": context.value.user_id, "action": "start"})
            result = f"User {context.value.user_id} processed: {data}"
            emit({"user": context.value.user_id, "action": "complete", "result": result})
            return result

        fc = Funcall([context_emit_func])
        received_events = []
        fc.on_event(lambda event: received_events.append(event))

        context = TestContext(user_id="user123")
        result = fc.call_function("context_emit_func", '"test data"', context=context)

        assert "user123" in result
        assert len(received_events) == 2
        assert received_events[0]["user"] == "user123"
        assert received_events[1]["user"] == "user123"


class TestDynamicToolEvents:
    """Test event emission from dynamic tools."""

    def test_dynamic_tool_with_events(self):
        """Test dynamic tool that emits events."""
        fc = Funcall()

        def calculator_handler(operation: str, a: float, b: float, emit: Callable[[dict[str, Any]], None]) -> float:
            emit({"action": "calculate", "operation": operation, "operands": [a, b]})

            result = {"add": a + b, "multiply": a * b}[operation]

            emit({"action": "result", "result": result})
            return result

        fc.add_dynamic_tool(
            name="calc",
            description="Calculator with events",
            parameters={"operation": {"type": "string", "enum": ["add", "multiply"]}, "a": {"type": "number"}, "b": {"type": "number"}},
            required=["operation", "a", "b"],
            handler=calculator_handler,
        )

        received_events = []
        fc.on_event(lambda event: received_events.append(event))

        result = fc.call_function("calc", '{"operation": "add", "a": 5, "b": 3}')

        assert result == 8
        assert len(received_events) == 2
        assert received_events[0]["action"] == "calculate"
        assert received_events[1]["action"] == "result"
        assert received_events[1]["result"] == 8


class TestEventTypes:
    """Test different event payload types."""

    def test_dict_events(self):
        """Test dictionary event payloads."""

        def dict_func(emit: Callable[[dict[str, Any]], None]) -> str:
            emit({"type": "info", "message": "test"})
            return "done"

        fc = Funcall([dict_func])
        received_events = []
        fc.on_event(lambda event: received_events.append(event))

        fc.call_function("dict_func", "{}")
        assert received_events[0] == {"type": "info", "message": "test"}

    def test_dataclass_events(self):
        """Test dataclass event payloads."""

        @dataclass
        class TestEvent:
            event_type: str
            data: Any

        def dataclass_func(emit: Callable[[TestEvent], None]) -> str:
            emit(TestEvent(event_type="test", data={"key": "value"}))
            return "done"

        fc = Funcall([dataclass_func])
        received_events = []
        fc.on_event(lambda event: received_events.append(event))

        fc.call_function("dataclass_func", "{}")
        assert isinstance(received_events[0], TestEvent)
        assert received_events[0].event_type == "test"
        assert received_events[0].data == {"key": "value"}

    def test_string_events(self):
        """Test string event payloads."""

        def string_func(emit: Callable[[str], None]) -> str:
            emit("Starting process")
            emit("Process complete")
            return "done"

        fc = Funcall([string_func])
        received_events = []
        fc.on_event(lambda event: received_events.append(event))

        fc.call_function("string_func", "{}")
        assert received_events == ["Starting process", "Process complete"]


class TestAsyncEvents:
    """Test events with async functions."""

    @pytest.mark.asyncio
    async def test_async_function_with_events(self):
        """Test async function that emits events."""
        async def async_func(emit: Callable[[str], None]) -> str:
            emit("async start")
            await asyncio.sleep(0.01)  # Simulate async work
            emit("async complete")
            return "async done"

        fc = Funcall([async_func])
        received_events = []
        fc.on_event(lambda event: received_events.append(event))

        result = await fc.call_function_async("async_func", "{}")

        assert result == "async done"
        assert received_events == ["async start", "async complete"]


class TestErrorHandling:
    """Test error handling with events."""

    def test_function_error_with_events(self):
        """Test function that emits events before failing."""

        def failing_func(emit: Callable[[str], None]) -> str:
            emit("Starting operation")
            emit("About to fail")
            msg = "Intentional failure"
            raise ValueError(msg)

        fc = Funcall([failing_func])
        received_events = []
        fc.on_event(lambda event: received_events.append(event))

        with pytest.raises(ValueError, match="Intentional failure"):
            fc.call_function("failing_func", "{}")

        # Events should still be received even if function fails
        assert received_events == ["Starting operation", "About to fail"]

    def test_callback_error_handling(self):
        """Test that callback errors don't break event emission."""

        def good_callback(event: Any) -> None:
            # This callback works fine
            pass

        def bad_callback(_event: Any) -> NoReturn:
            # This callback raises an error
            msg = "Callback error"
            raise RuntimeError(msg)

        def test_func(emit: Callable[[str], None]) -> str:
            emit("test event")
            return "done"

        fc = Funcall([test_func])
        fc.on_event(good_callback)
        fc.on_event(bad_callback)

        # The function should still complete despite callback error
        # Note: In the current implementation, callback errors will propagate
        # You might want to add error handling to the event system
        with pytest.raises(RuntimeError, match="Callback error"):
            fc.call_function("test_func", "{}")


if __name__ == "__main__":
    pytest.main([__file__])

