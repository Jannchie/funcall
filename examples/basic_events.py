"""
Example: Basic Event Usage

This example demonstrates the basic usage of event system in funcall.
Functions can emit events to communicate with parent components.
"""

import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from funcall import Funcall


# Example 1: Simple dictionary payloads
def simple_task(message: str, emit: Callable[[dict[str, Any]], None]) -> str:
    """A simple task that emits progress events using dictionary payloads."""
    emit({"type": "start", "message": f"Starting task: {message}"})

    # Simulate some work
    for i in range(3):
        emit({
            "type": "progress",
            "step": f"Step {i + 1}",
            "percent": (i + 1) * 33,
            "details": f"Processing step {i + 1} of 3",
        })
        time.sleep(0.1)  # Simulate work

    emit({"type": "complete", "result": f"Completed: {message}"})
    return f"Task '{message}' finished"


# Example 2: Typed events with dataclasses
@dataclass
class TaskEvent:
    event_type: str
    message: str
    progress: int = 0
    error: str = ""


def typed_task(data: dict[str, Any], emit: Callable[[TaskEvent], None]) -> dict[str, Any]:
    """A task that emits strongly typed events."""
    emit(TaskEvent(event_type="start", message="Beginning data processing"))

    try:
        # Validation step
        emit(TaskEvent(event_type="progress", message="Validating input", progress=25))
        if not data.get("valid", True):
            emit(TaskEvent(event_type="error", message="Invalid input data", error="VALIDATION_ERROR"))
            msg = "Invalid data"
            raise ValueError(msg)

        # Processing step
        emit(TaskEvent(event_type="progress", message="Processing data", progress=75))
        result = {"processed": True, "original": data, "result": "success"}

        # Complete
        emit(TaskEvent(event_type="complete", message="Processing finished", progress=100))
        return result

    except Exception as e:
        emit(TaskEvent(event_type="error", message=f"Task failed: {e!s}", error=str(e)))
        raise


# Example 3: API call with events
def api_call(url: str, emit: Callable[[dict[str, Any]], None]) -> dict[str, Any]:
    """Simulate API call with detailed event reporting."""

    start_time = time.time()
    emit({
        "action": "request_start",
        "url": url,
        "timestamp": start_time,
        "method": "GET",
    })

    try:
        # Simulate network delay
        time.sleep(0.2)

        # Simulate potential failure
        if "error" in url.lower():
            msg = "Simulated API error"
            raise Exception(msg)

        # Success response
        response_time = time.time() - start_time
        result = {
            "status": "success",
            "data": {"message": "API call successful", "url": url},
            "status_code": 200,
        }

        emit({
            "action": "request_success",
            "url": url,
            "status_code": 200,
            "response_time": response_time,
            "data_size": len(str(result)),
        })

        return result

    except Exception as e:
        response_time = time.time() - start_time
        emit({
            "action": "request_failed",
            "url": url,
            "error": str(e),
            "response_time": response_time,
        })
        raise


def main():
    """Demonstrate basic event usage."""
    print("=== Basic Events Demo ===\n")

    # Create funcall instance
    fc = Funcall([simple_task, typed_task, api_call])

    # Example 1: Dictionary events
    print("1. Simple Task with Dictionary Events:")

    def handle_simple_events(payload):
        event_type = payload.get("type", "unknown")
        if event_type == "start":
            print(f"   [START] {payload['message']}")
        elif event_type == "progress":
            print(f"   [PROGRESS] {payload['step']}: {payload['percent']}% - {payload['details']}")
        elif event_type == "complete":
            print(f"   [COMPLETE] {payload['result']}")

    fc.on_event(handle_simple_events)
    result = fc.call_function("simple_task", '"Hello World"')
    print(f"   Result: {result}\n")

    # Clear callbacks for next example
    fc._event_callbacks.clear()

    # Example 2: Typed events
    print("2. Typed Task with Dataclass Events:")

    def handle_typed_events(event: TaskEvent):
        if event.event_type == "start":
            print(f"   [START] Started: {event.message}")
        elif event.event_type == "progress":
            print(f"   [PROGRESS] Progress: {event.message} ({event.progress}%)")
        elif event.event_type == "complete":
            print(f"   [COMPLETE] Completed: {event.message}")
        elif event.event_type == "error":
            print(f"   [ERROR] Error: {event.message} - {event.error}")

    fc.on_event(handle_typed_events)
    result = fc.call_function("typed_task", '{"valid": true, "content": "test data"}')
    print(f"   Result: {result}\n")

    # Clear callbacks for next example
    fc._event_callbacks.clear()

    # Example 3: API events
    print("3. API Call with Events:")

    def handle_api_events(payload):
        action = payload.get("action", "unknown")
        if action == "request_start":
            print(f"   [REQUEST] Starting {payload['method']} request to {payload['url']}")
        elif action == "request_success":
            print(f"   [SUCCESS] Success: {payload['status_code']} ({payload['response_time']:.3f}s)")
        elif action == "request_failed":
            print(f"   [FAILED] Failed: {payload['error']} ({payload['response_time']:.3f}s)")

    fc.on_event(handle_api_events)

    # Successful API call
    print("   Successful API call:")
    result = fc.call_function("api_call", '"https://api.example.com/users"')
    print(f"   Result: {result}\n")

    # Failed API call
    print("   Failed API call:")
    try:
        fc.call_function("api_call", '"https://api.error.com/fail"')
    except Exception as e:
        print(f"   Exception caught: {e}\n")

    print("=== Demo Complete ===")


if __name__ == "__main__":
    main()

