"""
Example: Advanced Event Usage

This example demonstrates advanced event patterns including:
- Dynamic tools with events
- Context + events combination
- Stream processing
- Event-driven workflows
"""

import time
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from funcall import Context, Funcall


# Advanced event types
class EventType(Enum):
    PROGRESS = "progress"
    STATUS = "status"
    ERROR = "error"
    DATA = "data"
    COMPLETE = "complete"


@dataclass
class StreamEvent:
    """Structured event for streaming data."""
    event_type: EventType
    data: Any
    chunk_index: int = 0
    total_chunks: int | None = None
    is_final: bool = False
    timestamp: float = field(default_factory=time.time)


@dataclass
class ProcessingContext:
    """Business context for processing tasks."""
    user_id: str
    session_id: str
    config: dict[str, Any] = field(default_factory=dict)


# Example 1: Context + Events combination
def context_aware_task(
    data: dict[str, Any],
    context: Context[ProcessingContext],
    emit: Callable[[dict[str, Any]], None],
) -> dict[str, Any]:
    """Task that uses both context and events."""
    user_id = context.value.user_id
    session_id = context.value.session_id

    emit({
        "type": "session_start",
        "user_id": user_id,
        "session_id": session_id,
        "data_size": len(data),
    })

    # User-specific processing
    if context.value.config.get("detailed_logging", False):
        emit({
            "type": "debug",
            "message": f"User {user_id} processing data: {list(data.keys())}",
        })

    result = {
        "processed_by": user_id,
        "session": session_id,
        "result": "success",
        "data": data,
    }

    emit({
        "type": "session_complete",
        "user_id": user_id,
        "session_id": session_id,
        "result": result,
    })

    return result


# Example 2: Streaming data processor
def streaming_processor(
    query: str,
    emit: Callable[[StreamEvent], None],
) -> str:
    """Process data in chunks, emitting streaming events."""
    chunks = ["Analyzing", "Processing", "Synthesizing", "Finalizing"]
    accumulated_result = ""

    emit(StreamEvent(
        event_type=EventType.STATUS,
        data={"message": f"Starting stream processing for: {query}"},
        total_chunks=len(chunks),
    ))

    for i, chunk in enumerate(chunks):
        # Simulate processing time
        time.sleep(0.3)

        accumulated_result += f" {chunk}"

        emit(StreamEvent(
            event_type=EventType.DATA,
            data={
                "chunk": chunk,
                "partial_result": accumulated_result.strip(),
                "progress": (i + 1) / len(chunks),
            },
            chunk_index=i,
            total_chunks=len(chunks),
            is_final=(i == len(chunks) - 1),
        ))

    final_result = f"Query '{query}' result:{accumulated_result}"

    emit(StreamEvent(
        event_type=EventType.COMPLETE,
        data={"final_result": final_result},
        is_final=True,
    ))

    return final_result


# Example 3: Error handling with events
def risky_operation(
    params: dict[str, Any],
    emit: Callable[[dict[str, Any]], None],
) -> dict[str, Any]:
    """Operation that might fail, with detailed error reporting."""

    emit({"stage": "validation", "status": "start"})

    try:
        # Validation
        if not params.get("data"):
            emit({
                "stage": "validation",
                "status": "failed",
                "error": "Missing required 'data' field",
                "error_code": "VALIDATION_ERROR",
            })
            msg = "Missing required data"
            raise ValueError(msg)

        emit({"stage": "validation", "status": "passed"})

        # Processing
        emit({"stage": "processing", "status": "start"})

        if params.get("simulate_error"):
            emit({
                "stage": "processing",
                "status": "failed",
                "error": "Simulated processing error",
                "error_code": "PROCESSING_ERROR",
            })
            msg = "Simulated error"
            raise RuntimeError(msg)

        # Success
        result = {"status": "success", "processed_data": params["data"]}
        emit({"stage": "processing", "status": "completed", "result": result})

        return result

    except Exception as e:
        emit({
            "stage": "error_handling",
            "error": str(e),
            "error_type": type(e).__name__,
            "recoverable": isinstance(e, ValueError),
        })
        raise


# Example 4: Dynamic tool with events
def create_advanced_calculator():
    """Create a calculator with advanced event reporting."""
    fc = Funcall()

    def advanced_calc_handler(
        operation: str,
        a: float,
        b: float,
        emit: Callable[[dict[str, Any]], None],
    ) -> dict[str, Any]:
        """Calculator with detailed event reporting."""

        emit({
            "event": "calculation_start",
            "operation": operation,
            "operands": [a, b],
            "timestamp": time.time(),
        })

        try:
            operations = {
                "add": lambda x, y: x + y,
                "subtract": lambda x, y: x - y,
                "multiply": lambda x, y: x * y,
                "divide": lambda x, y: x / y if y != 0 else None,
            }

            if operation not in operations:
                emit({
                    "event": "invalid_operation",
                    "operation": operation,
                    "supported_operations": list(operations.keys()),
                })
                msg = f"Unsupported operation: {operation}"
                raise ValueError(msg)

            # Emit before calculation
            emit({
                "event": "calculating",
                "operation": operation,
                "step": "executing",
            })

            result = operations[operation](a, b)

            if result is None:
                emit({
                    "event": "division_by_zero",
                    "operands": [a, b],
                })
                msg = "Cannot divide by zero"
                raise ZeroDivisionError(msg)

            response = {
                "operation": operation,
                "operands": [a, b],
                "result": result,
                "success": True,
            }

            emit({
                "event": "calculation_success",
                "result": result,
                "operation_details": response,
            })

            return response

        except Exception as e:
            emit({
                "event": "calculation_error",
                "error": str(e),
                "error_type": type(e).__name__,
            })
            raise

    fc.add_dynamic_tool(
        name="advanced_calc",
        description="Advanced calculator with detailed event reporting",
        parameters={
            "operation": {
                "type": "string",
                "enum": ["add", "subtract", "multiply", "divide"],
                "description": "Mathematical operation to perform",
            },
            "a": {"type": "number", "description": "First operand"},
            "b": {"type": "number", "description": "Second operand"},
        },
        required=["operation", "a", "b"],
        handler=advanced_calc_handler,
    )

    return fc


class EventAggregator:
    """Utility class to collect and analyze events."""

    def __init__(self) -> None:
        self.events = []
        self.error_count = 0
        self.success_count = 0

    def handle_event(self, event: Any) -> None:
        """Collect events for analysis."""
        self.events.append({"timestamp": time.time(), "data": event})

        # Count successes and errors
        if isinstance(event, dict):
            if "error" in event or event.get("status") == "failed":
                self.error_count += 1
            elif event.get("status") == "success" or "success" in str(event):
                self.success_count += 1

    def get_summary(self) -> dict[str, Any]:
        """Get event summary."""
        return {
            "total_events": len(self.events),
            "errors": self.error_count,
            "successes": self.success_count,
            "latest_events": self.events[-5:] if self.events else [],
        }


def main():
    """Demonstrate advanced event patterns."""
    print("=== Advanced Events Demo ===\n")

    # Example 1: Context + Events
    print("1. Context + Events Combination:")
    fc = Funcall([context_aware_task])

    aggregator = EventAggregator()
    fc.on_event(aggregator.handle_event)

    # Also add specific handler for demo
    def handle_context_events(event: dict[str, Any]) -> None:
        event_type = event.get("type", "unknown")
        if event_type == "session_start":
            print(f"   🎬 Session started for user {event['user_id']} ({event['session_id']})")
        elif event_type == "session_complete":
            print(f"   ✅ Session completed for user {event['user_id']}")
        elif event_type == "debug":
            print(f"   🐛 Debug: {event['message']}")

    fc.on_event(handle_context_events)

    context = ProcessingContext(
        user_id="user123",
        session_id="session456",
        config={"detailed_logging": True},
    )

    result = fc.call_function(
        "context_aware_task",
        '{"input": "test data", "priority": "high"}',
        context=context,
    )
    print(f"   Result: {result['processed_by']} - {result['result']}\n")

    # Example 2: Streaming events
    print("2. Streaming Data Processing:")
    fc_stream = Funcall([streaming_processor])

    def handle_stream_events(event: StreamEvent) -> None:
        if event.event_type == EventType.STATUS:
            print(f"   📊 {event.data['message']}")
        elif event.event_type == EventType.DATA:
            data = event.data
            progress = f"({data['progress']*100:.0f}%)"
            status = "🏁" if event.is_final else "⏳"
            print(f"   {status} {progress} Chunk: {data['chunk']} -> {data['partial_result']}")
        elif event.event_type == EventType.COMPLETE:
            print(f"   ✨ Final: {event.data['final_result']}")

    fc_stream.on_event(handle_stream_events)

    result = fc_stream.call_function("streaming_processor", '"user query about data analysis"')
    print(f"   Streaming result: {result}\n")

    # Example 3: Error handling with events
    print("3. Error Handling with Events:")
    fc_error = Funcall([risky_operation])

    def handle_error_events(event: dict[str, Any]) -> None:
        stage = event.get("stage", "unknown")
        status = event.get("status", "unknown")

        if stage == "validation":
            if status == "start":
                print("   🔍 Starting validation...")
            elif status == "passed":
                print("   ✅ Validation passed")
            elif status == "failed":
                print(f"   ❌ Validation failed: {event['error']}")
        elif stage == "processing":
            if status == "start":
                print("   ⚙️  Starting processing...")
            elif status == "completed":
                print("   ✅ Processing completed")
            elif status == "failed":
                print(f"   ❌ Processing failed: {event['error']}")
        elif stage == "error_handling":
            recoverable = "🔄 Recoverable" if event.get("recoverable") else "💥 Fatal"
            print(f"   🚨 {recoverable} Error: {event['error']}")

    fc_error.on_event(handle_error_events)

    # Successful operation
    print("   Successful operation:")
    result = fc_error.call_function("risky_operation", '{"data": "valid input"}')
    print(f"   Result: {result}\n")

    # Failed operation
    print("   Failed operation:")
    try:
        fc_error.call_function("risky_operation", '{"simulate_error": true, "data": "test"}')
    except Exception as e:
        print(f"   Exception: {e}\n")

    # Example 4: Dynamic tool with events
    print("4. Dynamic Tool with Events:")
    calc = create_advanced_calculator()

    def handle_calc_events(event: dict[str, Any]) -> None:
        event_name = event.get("event", "unknown")
        if event_name == "calculation_start":
            op_str = f"{event['operands'][0]} {event['operation']} {event['operands'][1]}"
            print(f"   🧮 Starting calculation: {op_str}")
        elif event_name == "calculation_success":
            print(f"   ✅ Result: {event['result']}")
        elif event_name == "calculation_error":
            print(f"   ❌ Error: {event['error']}")
        elif event_name == "invalid_operation":
            print(f"   ❓ Invalid operation: {event['operation']}")

    calc.on_event(handle_calc_events)

    # Successful calculation
    result = calc.call_function("advanced_calc", '{"operation": "multiply", "a": 7, "b": 6}')
    print(f"   Calculation result: {result}\n")

    # Failed calculation
    try:
        calc.call_function("advanced_calc", '{"operation": "divide", "a": 10, "b": 0}')
    except Exception as e:
        print(f"   Division by zero handled: {e}\n")

    # Event aggregator summary
    print("5. Event Analytics:")
    summary = aggregator.get_summary()
    print(f"   Total events: {summary['total_events']}")
    print(f"   Errors: {summary['errors']}, Successes: {summary['successes']}")

    print("\n=== Advanced Demo Complete ===")


if __name__ == "__main__":
    main()

