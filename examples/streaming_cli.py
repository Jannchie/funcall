from abc import ABC, abstractmethod
from logging import getLogger

import openai
from openai.types.responses import (
    EasyInputMessageParam,
    ResponseCompletedEvent,
    ResponseContentPartAddedEvent,
    ResponseContentPartDoneEvent,
    ResponseCreatedEvent,
    ResponseFunctionCallArgumentsDeltaEvent,
    ResponseFunctionCallArgumentsDoneEvent,
    ResponseFunctionToolCall,
    ResponseInProgressEvent,
    ResponseInputParam,
    ResponseOutputItemAddedEvent,
    ResponseOutputItemDoneEvent,
    ResponseTextDeltaEvent,
    ResponseTextDoneEvent,
)
from pydantic import BaseModel, Field
from rich.console import Console

from funcall import Funcall

logger = getLogger("example")


class OutputChannel(ABC):
    @abstractmethod
    def send(self, text: str, style: str | None = None, end: str = "\n") -> None:
        """Normal/Label output, supports style"""

    @abstractmethod
    def stream(self, text: str, style: str | None = None) -> None:
        """Streaming output, supports style"""

    @abstractmethod
    def flush(self) -> None:
        """New line after streaming output"""


class RichConsoleOutputChannel(OutputChannel):
    def __init__(self) -> None:
        self.console = Console()

    def send(self, text: str, style: str | None = None, end: str = "\n") -> None:
        self.console.print(text, style=style, end=end)

    def stream(self, text: str, style: str | None = None) -> None:
        self.console.print(text, style=style, end="")

    def flush(self) -> None:
        self.console.print()


class AddForm(BaseModel):
    """Addition Form"""

    a: float = Field(description="First number")
    b: float = Field(description="Second number")


def add(data: AddForm) -> float:
    return data.a + data.b


functions = Funcall([add])


def chat_with_stream(output_channel: OutputChannel | None = None):  # noqa: C901, PLR0912
    system_prompt: EasyInputMessageParam = {"role": "system", "content": "You are a helpful assistant."}
    history: ResponseInputParam = [system_prompt]
    output = output_channel or RichConsoleOutputChannel()

    output.send("Welcome to the command line chat. Type 'exit' to quit.")
    continue_output = False
    while True:
        if not continue_output:
            # Render [User] label, green, 8 characters wide, label and content rendered separately
            output.send("\n[User]  ", style="bold green", end="")
            try:
                user_prompt = input("")

                if user_prompt.lower().strip() == "exit":
                    break
            except KeyboardInterrupt:
                # Ctrl+D to exit
                output.send("\nBye!")
                break
            history.append({"role": "user", "content": user_prompt})

            # Render [Agent] label, blue, 8 characters wide, label and content rendered separately
            output.send("[Agent] ", style="bold blue", end="")

        continue_output = False

        # Execute streaming output
        response = openai.responses.create(
            model="gpt-4.1",
            input=history,
            stream=True,
            tools=functions.get_tools(),
        )
        answer = ""
        for chunk in response:
            content = None
            if isinstance(chunk, ResponseOutputItemAddedEvent):
                if isinstance(chunk.item, ResponseFunctionToolCall):
                    output.flush()
                    output.send("[Function]", style="bold yellow")
            elif isinstance(chunk, ResponseTextDeltaEvent):
                content = getattr(chunk, "delta", None)
                if content:
                    output.stream(content, style="white")
                    answer += content
            elif isinstance(chunk, ResponseOutputItemDoneEvent):
                item = chunk.item
                if isinstance(item, ResponseFunctionToolCall):
                    resp = functions.handle_function_call(item)

                    result_prompt = f"The function call result is\n{resp}"

                    output.send(result_prompt, style="bold yellow")
                    output.send("[Function]", style="bold yellow")
                    history.append({"role": "user", "content": result_prompt})
                    continue_output = True
                    continue
            elif isinstance(
                chunk,
                (
                    ResponseCreatedEvent,
                    ResponseFunctionCallArgumentsDeltaEvent,
                    ResponseFunctionCallArgumentsDoneEvent,
                    ResponseContentPartAddedEvent,
                    ResponseInProgressEvent,
                    ResponseContentPartDoneEvent,
                    ResponseTextDoneEvent,
                    ResponseCompletedEvent,
                ),
            ):
                ...
            else:
                logger.debug("Unknown chunk type: %s, chunk: %s", type(chunk).__name__, chunk)

        output.flush()  # Newline
        history.append({"role": "assistant", "content": answer})


if __name__ == "__main__":
    chat_with_stream()
