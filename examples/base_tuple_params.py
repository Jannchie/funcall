import openai
from openai.types.responses import ResponseFunctionToolCall
from pydantic import BaseModel, Field

from funcall import Funcall


# 定义 pydantic item
class Item(BaseModel):
    name: str = Field(description="Item name")
    value: int = Field(description="Item value")


def sum_values(items: tuple[Item]) -> int:
    """Sum the 'value' field of all items in the tuple."""
    return sum(item.value for item in items)


fc = Funcall([sum_values])
print(fc.get_tools())

resp = openai.responses.create(
    model="gpt-4.1",
    input="Sum the values of items: apple=1, banana=2, cherry=3",
    tools=fc.get_tools(),
)

for o in resp.output:
    if isinstance(o, ResponseFunctionToolCall):
        result = fc.handle_function_call(o)
        print(result)
