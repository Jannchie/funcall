import openai

from funcall import Funcall


# Define the function to be called
def add(a: float, b: float) -> float:
    """Calculate the sum of two numbers"""
    return a + b


# Use Funcall to manage function
fc = Funcall([add])

resp = openai.chat.completions.create(
    model="gpt-4.1",
    messages=[{"role": "user", "content": "Use function call to calculate the sum of 114 and 514"}],
    tools=fc.get_tools(target="completion"),  # Get the function metadata
)
