import litellm

from funcall import Funcall


# Define the function to be called
def get_whether(city: str) -> float:
    """Get the weather for a specific city."""
    return "Sunny"  # Simulating a weather response


# Use Funcall to manage function
fc = Funcall([get_whether])


resp = litellm.completion(
    model="gpt-4.1-nano",
    messages=[
        {
            "role": "user",
            "content": "What is the weather like in Boston?",
        },
        {"role": "assistant", "content": None, "function_call": {"name": "get_whether", "arguments": '{ "city": "Boston, MA"}'}},
        {"role": "function", "name": "get_whether", "content": "Sunny"},
    ],
    tools=fc.get_tools(target="litellm"),  # Get the function metadata
)

choice = resp.choices[0]
print("Response:", choice)
