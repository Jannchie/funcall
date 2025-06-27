import logging

import litellm

from funcall import Funcall

logging.basicConfig(level=logging.DEBUG)


# Define the function to be called
def get_weather(city: str) -> str:
    """Get the weather for a specific city."""
    return f"The weather in {city} is sunny."  # Simulated response


# Use Funcall to manage function
fc = Funcall([get_weather])


async def main():
    resp = await litellm.acompletion(
        model="gpt-4.1-nano",
        messages=[
            {"role": "user", "content": "115 + 514 = ?"},
            # {
            #     "id": "chatcmpl-BkQu8HZFYogQwjcJXlukdL5B3e9bO",
            #     "role": "assistant",
            #     "content": "I will add 115 and 514 for you. Let me do that now.",
            #     "tool_calls": [{"function": {"arguments": '{"a":115,"b":514}', "name": "add"}, "id": "call_EzIVxomt3otdgnEHfQwrsdtc", "type": "function"}],
            # },
            # {"role": "tool", "tool_call_id": "call_EzIVxomt3otdgnEHfQwrsdtc", "content": "629"},
        ],
        tools=fc.get_tools(target="completion"),  # Get the function metadata
        stream=True,
    )
    print(resp)


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
    # resp = litellm.completion(
    #     model="gpt-4.1-nano",
    #     messages=[
    #         {"role": "user", "content": "What is the weather in New York?"},
    #     ],
    #     tools=fc.get_tools(target="completion"),  # Get the function metadata
    # )
    # print(resp)
