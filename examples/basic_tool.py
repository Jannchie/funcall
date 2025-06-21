import logging

from openai import AsyncClient

from funcall import Funcall

logging.basicConfig(level=logging.DEBUG)


# Define the function to be called
def add(a: float, b: float) -> float:
    """Calculate the sum of two numbers"""
    return a + b


# Use Funcall to manage function
fc = Funcall([add])
client = AsyncClient()


async def main():
    resp = await client.responses.create(
        model="gpt-4.1-nano",
        input=[
            {
                "role": "user",
                "content": "Use function call to calculate the sum of 1000 and 7",
            },
        ],
        tools=fc.get_tools(),  # Get the function metadata
        stream=True,
    )
    async for _chunk in resp:
        ...


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
    # resp = openai.responses.completion(
    #     model="gpt-4.1-nano",
    #     input=[
    #         {
    #             "role": "user",
    #             "content": "Use function call to calculate the sum of 114 and 514",
    #         },
    #     ],
    #     tools=fc.get_tools(),  # Get the function metadata
    # )
    # print(resp)
