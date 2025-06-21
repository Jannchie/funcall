from funcall.decorators import tool
from funcall.funcall import Funcall


@tool(require_confirmation=True)
def get_weather(city: str) -> str:
    """Get the weather for a specific city."""
    return f"The weather in {city} is sunny."


@tool(return_immediately=True)
async def get_temperature(city: str) -> str:
    """Get the temperature for a specific city."""
    return f"The temperature in {city} is 25°C."


# Use Funcall to manage function
fc = Funcall([get_weather, get_temperature])


async def main():
    weather_resp = await fc.call_function_async("get_weather", '{"city": "New York"}')
    temperature_resp = await fc.call_function_async("get_temperature", '{"city": "New York"}')
    get_weather_meta = fc.get_tool_meta("get_weather")
    get_temperature_meta = fc.get_tool_meta("get_temperature")
    print(weather_resp)  # Output: The weather in New York is sunny.
    print(temperature_resp)  # Output: The temperature in New York is 25°C.
    print(get_weather_meta)  # Metadata for get_weather
    print(get_temperature_meta)  # Metadata for get_temperature


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
