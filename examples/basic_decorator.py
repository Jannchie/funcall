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
    assert weather_resp == "The weather in New York is sunny.", "Weather response does not match expected output"
    temperature_resp = await fc.call_function_async("get_temperature", '{"city": "New York"}')
    assert temperature_resp == "The temperature in New York is 25°C.", "Temperature response does not match expected output"
    get_weather_meta = fc.get_tool_meta("get_weather")
    assert get_weather_meta["require_confirm"] == True, "Tool metadata does not match"
    get_temperature_meta = fc.get_tool_meta("get_temperature")
    assert get_temperature_meta["return_direct"] == True, "Tool metadata does not match"


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
