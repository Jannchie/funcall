import pytest

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


@tool
def echo(text: str) -> str:
    """Echo the input text."""
    return text


@tool
async def async_echo(text: str) -> str:
    """Async echo the input text."""
    return text


@pytest.mark.asyncio
async def test_get_weather():
    fc = Funcall([get_weather])
    weather_resp = await fc.call_function_async("get_weather", '{"city": "New York"}')
    assert weather_resp == "The weather in New York is sunny."
    get_weather_meta = fc.get_tool_meta("get_weather")
    assert get_weather_meta["require_confirm"]


@pytest.mark.asyncio
async def test_get_temperature():
    fc = Funcall([get_temperature])
    temperature_resp = await fc.call_function_async("get_temperature", '{"city": "New York"}')
    assert temperature_resp == "The temperature in New York is 25°C."
    get_temperature_meta = fc.get_tool_meta("get_temperature")
    assert get_temperature_meta["return_direct"]


@pytest.mark.asyncio
async def test_echo():
    fc = Funcall([echo])
    resp = await fc.call_function_async("echo", '{"text": "hello"}')
    assert resp == "hello"
    meta = fc.get_tool_meta("echo")
    assert not meta["require_confirm"]
    assert not meta["return_direct"]


@pytest.mark.asyncio
async def test_async_echo():
    fc = Funcall([async_echo])
    resp = await fc.call_function_async("async_echo", '{"text": "world"}')
    assert resp == "world"
    meta = fc.get_tool_meta("async_echo")
    assert not meta["require_confirm"]
    assert not meta["return_direct"]


def test_sync_echo():
    fc = Funcall([echo])
    resp = fc.call_function("echo", '{"text": "hello"}')
    assert resp == "hello"
    meta = fc.get_tool_meta("echo")
    assert not meta["require_confirm"]
    assert not meta["return_direct"]


def test_sync_require_confirm():
    fc = Funcall([get_weather])
    resp = fc.call_function("get_weather", '{"city": "New York"}')
    assert resp == "The weather in New York is sunny."
    get_weather_meta = fc.get_tool_meta("get_weather")
    assert get_weather_meta["require_confirm"]
