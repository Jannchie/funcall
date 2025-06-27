from funcall import Funcall


def add(a: str, b: str) -> str:
    """Concatenate two strings"""
    return a + b


def test_litellm_funcall_add():
    fc = Funcall([add])
    tools_funcall = fc.get_tools(target="completion")
    assert tools_funcall == [
        {
            "type": "function",
            "function": {
                "name": "add",
                "description": "Concatenate two strings",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "a": {
                            "title": "Param 0",
                            "type": "string",
                        },
                        "b": {
                            "title": "Param 1",
                            "type": "string",
                        },
                    },
                    "additionalProperties": False,
                    "required": [
                        "a",
                        "b",
                    ],
                },
            },
        },
    ], "Funcall tools do not match Litellm tools"


def get_weather(city: str, date: str | None = None) -> str:
    """Get the weather for a specific city and date."""
    return f"The weather in {city} on {date if date else 'today'} is sunny."


def test_litellm_funcall_get_weather():
    fc = Funcall([get_weather])
    tools_funcall = fc.get_tools(target="completion")
    assert tools_funcall == [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get the weather for a specific city and date.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "city": {
                            "title": "Param 0",
                            "type": "string",
                        },
                        "date": {
                            "title": "Param 1",
                            "anyOf": [{"type": "string"}, {"type": "null"}],
                        },
                    },
                    "additionalProperties": False,
                    "required": [
                        "city",
                    ],
                },
            },
        },
    ], "Funcall tools do not match Litellm tools"


if __name__ == "__main__":
    test_litellm_funcall_add()
    print("Test passed successfully!")
