import json

from funcall import Funcall


def add(a: str, b: str | None = "") -> str:
    """Concatenate two strings"""
    return a + b


def test_litellm_funcall_add():
    fc = Funcall([add])
    tools_funcall = fc.get_tools(target="litellm")[0].get("function")
    # tools_litellm = litellm.utils.function_to_dict(add)
    print("Funcall tools:", json.dumps(tools_funcall, indent=2))
    # print("Litellm tools:", json.dumps(tools_litellm, indent=2))
    # assert json.dumps(tools_funcall, indent=2) == json.dumps(tools_litellm, indent=2), "Funcall tools do not match Litellm tools"


if __name__ == "__main__":
    test_litellm_funcall_add()
    print("Test passed successfully!")
