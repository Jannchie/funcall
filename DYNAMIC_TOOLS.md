# 动态工具功能

Funcall 库现在支持动态添加工具，无需定义实际的函数。这提供了更灵活的工具管理方式。

## 主要功能

### 1. 添加动态工具

使用 `add_dynamic_tool` 方法可以直接通过工具元数据注册工具：

```python
from funcall import Funcall

funcall = Funcall()

# 添加一个基本工具
funcall.add_dynamic_tool(
    name="calculator",
    description="执行基本的数学运算",
    parameters={
        "operation": {
            "type": "string",
            "description": "要执行的运算",
            "enum": ["add", "subtract", "multiply", "divide"]
        },
        "a": {
            "type": "number", 
            "description": "第一个数字"
        },
        "b": {
            "type": "number",
            "description": "第二个数字"
        }
    },
    required=["operation", "a", "b"],
    handler=lambda operation, a, b: {
        "add": a + b,
        "subtract": a - b, 
        "multiply": a * b,
        "divide": a / b if b != 0 else "除数不能为0"
    }[operation]
)
```

### 2. 参数说明

- `name`: 工具名称
- `description`: 工具描述  
- `parameters`: 参数定义（JSON Schema 格式）
- `required`: 必需参数列表（可选）
- `handler`: 自定义处理函数（可选）

### 3. 处理器选项

#### 使用自定义处理器

```python
def custom_handler(city: str, units: str = "celsius") -> dict:
    return {"city": city, "temperature": "25°C", "units": units}

funcall.add_dynamic_tool(
    name="get_weather",
    description="获取天气信息",
    parameters={
        "city": {
            "type": "string", "description": "城市名称"
        },
        "units": {
            "type": "string", "description": "温度单位", "default": "celsius"
        }
    },
    required=["city"],
    handler=custom_handler
)
```

#### 无处理器（默认行为）

如果不提供处理器，工具将返回调用信息：

```python
funcall.add_dynamic_tool(
    name="simple_tool",
    description="简单工具",
    parameters={
        "input": {"type": "string", "description": "输入参数"}
    },
    required=["input"]
)

# 调用时会返回:
# {
#     "tool": "simple_tool",
#     "arguments": {"input": "value"},
#     "message": "Tool 'simple_tool' called with arguments: {'input': 'value'}"
# }
```

### 4. 移除动态工具

```python
funcall.remove_dynamic_tool("tool_name")
```

### 5. 集成现有功能

动态工具与现有的功能完全集成：

- `get_tools()` - 包含动态工具的定义
- `call_function()` / `call_function_async()` - 调用动态工具
- `handle_function_call()` - 处理来自LLM的工具调用
- `get_tool_meta()` - 获取工具元数据

## 使用场景

1. **快速原型开发** - 无需定义函数即可测试工具
2. **配置驱动的工具** - 从配置文件动态创建工具
3. **API代理工具** - 创建调用外部API的工具
4. **模拟和测试** - 创建用于测试的模拟工具

## 示例

查看 `examples/dynamic_tools.py` 获取完整的使用示例。

运行示例：

```bash
python examples/dynamic_tools.py
```
