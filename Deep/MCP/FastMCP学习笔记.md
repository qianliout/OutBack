# FastMCP 深度学习笔记 (v2)

> **版本说明**: 此版本是根据您的反馈重写的深度版。它将超越“如何使用”的层面，深入探讨 FastMCP 框架**内部如何工作**、**如何封装 MCP 协议**以及其**核心架构设计**，为您提供专家级的知识。

---

## Part 1: 核心概念再审视 (从架构师视角)

### 1.1 [[MCP (Magentic Control Protocol)]] - 不仅仅是消息格式

从专业角度看，MCP 是一套**状态机同步协议**。它的核心价值在于，它允许两个独立的系统（Agent 服务端和 Client 执行端）通过异步消息来同步一个“任务”的状态。

-   **Agent (MCP Server, e.g., FastMCP)**: 维护一个**“思考”状态机**。它的状态可以是 `THINKING`, `WAITING_FOR_TOOL_OUTPUT` 等。
-   **Client (MCP Client)**: 维护一个**“执行”状态机**。它的状态可以是 `IDLE`, `EXECUTING_TOOL`。

`[[Control Loop]]` 的本质就是通过交换 MCP 消息（如 `tool_code`, `tool_output`）来驱动这两个状态机向前演进，直至任务完成。

### 1.2 [[FastMCP]] - 一个基于 ASGI 的 MCP 协议服务器

FastMCP 的本质是一个 **ASGI (Asynchronous Server Gateway Interface) 应用**，就像 FastAPI 或 Starlette 一样。这意味着：

1.  **协议解耦**: 它不直接处理原始的 TCP 连接或 HTTP 请求。它运行在 ASGI 服务器（如 Uvicorn）之上。
2.  **异步核心**: 它的根基是 Python 的 `asyncio`，所有操作都围绕事件循环构建，这使其能够高效地处理大量并发的、长生命周期的 Agent 会话。
3.  **Web-Native**: 它天然地利用了现代 Web 技术，尤其是 **WebSockets**，作为 `[[MCP (Magentic Control Protocol)]]` 的主要传输层。

**为什么是 WebSocket?**
MCP 要求一个**持久、双向、低延迟**的通信通道来维持 `[[Control Loop]]`。WebSocket 完美地满足了这些要求，允许 Agent 和 Client 随时互相发送消息，而无需每次都建立新的 HTTP 连接。

---

## Part 2 & 3 (精简): 用户层回顾

此部分内容与前一版类似，我们快速回顾关键点：

-   `@mcp.tool()`: **声明**一个函数为工具。
-   `@mcp.prompt()`: **声明**一个提示词模板。
-   `mcp.run()`: 通过 ASGI 服务器**启动**服务。
-   **Pydantic 集成**: 用于**声明**工具参数的数据结构和校验规则。

关键在于理解“声明”这个词。你只是告诉 FastMCP “有什么”，而框架本身负责处理“如何暴露、如何调用、如何校验”。现在，我们来看框架是如何做到这一切的。

---

## Part 4:【核心】揭秘 FastMCP 内部工作流

这是本笔记的核心。我们将深入框架内部，理解从一个 HTTP 请求到工具执行的全过程。

### 4.1 启动流程：装饰器的“魔法”揭秘

当你运行 `python -m fastmcp run main:mcp` 时，发生了什么？

1.  **代码加载**: Python 解释器加载 `main.py`。
2.  **装饰器执行**: 在加载过程中，`@mcp.tool()` 装饰器被**立即执行**。它不是在请求时才运行！
3.  **工具注册**: `@mcp.tool()` 装饰器对 `add` 函数做了什么？
    a.  **函数自省 (Introspection)**: 它使用 Python 的 `inspect` 模块来读取 `add` 函数的元信息：
        -   函数名: `add`
        -   参数: `a` 和 `b`
        -   类型提示: `a: int`, `b: int`, `-> int`
        -   文档字符串: `"""计算两个整数的和..."""`
    b.  **构建工具规约 (Tool Specification)**: 它将这些信息整合成一个结构化的对象（可以理解为一个内部的 `ToolSpec` 类实例），其中包含了符合 OpenAI Function Calling 或类似规范的 JSON Schema 定义。
    c.  **中心化注册**: 这个 `ToolSpec` 对象被添加到一个 `FastMCP` 实例内部的一个**“工具注册表”**中，通常是一个字典，类似：`self.tools['add'] = tool_spec_object`。

**结论**: 到服务启动完成时，`mcp` 实例已经拥有了一个完整的、关于所有可用工具的、结构化的“知识库”。它知道每个工具的名字、功能描述、以及参数的精确定义。

### 4.2 MCP 会话生命周期：WebSocket 的角色

1.  **连接建立**: 一个 MCP Client 通过 WebSocket 连接到 FastMCP 服务器的特定端点（例如 `ws://localhost:8000/mcp/v1/session`）。
2.  **会话创建**: FastMCP 为这个 WebSocket 连接创建一个独立的**会话上下文 (Session Context)**。这个上下文会跟踪该会-话的所有状态，包括对话历史、当前正在等待哪个工具的返回等。这使得 FastMCP 能够同时管理成百上千个独立的 Agent 会话。
3.  **消息循环**: 连接建立后，FastMCP 在一个 `async for` 循环中等待来自该 WebSocket 的消息。

### 4.3 深入 `Control Loop` 的一次完整交互

假设 Client 发送了一个用户请求: "计算5和10的和"。

**Step 1: LLM "思考" 与 `tool_code` 的生成**

-   FastMCP 将对话历史和用户的新请求，连同从**工具注册表**中获取的所有工具的规约，一同发送给 LLM。
-   LLM 回复一个 "function call" 请求，意图调用 `add(a=5, b=10)`。
-   FastMCP 捕获到这个意图，并构造一个 `[[Pydantic]]` 模型实例，例如 `ToolCodeMessage(name='add', arguments={'a': 5, 'b': 10})`。
-   此 Pydantic 模型被序列化为 JSON 字符串: `{"type": "tool_code", "call": ...}`。
-   该 JSON 字符串通过 WebSocket **发送给 Client**。
-   **关键**: Agent 的状态切换为 `WAITING_FOR_TOOL_OUTPUT`，并 `await` 等待一个与此次调用 ID 匹配的 `tool_output` 事件。

**Step 2: Client 执行与 `tool_output` 的返回**

-   Client 接收到 `tool_code` 消息，解析 JSON。
-   Client 在自己的环境中查找并执行 `add` 函数，得到结果 `15`。
-   Client 构造 `tool_output` 消息: `{"type": "tool_output", ..., "output": "15"}`。
-   该消息通过 WebSocket **发送回 FastMCP 服务器**。

**Step 3: FastMCP 处理 `tool_output`**

-   服务器的 WebSocket 消息循环接收到该消息。
-   **反序列化与校验**: FastMCP 使用 `[[Pydantic]]` 模型（例如 `ToolOutputMessage`) 来解析和校验这个 JSON 字符串。如果格式或类型不匹配，会立刻向 Client 抛出 `error` 消息。
-   **唤醒等待的 Agent**: FastMCP 根据调用 ID 找到正在 `await` 的那个 Agent 会话。
-   **注入结果**: 工具的输出结果 (`15`) 被作为一个事件或变量，添加到 Agent 的上下文中。
-   **恢复 `Control Loop`**: Agent 的 `await` 结束，`[[Control Loop]]` 继续。它将工具的执行结果加入到对话历史中，再次调用 LLM 进行下一步“思考”。
-   LLM 看到 `tool_output` 后，知道计算已完成，于是生成最终的用户回答。
-   FastMCP 将最终回答通过 WebSocket 发送给 Client。

**伪代码：FastMCP 内部的会话处理器**

```python
# 这是一个简化的概念性伪代码，不代表真实源码
class MCPSessionHandler:
    def __init__(self, websocket, tool_registry):
        self.websocket = websocket
        self.tool_registry = tool_registry
        self.conversation_history = []

    async def handle(self):
        # 接收用户初始请求
        user_prompt = await self.websocket.receive_json()
        self.conversation_history.append({"role": "user", "content": user_prompt})

        while True:
            # 1. 思考：调用 LLM
            llm_response = self.call_llm(
                history=self.conversation_history,
                tools=self.tool_registry.get_specs() # 获取所有工具的规约
            )

            if llm_response.is_function_call():
                # 2. 生成 tool_code 并发送
                tool_code_msg = self.create_tool_code_message(llm_response.function_call)
                await self.websocket.send_json(tool_code_msg.model_dump())

                # 3. 等待 tool_output
                # 框架在这里会有一个复杂的事件/回调机制
                tool_output_json = await self.wait_for_tool_output(tool_code_msg.call_id)
                
                # 4. 校验并恢复
                tool_output_msg = ToolOutputMessage.model_validate(tool_output_json)
                self.conversation_history.append(
                    {"role": "tool", "content": tool_output_msg.output}
                )
                # 继续下一个 while 循环，把工具结果带给 LLM

            else: # 如果 LLM 返回的是最终答案
                await self.websocket.send_json({"type": "final_answer", "content": llm_response.text})
                break # 结束循环
```

---

## Part 5: 专业级架构与实践

### 5.1 依赖注入 (Dependency Injection) 的高级用法

FastMCP 借鉴了 FastAPI 的依赖注入系统，这不仅仅是为了方便，更是为了**解耦和可测试性**。

**场景**: 你的工具需要访问数据库。

**错误的做法**: 在工具函数内部创建数据库连接。这会导致连接泛滥且难以测试。

**专业的做法**:

1.  **定义依赖项**: 创建一个函数，它负责提供数据库连接。

    ```python
    # app/dependencies.py
    def get_db_session():
        db = SessionLocal()
        try:
            yield db
        finally:
            db.close()
    ```

2.  **在工具中声明依赖**: 使用 `Depends`。

    ```python
    # app/tools/user_tools.py
    from fastapi import Depends # FastMCP 复用了 FastAPI 的 Depends
    from sqlalchemy.orm import Session
    from .. import models
    from ..dependencies import get_db_session
    
    @mcp.tool()
    def get_user(user_id: int, db: Session = Depends(get_db_session)):
        """根据用户 ID 获取用户信息。"""
        return db.query(models.User).filter(models.User.id == user_id).first()
    ```

**框架在背后做了什么？**
在调用 `get_user` 工具之前，FastMCP 的依赖注入系统会：
1.  看到 `db: Session = Depends(get_db_session)`。
2.  执行 `get_db_session` 函数。
3.  `yield db` 将数据库会话 `db` 作为值，**注入**到 `get_user` 函数的 `db` 参数中。
4.  当 `get_user` 执行完毕后，`finally` 块中的 `db.close()` 会被执行，实现资源的自动管理。

这使得你的工具逻辑与数据库连接的创建和销毁完全分离，在测试时，你可以轻松地提供一个假的数据库会话。

### 5.2 总结

-   FastMCP 的核心是一个 **ASGI 应用**，利用 **WebSocket** 实现与 Client 的持久化双向通信。
-   `@mcp.tool` 装饰器在**启动时**通过**函数自省**来构建一个中心化的**工具注册表**。
-   每个 WebSocket 连接都会创建一个独立的**会话上下文**，实现多租户隔离。
-   框架内部通过 **Pydantic 模型**对 MCP 消息进行严格的**序列化、反序列化和校验**。
-   **依赖注入**是实现代码解耦、状态管理和可测试性的关键高级特性。

这份深度笔记希望能帮助你构建一个关于 FastMCP 内部机制的清晰心智模型。接下来，你可以尝试阅读框架的部分源码，或者设计更复杂的、带有依赖注入的工具，来巩固这些专业知识。
