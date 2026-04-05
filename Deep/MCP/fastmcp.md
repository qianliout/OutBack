# FastMCP 深度学习笔记 (v3)

> **版本说明**: 此版本结合了官方文档、知识大纲和学习笔记，提供最全面、最深入的FastMCP学习指南。

---

## 第一部分：FastMCP 核心概念与架构

### 1.1 FastMCP 是什么？

FastMCP 是一个基于 Python 的 MCP (Model Context Protocol) 服务器框架，专门用于构建 AI Agent 应用。它借鉴了 FastAPI 的设计哲学，强调：

- **高性能**: 基于 ASGI 和异步架构
- **易用性**: 声明式 API，自动生成文档
- **类型安全**: 基于 Pydantic 的自动数据校验
- **生产就绪**: 内置错误处理、序列化、协议合规性

### 1.4 FastMCP 源码架构深度解析

FastMCP 的源码架构遵循清晰的层次化设计：

#### 1.4.1 核心模块结构

```
fastmcp/
├── __init__.py          # 包入口，导出主要类和函数
├── main.py              # FastMCP 核心类实现
├── tools.py             # 工具装饰器和注册机制
├── server.py            # ASGI 服务器实现
├── models.py            # MCP 消息模型定义
├── dependencies.py       # 依赖注入系统
├── exceptions.py        # 异常类定义
├── types.py             # 类型定义和别名
└── utils.py             # 工具函数
```

#### 1.4.2 核心类设计模式

**FastMCP 类 (main.py)** 采用单例模式，负责：
- 工具注册表的维护
- 依赖注入容器的管理
- ASGI 应用的创建
- 生命周期事件处理

**工具注册机制** 采用装饰器模式：
```python
# 伪代码展示装饰器内部实现
class ToolRegistry:
    def __init__(self):
        self._tools = {}
    
    def register(self, func, name=None, description=None):
        # 1. 提取函数签名和类型注解
        # 2. 生成 JSON Schema
        # 3. 存储到注册表
        self._tools[name or func.__name__] = {
            'func': func,
            'schema': generate_schema(func),
            'description': description
        }

# 装饰器工厂函数
def tool(name=None, description=None):
    def decorator(func):
        ToolRegistry.register(func, name, description)
        return func
    return decorator
```

#### 1.4.3 ASGI 应用架构

FastMCP 的 ASGI 实现基于三层架构：

1. **协议层**: 处理 WebSocket 连接和 MCP 消息解析
2. **会话层**: 管理客户端会话状态和消息路由
3. **业务层**: 执行工具调用和结果处理

```python
# 伪代码展示 ASGI 应用结构
async def app(scope, receive, send):
    if scope['type'] == 'websocket':
        # 1. 建立 WebSocket 连接
        websocket = WebSocket(scope, receive, send)
        await websocket.accept()
        
        # 2. 创建会话上下文
        session = SessionContext(websocket)
        
        # 3. 进入消息循环
        while True:
            message = await websocket.receive_json()
            
            # 4. 消息路由和处理
            if message['type'] == 'tool_call':
                result = await handle_tool_call(message, session)
                await websocket.send_json(result)
            
            # 5. 心跳和状态维护
            elif message['type'] == 'heartbeat':
                await handle_heartbeat(message, session)
```

### 1.2 MCP (Model Context Protocol) 协议

MCP 是一个标准化的协议，用于连接大型语言模型 (LLM) 与外部工具和数据。其核心思想是：

- **解耦思考与执行**: LLM 负责"思考"，外部工具负责"执行"
- **状态机同步**: Agent 和 Client 通过异步消息同步任务状态
- **双向通信**: 基于 WebSocket 的持久化双向通信通道

### 1.5 MCP 协议消息处理机制深度解析

#### 1.5.1 MCP 消息类型和状态机

MCP 协议定义了一套完整的消息类型和状态转换机制：

```python
# 伪代码展示 MCP 状态机实现
class MCPStateMachine:
    def __init__(self):
        self.state = 'IDLE'
        self.current_tool = None
        self.pending_results = {}
    
    async def handle_message(self, message: dict):
        """处理 MCP 消息的状态机"""
        msg_type = message.get('type')
        
        if self.state == 'IDLE':
            if msg_type == 'tool_call_request':
                # 接收到工具调用请求
                self.state = 'PROCESSING_TOOL'
                self.current_tool = message['tool_name']
                await self.process_tool_call(message)
            
        elif self.state == 'PROCESSING_TOOL':
            if msg_type == 'tool_result':
                # 接收到工具执行结果
                self.state = 'IDLE'
                await self.handle_tool_result(message)
            elif msg_type == 'tool_error':
                # 工具执行出错
                self.state = 'IDLE'
                await self.handle_tool_error(message)
```

#### 1.5.2 消息序列化和验证

FastMCP 使用 Pydantic 进行消息的序列化和验证：

```python
# 伪代码展示消息模型定义
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any

class MCPMessage(BaseModel):
    """MCP 消息基类"""
    type: str = Field(..., description="消息类型")
    message_id: str = Field(..., description="消息ID")
    timestamp: float = Field(default_factory=time.time, description="时间戳")

class ToolCallRequest(MCPMessage):
    """工具调用请求消息"""
    type: str = "tool_call_request"
    tool_name: str = Field(..., description="工具名称")
    arguments: Dict[str, Any] = Field(default_factory=dict, description="参数")
    session_id: Optional[str] = Field(None, description="会话ID")

class ToolResult(MCPMessage):
    """工具执行结果消息"""
    type: str = "tool_result"
    tool_name: str = Field(..., description="工具名称")
    result: Any = Field(..., description="执行结果")
    execution_time: float = Field(..., description="执行时间")

# 消息工厂函数
def create_message(msg_type: str, **kwargs) -> MCPMessage:
    """创建并验证 MCP 消息"""
    message_classes = {
        'tool_call_request': ToolCallRequest,
        'tool_result': ToolResult,
        # ... 其他消息类型
    }
    
    if msg_type not in message_classes:
        raise ValueError(f"Unknown message type: {msg_type}")
    
    return message_classes[msg_type](**kwargs)
```

#### 1.5.3 WebSocket 连接管理

FastMCP 的 WebSocket 连接管理采用连接池和会话隔离：

```python
# 伪代码展示连接管理
class ConnectionManager:
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}
        self.connection_sessions: Dict[str, SessionContext] = {}
    
    async def connect(self, websocket: WebSocket, session_id: str):
        """建立 WebSocket 连接"""
        await websocket.accept()
        self.active_connections[session_id] = websocket
        self.connection_sessions[session_id] = SessionContext(session_id)
    
    async def disconnect(self, session_id: str):
        """断开 WebSocket 连接"""
        if session_id in self.active_connections:
            websocket = self.active_connections[session_id]
            await websocket.close()
            del self.active_connections[session_id]
            del self.connection_sessions[session_id]
    
    async def broadcast(self, message: dict, exclude: List[str] = None):
        """广播消息到所有连接"""
        for session_id, websocket in self.active_connections.items():
            if exclude and session_id in exclude:
                continue
            try:
                await websocket.send_json(message)
            except Exception:
                # 连接异常，移除连接
                await self.disconnect(session_id)
```

### 1.3 FastMCP 的三层抽象

根据官方文档，FastMCP 构建在三个核心抽象上：

1. **Components (组件)**: 你暴露的内容 - 工具、资源、提示词
2. **Providers (提供者)**: 组件来源 - 装饰函数、文件、OpenAPI 规范、远程服务器
3. **Transforms (转换)**: 客户端看到的内容 - 命名空间、过滤、授权、版本控制

---

## 第二部分：环境搭建与快速入门

### 2.1 安装与环境配置

```bash
# 安装 FastMCP
pip install fastmcp

# 验证安装
python -c "import fastmcp; print(f'FastMCP version: {fastmcp.__version__}')"
```

### 2.2 第一个 FastMCP 应用

```python
# main.py
from fastmcp import FastMCP

# 创建 FastMCP 实例
mcp = FastMCP("Demo App")

@mcp.tool()
def add(a: int, b: int) -> int:
    """计算两个整数的和
    
    Args:
        a: 第一个整数
        b: 第二个整数
        
    Returns:
        两个整数的和
    """
    return a + b

@mcp.tool()
def greet(name: str) -> str:
    """向用户问好
    
    Args:
        name: 用户名
        
    Returns:
        问候语
    """
    return f"Hello, {name}!"

if __name__ == "__main__":
    mcp.run()
```

### 2.3 运行应用

```bash
# 运行 FastMCP 服务器
python -m fastmcp run main:mcp

# 或者直接运行
python main.py
```

---

## 第三部分：核心功能深入解析

### 3.1 工具定义 (Tool Definition)

#### 3.1.1 基本工具定义

```python
from fastmcp import FastMCP
from pydantic import BaseModel, Field
from typing import List, Optional

mcp = FastMCP("Advanced Demo")

# 简单工具
@mcp.tool()
def calculate_area(length: float, width: float) -> float:
    """计算矩形面积"""
    return length * width

# 使用 Pydantic 模型
class UserInfo(BaseModel):
    name: str = Field(..., description="用户姓名")
    age: int = Field(..., ge=0, le=150, description="用户年龄")
    email: Optional[str] = Field(None, description="用户邮箱")

@mcp.tool()
def create_user(user: UserInfo) -> dict:
    """创建用户信息"""
    return {
        "message": f"User {user.name} created successfully",
        "age": user.age,
        "email": user.email
    }
```

#### 3.1.2 异步工具

```python
import asyncio
from fastmcp import FastMCP

mcp = FastMCP("Async Demo")

@mcp.tool()
async def fetch_data(url: str) -> str:
    """异步获取数据"""
    # 模拟异步操作
    await asyncio.sleep(1)
    return f"Data from {url}"

@mcp.tool()
async def process_batch(items: List[str]) -> List[str]:
    """批量处理数据"""
    results = []
    for item in items:
        # 模拟异步处理
        await asyncio.sleep(0.1)
        results.append(f"processed_{item}")
    return results
```

### 3.2 提示词管理 (Prompt Management)

```python
from fastmcp import FastMCP

mcp = FastMCP("Prompt Demo")

@mcp.prompt()
def analysis_prompt(data: str) -> str:
    """数据分析提示词模板
    
    Args:
        data: 要分析的数据
        
    Returns:
        格式化后的提示词
    """
    return f"""
请分析以下数据并提供详细的报告：

数据内容：
{data}

请按照以下结构提供分析：
1. 数据摘要
2. 关键发现
3. 建议措施
"""

@mcp.prompt()
def translation_prompt(text: str, target_language: str) -> str:
    """翻译提示词模板"""
    return f"""
请将以下文本翻译成 {target_language}：

原文：
{text}

请确保翻译准确且自然。
"""
```

### 3.3 依赖注入 (Dependency Injection)

```python
from fastmcp import FastMCP, Depends
from typing import Generator
import sqlite3

mcp = FastMCP("DI Demo")

# 数据库依赖
def get_db() -> Generator[sqlite3.Connection, None, None]:
    """获取数据库连接"""
    conn = sqlite3.connect(':memory:')
    try:
        # 创建示例表
        conn.execute('''CREATE TABLE IF NOT EXISTS users 
                      (id INTEGER PRIMARY KEY, name TEXT, email TEXT)''')
        yield conn
    finally:
        conn.close()

@mcp.tool()
def add_user(name: str, email: str, db: sqlite3.Connection = Depends(get_db)) -> dict:
    """添加用户到数据库"""
    cursor = db.execute('INSERT INTO users (name, email) VALUES (?, ?)', (name, email))
    db.commit()
    return {
        "id": cursor.lastrowid,
        "name": name,
        "email": email,
        "message": "User added successfully"
    }

@mcp.tool()
def get_users(db: sqlite3.Connection = Depends(get_db)) -> list:
    """获取所有用户"""
    cursor = db.execute('SELECT * FROM users')
    return [dict(row) for row in cursor.fetchall()]
```

---

## 第四部分：FastMCP 内部机制深度解析

### 4.1 FastMCP 启动流程

#### 4.1.1 装饰器注册机制

当 Python 解释器加载包含 `@mcp.tool()` 装饰器的模块时：

1. **模块加载阶段**: 装饰器在导入时立即执行，收集函数元数据
2. **工具注册**: 函数信息被注册到 FastMCP 实例的工具注册表中
3. **Schema 生成**: 基于函数签名和类型注解自动生成 JSON Schema
4. **元数据存储**: 工具名称、描述、参数信息等被存储供后续使用

#### 4.1.2 启动过程源码级分析

```python
# 伪代码展示 FastMCP.run() 的内部实现
class FastMCP:
    def __init__(self, name: str = "FastMCP App"):
        self.name = name
        self._tools = {}  # 工具注册表
        self._prompts = {}  # 提示词注册表
        self._dependencies = {}  # 依赖项注册表
        self._asgi_app = None  # ASGI 应用实例
    
    def run(self, host: str = "0.0.0.0", port: int = 8000, **kwargs):
        """启动 FastMCP 服务器的核心方法"""
        
        # 1. 创建 ASGI 应用
        self._asgi_app = self._create_asgi_app()
        
        # 2. 配置日志和中间件
        self._configure_logging()
        self._setup_middleware()
        
        # 3. 启动 ASGI 服务器
        import uvicorn
        uvicorn.run(
            app=self._asgi_app,
            host=host,
            port=port,
            **kwargs
        )
    
    def _create_asgi_app(self):
        """创建 ASGI 应用的核心逻辑"""
        
        async def asgi_app(scope, receive, send):
            """FastMCP 的 ASGI 应用入口"""
            
            if scope['type'] == 'http':
                # HTTP 请求处理（健康检查、文档等）
                await self._handle_http_request(scope, receive, send)
            
            elif scope['type'] == 'websocket':
                # WebSocket 连接处理（MCP 协议）
                await self._handle_websocket_connection(scope, receive, send)
            
            else:
                # 其他协议类型
                await send({
                    'type': 'http.response.start',
                    'status': 400,
                    'headers': [[b'content-type', b'text/plain']]
                })
                await send({
                    'type': 'http.response.body',
                    'body': b'Unsupported protocol'
                })
        
        return asgi_app
    
    async def _handle_websocket_connection(self, scope, receive, send):
        """处理 WebSocket 连接的详细流程"""
        
        # 1. 建立 WebSocket 连接
        websocket = WebSocket(scope, receive, send)
        await websocket.accept()
        
        # 2. 创建会话上下文
        session_id = self._generate_session_id()
        session_context = SessionContext(session_id, websocket)
        
        # 3. 发送初始化消息
        await self._send_initialization_message(websocket, session_context)
        
        # 4. 进入消息循环
        try:
            while True:
                # 接收消息
                message = await websocket.receive_json()
                
                # 验证消息格式
                validated_message = self._validate_message(message)
                
                # 路由和处理消息
                await self._route_message(validated_message, session_context)
                
        except WebSocketDisconnect:
            # 连接断开处理
            await self._handle_disconnect(session_context)
        except Exception as e:
            # 异常处理
            await self._handle_error(e, session_context)
```

### 4.2 消息处理流程深度解析

#### 4.2.1 消息路由机制

```python
# 伪代码展示消息路由实现
async def _route_message(self, message: dict, session_context: SessionContext):
    """消息路由的核心逻辑"""
    
    message_type = message.get('type')
    
    # 消息类型路由表
    route_table = {
        'tool_call_request': self._handle_tool_call_request,
        'tool_result': self._handle_tool_result,
        'prompt_request': self._handle_prompt_request,
        'heartbeat': self._handle_heartbeat,
        'session_start': self._handle_session_start,
        'session_end': self._handle_session_end,
    }
    
    # 查找对应的处理器
    handler = route_table.get(message_type)
    if not handler:
        # 未知消息类型
        await self._send_error(
            session_context.websocket,
            f"Unknown message type: {message_type}"
        )
        return
    
    # 执行处理器
    try:
        await handler(message, session_context)
    except Exception as e:
        # 处理器异常
        await self._handle_handler_error(e, session_context)

async def _handle_tool_call_request(self, message: dict, session_context: SessionContext):
    """处理工具调用请求的详细流程"""
    
    tool_name = message['tool_name']
    arguments = message['arguments']
    
    # 1. 查找工具
    tool_info = self._tools.get(tool_name)
    if not tool_info:
        await self._send_error(
            session_context.websocket,
            f"Tool not found: {tool_name}"
        )
        return
    
    # 2. 验证参数
    try:
        validated_args = self._validate_arguments(
            tool_info['schema'], 
            arguments
        )
    except ValidationError as e:
        await self._send_error(
            session_context.websocket,
            f"Invalid arguments: {e}"
        )
        return
    
    # 3. 执行工具
    try:
        result = await self._execute_tool(
            tool_info['func'],
            validated_args,
            session_context
        )
        
        # 4. 发送执行结果
        await self._send_tool_result(
            session_context.websocket,
            tool_name,
            result
        )
        
    except Exception as e:
        # 工具执行异常
        await self._send_tool_error(
            session_context.websocket,
            tool_name,
            str(e)
        )

async def _execute_tool(self, tool_func, arguments, session_context):
    """执行工具的核心逻辑"""
    
    # 1. 准备依赖项
    dependencies = self._prepare_dependencies(session_context)
    
    # 2. 注入依赖项
    if asyncio.iscoroutinefunction(tool_func):
        # 异步工具
        result = await tool_func(**arguments, **dependencies)
    else:
        # 同步工具（在线程池中执行）
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None, 
            lambda: tool_func(**arguments, **dependencies)
        )
    
    return result
```

#### 4.1.2 工具规约生成

FastMCP 使用 Python 的 `inspect` 模块进行函数自省：

```python
# 伪代码展示工具规约生成
import inspect
from typing import get_type_hints

def generate_tool_spec(func):
    """生成工具规约的核心逻辑"""
    
    # 获取函数签名
    sig = inspect.signature(func)
    
    # 获取类型提示
    type_hints = get_type_hints(func)
    
    # 构建参数规约
    parameters = {}
    for param_name, param in sig.parameters.items():
        param_type = type_hints.get(param_name, Any)
        param_spec = {
            'type': param_type.__name__,
            'required': param.default is param.empty,
            'default': param.default if param.default is not param.empty else None
        }
        parameters[param_name] = param_spec
    
    # 构建返回值规约
    return_type = type_hints.get('return', Any)
    
    # 生成完整的工具规约
    tool_spec = {
        'name': func.__name__,
        'description': func.__doc__ or '',
        'parameters': parameters,
        'returns': {
            'type': return_type.__name__,
            'description': 'Tool execution result'
        }
    }
    
    return tool_spec
```

### 4.3 依赖注入系统深度解析

#### 4.3.1 依赖注入容器实现

FastMCP 的依赖注入系统采用容器模式：

```python
# 伪代码展示依赖注入容器
class DependencyContainer:
    """依赖注入容器的核心实现"""
    
    def __init__(self):
        self._services = {}  # 服务实例缓存
        self._factories = {}  # 服务工厂函数
        self._singletons = {}  # 单例服务
        self._scoped = {}  # 作用域服务（按会话）
    
    def register_factory(self, service_type, factory_func, scope='transient'):
        """注册服务工厂"""
        self._factories[service_type] = {
            'factory': factory_func,
            'scope': scope
        }
    
    def register_singleton(self, service_type, instance):
        """注册单例服务"""
        self._singletons[service_type] = instance
    
    async def resolve(self, service_type, session_context=None):
        """解析依赖项的核心方法"""
        
        # 1. 检查单例缓存
        if service_type in self._singletons:
            return self._singletons[service_type]
        
        # 2. 检查作用域缓存（按会话）
        if session_context and service_type in self._scoped:
            scoped_services = self._scoped.get(session_context.session_id, {})
            if service_type in scoped_services:
                return scoped_services[service_type]
        
        # 3. 查找工厂函数
        factory_info = self._factories.get(service_type)
        if not factory_info:
            raise DependencyResolutionError(f"No factory found for {service_type}")
        
        # 4. 创建服务实例
        factory_func = factory_info['factory']
        scope = factory_info['scope']
        
        # 解析工厂函数的依赖
        factory_deps = await self._resolve_dependencies(factory_func, session_context)
        
        # 调用工厂函数
        instance = factory_func(**factory_deps)
        
        # 5. 根据作用域缓存实例
        if scope == 'singleton':
            self._singletons[service_type] = instance
        elif scope == 'scoped' and session_context:
            if session_context.session_id not in self._scoped:
                self._scoped[session_context.session_id] = {}
            self._scoped[session_context.session_id][service_type] = instance
        # transient 作用域不缓存
        
        return instance
    
    async def _resolve_dependencies(self, func, session_context):
        """解析函数依赖项"""
        dependencies = {}
        
        # 获取函数签名
        sig = inspect.signature(func)
        
        for param_name, param in sig.parameters.items():
            # 跳过 self 参数（如果是方法）
            if param_name == 'self':
                continue
            
            # 获取参数类型注解
            type_hints = get_type_hints(func)
            param_type = type_hints.get(param_name)
            
            if param_type:
                # 解析依赖项
                dependency = await self.resolve(param_type, session_context)
                dependencies[param_name] = dependency
            else:
                # 没有类型注解的参数
                if param.default is param.empty:
                    raise DependencyResolutionError(
                        f"Parameter '{param_name}' has no type annotation"
                    )
                # 使用默认值
                dependencies[param_name] = param.default
        
        return dependencies
```

#### 4.3.2 依赖注入在工具执行中的应用

```python
# 伪代码展示依赖注入在工具执行中的集成
async def _prepare_dependencies(self, session_context):
    """为工具执行准备依赖项"""
    dependencies = {}
    
    # 获取当前会话的所有依赖项
    for dep_type in self._required_dependencies:
        try:
            dependency = await self._dependency_container.resolve(
                dep_type, session_context
            )
            dependencies[dep_type.__name__.lower()] = dependency
        except DependencyResolutionError as e:
            logger.warning(f"Failed to resolve dependency {dep_type}: {e}")
    
    return dependencies

def tool(self, name=None, description=None, dependencies=None):
    """增强的工具装饰器，支持依赖注入"""
    
    def decorator(func):
        # 提取函数需要的依赖项类型
        func_dependencies = self._extract_dependencies(func)
        
        # 注册工具时记录依赖项需求
        tool_info = {
            'func': func,
            'name': name or func.__name__,
            'description': description or func.__doc__ or '',
            'dependencies': func_dependencies,
            'schema': generate_tool_spec(func)
        }
        
        # 存储到工具注册表
        self._tools[tool_info['name']] = tool_info
        
        # 记录全局依赖项需求
        for dep_type in func_dependencies:
            if dep_type not in self._required_dependencies:
                self._required_dependencies.append(dep_type)
        
        return func
    
    return decorator

def _extract_dependencies(self, func):
    """从函数签名中提取依赖项类型"""
    dependencies = []
    
    # 获取函数签名和类型提示
    sig = inspect.signature(func)
    type_hints = get_type_hints(func)
    
    for param_name, param in sig.parameters.items():
        # 跳过 self 参数
        if param_name == 'self':
            continue
        
        # 获取参数类型
        param_type = type_hints.get(param_name)
        
        # 检查是否是注册的依赖项类型
        if (param_type and 
            param_type in self._dependency_container._factories or
            param_type in self._dependency_container._singletons):
            dependencies.append(param_type)
    
    return dependencies
```

#### 4.3.3 依赖注入的生命周期管理

FastMCP 支持三种依赖项生命周期：

1. **Transient (瞬时)**: 每次请求都创建新实例
2. **Scoped (作用域)**: 每个会话创建一个实例
3. **Singleton (单例)**: 整个应用生命周期只有一个实例

```python
# 生命周期管理的伪代码实现
class SessionContext:
    """会话上下文，管理作用域依赖项"""
    
    def __init__(self, session_id):
        self.session_id = session_id
        self._disposables = []  # 需要清理的资源
    
    async def dispose(self):
        """清理会话资源"""
        for disposable in self._disposables:
            if hasattr(disposable, 'dispose'):
                await disposable.dispose()
            elif hasattr(disposable, 'close'):
                disposable.close()
            elif hasattr(disposable, '__aexit__'):
                await disposable.__aexit__(None, None, None)
        
        self._disposables.clear()

# 在依赖注入容器中集成生命周期管理
async def resolve(self, service_type, session_context=None):
    # ... 前面的解析逻辑
    
    # 如果是作用域依赖且需要清理，注册到会话上下文
    if (session_context and 
        scope == 'scoped' and 
        hasattr(instance, 'dispose') or hasattr(instance, 'close')):
        session_context._disposables.append(instance)
    
    return instance
```

### 4.2 MCP 会话生命周期

#### 4.2.1 WebSocket 连接建立

1. **连接建立**: MCP Client 连接到 `ws://localhost:8000/mcp/v1/session`
2. **会话创建**: 为每个连接创建独立的会话上下文
3. **状态管理**: 维护对话历史、工具调用状态

#### 4.2.2 消息处理循环

```python
# 伪代码：会话处理器
class MCPSessionHandler:
    def __init__(self, websocket, tool_registry):
        self.websocket = websocket
        self.tool_registry = tool_registry
        self.conversation_history = []
        self.pending_tools = {}  # 等待中的工具调用

    async def handle_session(self):
        async for message in self.websocket:
            # 解析 MCP 消息
            mcp_message = parse_mcp_message(message)
            
            if mcp_message.type == "tool_code":
                await self.handle_tool_call(mcp_message)
            elif mcp_message.type == "tool_output":
                await self.handle_tool_output(mcp_message)
            elif mcp_message.type == "heartbeat":
                await self.handle_heartbeat(mcp_message)
            elif mcp_message.type == "error":
                await self.handle_error(mcp_message)
```

### 4.3 Control Loop 完整交互流程

#### 4.3.1 工具调用流程

**Step 1: LLM 思考与 tool_code 生成**

1. FastMCP 将对话历史 + 工具规约发送给 LLM
2. LLM 返回 function call 意图
3. FastMCP 构造 `ToolCodeMessage`
4. 通过 WebSocket 发送给 Client
5. Agent 状态变为 `WAITING_FOR_TOOL_OUTPUT`

**Step 2: Client 执行与 tool_output 返回**

1. Client 接收并解析 `tool_code` 消息
2. 查找并执行对应工具函数
3. 构造 `ToolOutputMessage`
4. 通过 WebSocket 返回结果

**Step 3: FastMCP 处理 tool_output**

1. 接收并验证 `tool_output` 消息
2. 根据调用 ID 找到等待的会话
3. 将结果注入会话上下文
4. 恢复 Control Loop，继续 LLM 思考

#### 4.3.2 消息序列图

```
Client          FastMCP Server         LLM
  |                  |                  |
  |--- WebSocket --->|                  |
  |                  |                  |
  |--- user input --->|                  |
  |                  |--- history+tools -->
  |                  |<---- function call --
  |<-- tool_code ----|                  |
  |                  |                  |
  |--- tool_output -->|                  |
  |                  |--- history+result -->
  |                  |<----- response -----|
  |<-- final answer -|                  |
```

### 4.4 Pydantic 集成与消息验证

FastMCP 使用 Pydantic 对所有 MCP 消息进行严格的序列化和验证：

```python
from pydantic import BaseModel, Field
from typing import Literal

class ToolCodeMessage(BaseModel):
    type: Literal["tool_code"] = "tool_code"
    call_id: str = Field(..., description="调用ID")
    name: str = Field(..., description="工具名称")
    arguments: dict = Field(..., description="参数")

class ToolOutputMessage(BaseModel):
    type: Literal["tool_output"] = "tool_output"
    call_id: str = Field(..., description="调用ID")
    output: str = Field(..., description="工具输出")

class ErrorMessage(BaseModel):
    type: Literal["error"] = "error"
    message: str = Field(..., description="错误信息")
    code: str = Field(..., description="错误代码")
```

---

## 第五部分：高级特性与最佳实践

### 5.1 错误处理与调试

#### 5.1.1 自定义错误处理

```python
from fastmcp import FastMCP
from fastmcp.exceptions import MCPServerError

mcp = FastMCP("Error Handling Demo")

@mcp.tool()
def risky_operation(value: int) -> str:
    """有风险的操作"""
    if value < 0:
        raise MCPServerError(
            code="INVALID_VALUE",
            message="Value must be positive",
            details={"min_value": 0}
        )
    
    if value > 100:
        raise ValueError("Value too large")
    
    return f"Operation successful with value {value}"

# 全局错误处理器
@mcp.exception_handler(ValueError)
def handle_value_error(exc: ValueError) -> dict:
    return {
        "error": "ValueError occurred",
        "message": str(exc),
        "suggestion": "Please provide a value between 0 and 100"
    }
```

#### 5.1.2 日志与监控

```python
import logging
from fastmcp import FastMCP

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

mcp = FastMCP("Logging Demo")

@mcp.tool()
def monitored_operation(data: str) -> str:
    """被监控的操作"""
    logger.info(f"Starting operation with data: {data}")
    
    try:
        result = process_data(data)
        logger.info(f"Operation completed successfully")
        return result
    except Exception as e:
        logger.error(f"Operation failed: {e}")
        raise
```

### 5.2 安全最佳实践

#### 5.2.1 输入验证与清理

```python
from fastmcp import FastMCP
from pydantic import BaseModel, validator
import html

mcp = FastMCP("Security Demo")

class UserInput(BaseModel):
    username: str
    bio: str
    
    @validator('username')
    def validate_username(cls, v):
        if not v.isalnum():
            raise ValueError('Username must be alphanumeric')
        return v
    
    @validator('bio')
    def sanitize_bio(cls, v):
        # HTML 转义防止 XSS
        return html.escape(v)

@mcp.tool()
def create_profile(user_input: UserInput) -> dict:
    """创建用户档案"""
    return {
        "username": user_input.username,
        "bio": user_input.bio,
        "status": "created"
    }
```

#### 5.2.2 权限控制

```python
from fastmcp import FastMCP, Depends
from typing import Optional

mcp = FastMCP("Auth Demo")

# 简单的认证依赖
def get_current_user(token: Optional[str] = None) -> Optional[dict]:
    """获取当前用户"""
    if token == "secret-token":
        return {"username": "admin", "role": "administrator"}
    return None

@mcp.tool()
def admin_only_operation(
    data: str, 
    user: dict = Depends(get_current_user)
) -> str:
    """仅管理员可用的操作"""
    if not user or user.get("role") != "administrator":
        raise PermissionError("Admin access required")
    
    return f"Admin operation completed: {data}"
```

### 5.3 性能优化

#### 5.3.1 异步优化

```python
import asyncio
from fastmcp import FastMCP
import aiohttp

mcp = FastMCP("Performance Demo")

@mcp.tool()
async def fetch_multiple_urls(urls: List[str]) -> List[str]:
    """并发获取多个URL"""
    async with aiohttp.ClientSession() as session:
        tasks = []
        for url in urls:
            task = fetch_url(session, url)
            tasks.append(task)
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        return [str(r) if not isinstance(r, Exception) else f"Error: {r}" for r in results]

async def fetch_url(session: aiohttp.ClientSession, url: str) -> str:
    """获取单个URL"""
    async with session.get(url) as response:
        return await response.text()
```

#### 5.3.2 缓存策略

```python
from fastmcp import FastMCP
from functools import lru_cache
from typing import List

mcp = FastMCP("Caching Demo")

@lru_cache(maxsize=100)
def expensive_computation(n: int) -> int:
    """昂贵的计算（带缓存）"""
    print(f"Computing for {n}...")
    # 模拟昂贵计算
    result = sum(i * i for i in range(n))
    return result

@mcp.tool()
def compute_with_cache(n: int) -> dict:
    """使用缓存的计算"""
    result = expensive_computation(n)
    return {
        "input": n,
        "result": result,
        "from_cache": expensive_computation.cache_info().hits > 0
    }
```

---

## 第六部分：项目结构与部署

### 6.1 推荐的项目结构

```
my_fastmcp_app/
├── app/
│   ├── __init__.py
│   ├── main.py          # FastMCP 实例和启动
│   ├── tools/           # 工具模块
│   │   ├── __init__.py
│   │   ├── math_tools.py
│   │   ├── data_tools.py
│   │   └── api_tools.py
│   ├── prompts/         # 提示词模块
│   │   ├── __init__.py
│   │   ├── analysis_prompts.py
│   │   └── translation_prompts.py
│   ├── models/          # Pydantic 模型
│   │   ├── __init__.py
│   │   ├── user_models.py
│   │   └── data_models.py
│   ├── dependencies/    # 依赖项
│   │   ├── __init__.py
│   │   ├── database.py
│   │   └── auth.py
│   └── utils/           # 工具函数
│       ├── __init__.py
│       ├── logging.py
│       └── validation.py
├── tests/               # 测试
│   ├── __init__.py
│   ├── test_tools.py
│   └── test_integration.py
├── requirements.txt     # 依赖
├── Dockerfile           # Docker 配置
└── README.md           # 项目说明
```

### 6.2 模块化工具定义

```python
# app/tools/math_tools.py
from fastmcp import FastMCP
from typing import List

# 在模块中定义工具，而不是全部在 main.py 中

def register_math_tools(mcp: FastMCP):
    """注册数学工具"""
    
    @mcp.tool()
    def add_numbers(a: float, b: float) -> float:
        """加法运算"""
        return a + b
    
    @mcp.tool()
    def calculate_statistics(numbers: List[float]) -> dict:
        """计算统计信息"""
        if not numbers:
            return {"error": "Empty list"}
        
        return {
            "mean": sum(numbers) / len(numbers),
            "min": min(numbers),
            "max": max(numbers),
            "count": len(numbers)
        }

# app/tools/data_tools.py
from fastmcp import FastMCP, Depends
from typing import List
import json

def register_data_tools(mcp: FastMCP):
    """注册数据处理工具"""
    
    @mcp.tool()
    def process_json_data(json_str: str) -> dict:
        """处理JSON数据"""
        try:
            data = json.loads(json_str)
            return {"success": True, "data": data}
        except json.JSONDecodeError as e:
            return {"success": False, "error": str(e)}
```

### 6.3 部署配置

#### 6.3.1 Docker 部署

```dockerfile
# Dockerfile
FROM python:3.11-slim

WORKDIR /app

# 复制依赖文件
COPY requirements.txt .

# 安装依赖
RUN pip install --no-cache-dir -r requirements.txt

# 复制应用代码
COPY app/ ./app/

# 暴露端口
EXPOSE 8000

# 启动命令
CMD ["python", "-m", "fastmcp", "run", "app.main:mcp", "--host", "0.0.0.0", "--port", "8000"]
```

#### 6.3.2 生产环境配置

```python
# app/main.py
from fastmcp import FastMCP
import os

# 根据环境变量配置
is_production = os.getenv("ENVIRONMENT") == "production"

mcp = FastMCP("My App")

# 注册所有工具
from app.tools.math_tools import register_math_tools
from app.tools.data_tools import register_data_tools

register_math_tools(mcp)
register_data_tools(mcp)

if __name__ == "__main__":
    # 生产环境使用更严格的配置
    if is_production:
        mcp.run(
            host="0.0.0.0",
            port=8000,
            log_level="warning",
            reload=False
        )
    else:
        mcp.run(
            host="127.0.0.1", 
            port=8000,
            log_level="info",
            reload=True
        )
```

---

## 第七部分：实战案例

### 7.1 案例一：CLI 助手

```python
# app/tools/cli_tools.py
from fastmcp import FastMCP, Depends
from typing import List
import subprocess
import shlex

# 安全执行命令的依赖
def get_safe_executor():
    """获取安全的命令执行器"""
    allowed_commands = {
        'ls', 'pwd', 'date', 'whoami', 'echo',
        'find', 'grep', 'cat', 'head', 'tail'
    }
    
    def safe_execute(command: str) -> dict:
        """安全执行命令"""
        parts = shlex.split(command)
        if not parts or parts[0] not in allowed_commands:
            return {
                "success": False,
                "error": f"Command not allowed: {parts[0] if parts else 'empty'}"
            }
        
        try:
            result = subprocess.run(
                parts,
                capture_output=True,
                text=True,
                timeout=30
            )
            return {
                "success": True,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "returncode": result.returncode
            }
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    return safe_execute

def register_cli_tools(mcp: FastMCP):
    """注册CLI工具"""
    
    @mcp.tool()
    def execute_command(
        command: str, 
        executor = Depends(get_safe_executor)
    ) -> dict:
        """安全执行Shell命令"""
        return executor(command)
    
    @mcp.tool()
    def list_files(directory: str = ".") -> dict:
        """列出目录文件"""
        return execute_command(f"ls -la {shlex.quote(directory)}")
```

### 7.2 案例二：API 集成器

```python
# app/tools/api_tools.py
from fastmcp import FastMCP
import aiohttp
import asyncio
from typing import List, Dict
import json

def register_api_tools(mcp: FastMCP):
    """注册API工具"""
    
    @mcp.tool()
    async def fetch_weather(city: str) -> dict:
        """获取城市天气信息"""
        # 这里使用模拟数据，实际应该调用天气API
        await asyncio.sleep(0.5)  # 模拟网络延迟
        
        weather_data = {
            "city": city,
            "temperature": 22.5,
            "condition": "Sunny",
            "humidity": 65,
            "wind_speed": 3.2
        }
        
        return weather_data
    
    @mcp.tool()
    async def send_email(to: str, subject: str, body: str) -> dict:
        """发送邮件"""
        # 模拟邮件发送
        await asyncio.sleep(1)
        
        return {
            "success": True,
            "to": to,
            "subject": subject,
            "message": "Email sent successfully"
        }
    
    @mcp.tool()
    async def search_web(query: str, max_results: int = 5) -> List[Dict]:
        """网页搜索"""
        # 模拟搜索结果
        await asyncio.sleep(0.8)
        
        results = []
        for i in range(max_results):
            results.append({
                "title": f"Result {i+1} for '{query}'",
                "url": f"https://example.com/result/{i+1}",
                "snippet": f"This is a snippet about {query} from result {i+1}"
            })
        
        return results
```

### 7.3 案例三：数据分析助手

```python
# app/tools/analysis_tools.py
from fastmcp import FastMCP
from typing import List, Dict
import pandas as pd
import numpy as np
from io import StringIO

def register_analysis_tools(mcp: FastMCP):
    """注册数据分析工具"""
    
    @mcp.tool()
    def analyze_csv_data(csv_content: str) -> dict:
        """分析CSV数据"""
        try:
            # 从字符串读取CSV
            df = pd.read_csv(StringIO(csv_content))
            
            analysis = {
                "row_count": len(df),
                "column_count": len(df.columns),
                "columns": list(df.columns),
                "data_types": df.dtypes.to_dict(),
                "summary_stats": {}
            }
            
            # 为数值列计算统计信息
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                analysis["summary_stats"][col] = {
                    "mean": float(df[col].mean()),
                    "std": float(df[col].std()),
                    "min": float(df[col].min()),
                    "max": float(df[col].max()),
                    "null_count": int(df[col].isnull().sum())
                }
            
            return {"success": True, "analysis": analysis}
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    @mcp.tool()
    def generate_sample_data(rows: int = 100) -> dict:
        """生成示例数据"""
        np.random.seed(42)
        
        data = {
            'id': range(1, rows + 1),
            'value1': np.random.normal(50, 15, rows),
            'value2': np.random.exponential(2, rows),
            'category': np.random.choice(['A', 'B', 'C'], rows),
            'timestamp': pd.date_range('2024-01-01', periods=rows, freq='H')
        }
        
        df = pd.DataFrame(data)
        csv_content = df.to_csv(index=False)
        
        return {
            "success": True,
            "csv_content": csv_content,
            "description": f"Generated {rows} rows of sample data"
        }
```

---

## 第八部分：测试与调试

### 8.1 单元测试

```python
# tests/test_tools.py
import pytest
from fastmcp import FastMCP
from app.tools.math_tools import register_math_tools

@pytest.fixture
def test_mcp():
    """创建测试用的 FastMCP 实例"""
    mcp = FastMCP("Test")
    register_math_tools(mcp)
    return mcp

def test_add_numbers(test_mcp):
    """测试加法工具"""
    result = test_mcp.tools["add_numbers"].function(2, 3)
    assert result == 5
    
    result = test_mcp.tools["add_numbers"].function(-1, 1)
    assert result == 0

def test_calculate_statistics(test_mcp):
    """测试统计工具"""
    numbers = [1, 2, 3, 4, 5]
    result = test_mcp.tools["calculate_statistics"].function(numbers)
    
    assert result["mean"] == 3.0
    assert result["min"] == 1
    assert result["max"] == 5
    assert result["count"] == 5

def test_empty_statistics(test_mcp):
    """测试空列表统计"""
    result = test_mcp.tools["calculate_statistics"].function([])
    assert "error" in result
```

### 8.2 集成测试

```python
# tests/test_integration.py
import pytest
import asyncio
from fastmcp import FastMCP
from app.main import mcp

@pytest.mark.asyncio
async def test_complete_workflow():
    """测试完整的工作流程"""
    # 这里应该使用 FastMCP 的测试客户端
    # 以下为概念性代码
    
    # 模拟用户输入
    user_input = "请计算 5 和 10 的和"
    
    # 模拟 LLM 响应（应该调用工具）
    # 实际测试中应该使用 mock
    
    # 验证工具被正确调用
    # 验证最终结果
    
    pass
```

### 8.3 调试技巧

#### 8.3.1 日志调试

```python
import logging

# 启用详细日志
logging.basicConfig(level=logging.DEBUG)

# 在工具中添加调试信息
@mcp.tool()
def debug_tool(param: str) -> str:
    """调试工具"""
    logging.debug(f"Tool called with param: {param}")
    
    try:
        result = process_param(param)
        logging.debug(f"Tool completed successfully: {result}")
        return result
    except Exception as e:
        logging.error(f"Tool failed: {e}", exc_info=True)
        raise
```

#### 8.3.2 使用 FastMCP 客户端测试

```python
# debug_client.py
import asyncio
from fastmcp import Client

async def test_tool():
    """使用 FastMCP 客户端测试工具"""
    async with Client("http://localhost:8000") as client:
        # 直接调用工具
        result = await client.call_tool(
            name="add_numbers",
            arguments={"a": 5, "b": 10}
        )
        print(f"Result: {result}")

if __name__ == "__main__":
    asyncio.run(test_tool())
```

### 8.4 性能监控与优化

```python
# app/utils/monitoring.py
import time
from functools import wraps
from typing import Callable, Any

def track_performance(func: Callable) -> Callable:
    """性能监控装饰器"""
    @wraps(func)
    async def async_wrapper(*args, **kwargs):
        start_time = time.time()
        try:
            result = await func(*args, **kwargs)
            end_time = time.time()
            print(f"{func.__name__} executed in {end_time - start_time:.3f}s")
            return result
        except Exception as e:
            end_time = time.time()
            print(f"{func.__name__} failed after {end_time - start_time:.3f}s: {e}")
            raise
    
    @wraps(func)
    def sync_wrapper(*args, **kwargs):
        start_time = time.time()
        try:
            result = func(*args, **kwargs)
            end_time = time.time()
            print(f"{func.__name__} executed in {end_time - start_time:.3f}s")
            return result
        except Exception as e:
            end_time = time.time()
            print(f"{func.__name__} failed after {end_time - start_time:.3f}s: {e}")
            raise
    
    return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper

# 使用示例
@mcp.tool()
@track_performance
def monitored_tool(data: str) -> str:
    """被监控的工具"""
    # 工具逻辑
    return processed_data
```

---

## 第九部分：总结与进阶学习

### 9.1 FastMCP 核心优势总结

1. **协议抽象**: 完全封装 MCP 协议细节，开发者只需关注业务逻辑
2. **类型安全**: 基于 Pydantic 的自动验证和序列化
3. **异步优先**: 原生支持异步操作，适合 I/O 密集型应用
4. **依赖注入**: 清晰的依赖管理，便于测试和维护
5. **生产就绪**: 内置错误处理、日志、监控等生产环境特性
6. **生态集成**: 与 FastAPI 生态完美集成，可复用现有中间件

### 9.2 进阶学习路径

#### 9.2.1 源码学习

建议阅读以下核心模块源码：
- `fastmcp/main.py`: 核心 FastMCP 类实现
- `fastmcp/tools.py`: 工具装饰器和注册机制
- `fastmcp/server.py`: ASGI 服务器实现
- `fastmcp/models.py`: MCP 消息模型定义

#### 9.2.2 相关技术深入学习

1. **ASGI 协议**: 理解异步服务器网关接口
2. **WebSocket**: 掌握双向通信协议
3. **Pydantic**: 深入学习数据验证和序列化
4. **依赖注入**: 掌握现代应用架构模式
5. **异步编程**: 精通 Python asyncio

#### 9.2.3 扩展开发

1. **自定义中间件**: 开发 FastMCP 中间件
2. **协议扩展**: 实现自定义 MCP 消息类型
3. **客户端开发**: 开发 MCP 客户端应用
4. **监控集成**: 集成 Prometheus、Grafana 等监控系统

### 9.3 社区资源

1. **官方文档**: https://gofastmcp.com/
2. **GitHub 仓库**: https://github.com/prefecthq/fastmcp
3. **Discord 社区**: 加入 FastMCP 开发者社区
4. **示例项目**: 学习官方示例和社区项目

---

## 附录：常用命令速查

### 安装与运行
```bash
# 安装 FastMCP
pip install fastmcp

# 运行应用
python -m fastmcp run main:mcp

# 带调试信息运行
python -m fastmcp run main:mcp --log-level debug

# 指定端口运行
python -m fastmcp run main:mcp --port 8080
```

### 开发调试
```bash
# 安装开发依赖
pip install fastmcp[dev]

# 运行测试
pytest tests/

# 代码格式化
black app/

# 类型检查
mypy app/
```

### 生产部署
```bash
# 使用 Gunicorn 部署
gunicorn -w 4 -k uvicorn.workers.UvicornWorker app.main:mcp

# 使用 Docker 部署
docker build -t my-fastmcp-app .
docker run -p 8000:8000 my-fastmcp-app

# 使用 Docker Compose
docker-compose up -d
```

---

> **学习建议**: 建议按照从基础到高级的顺序学习，先掌握工具定义和基本概念，再深入学习内部机制和高级特性。实践是最好的学习方式，建议边学边做小项目。

祝您学习愉快！ 🚀