# FastAPI 系统性学习笔记（基于官方文档 v0.115+）

> **作者**：资深 Python & Web 开发工程师  
> **目标读者**：具备 Python 基础，希望快速掌握 FastAPI 构建高性能 API 的开发者  
> **当前 FastAPI 版本参考**：≥0.115（支持 `Annotated` 类型注解等现代特性）  
> **文档来源**：[FastAPI 官方中文文档](https://fastapi.tiangolo.com/zh/)

---

## 一、核心概念：为什么选择 FastAPI？

FastAPI 是一个现代、快速（高性能）的 Web 框架，用于构建 API，基于 **Python 3.8+** 的类型提示（type hints）构建，其核心优势包括：

| 优势 | 说明 |
|------|------|
| ⚡ 高性能 | 接近 Node.js 和 Go 的性能（得益于 Starlette 和 Pydantic） |
| 📚 自动文档 | 自动生成交互式 API 文档（Swagger UI 和 ReDoc） |
| 🔍 类型安全 | 基于 Python 类型提示，IDE 支持好，减少运行时错误 |
| ✅ 自动验证 | 请求数据自动校验，错误返回清晰 JSON 提示 |
| 🧩 依赖注入系统 | 强大灵活的依赖管理，提升代码可维护性 |
| 🔄 异步支持 | 完美支持 `async/await`，适合 I/O 密集型任务 |
| 🧱 Pydantic 深度集成 | 数据模型定义、校验、序列化一体化 |

> 💡 **一句话总结**：FastAPI = Starlette（异步框架） + Pydantic（数据模型） + 自动化魔法（文档 + 验证）

---

## 二、快速入门：从 Hello World 开始

### 1. 安装 FastAPI 与 Uvicorn

```bash
# 安装 FastAPI（包含 Starlette）
pip install fastapi

# 安装 ASGI 服务器（用于运行）
pip install "uvicorn[standard]"
```

### 2. 最简示例：`main.py`

```python
from fastapi import FastAPI

# 创建应用实例
app = FastAPI()

# 定义一个 GET 路由
@app.get("/")
async def read_root():
    return {"message": "Hello, FastAPI!"}

# 路径参数示例
@app.get("/items/{item_id}")
async def read_item(item_id: int):
    return {"item_id": item_id}
```

### 3. 启动服务

```bash
uvicorn main:app --reload
```

- `main`: Python 文件名
- `app`: FastAPI 实例变量名
- `--reload`: 开发模式下自动重启（生产环境禁用）

### 4. 访问自动文档

启动后访问：
- **Swagger UI**: `http://127.0.0.1:8000/docs`
- **ReDoc**: `http://127.0.0.1:8000/redoc`

> ✅ **关键点**：
> - 所有路由自动出现在文档中
> - 支持在线测试 API
> - 类型注解 `int` 自动转换并校验路径参数

---

## 三、核心功能详解

### 1. 路径参数（Path Parameters）

```python
from fastapi import FastAPI

app = FastAPI()

# 使用类型注解声明参数类型，自动转换和校验
@app.get("/users/{user_id}/orders/{order_id}")
async def read_order(user_id: int, order_id: str):
    return {
        "user_id": user_id,
        "order_id": order_id
    }
```

> 🔍 **说明**：`{user_id}` 是路径参数，`int` 类型会自动转换并校验。

---

### 2. 查询参数（Query Parameters）

```python
from typing import Optional
from fastapi import FastAPI

app = FastAPI()

@app.get("/items/")
async def list_items(
    skip: int = 0,           # 默认值 → 可选参数
    limit: int = 10,
    q: Optional[str] = None  # 显式可选
):
    return {
        "skip": skip,
        "limit": limit,
        "q": q
    }
```

> 🌐 请求示例：`/items/?skip=10&limit=5&q=book`

---

### 3. 请求体（Request Body）与 Pydantic 模型

```python
from pydantic import BaseModel
from typing import Optional
from fastapi import FastAPI

app = FastAPI()

# 定义数据模型
class Item(BaseModel):
    name: str
    description: Optional[str] = None
    price: float
    tax: Optional[float] = None

@app.post("/items/")
async def create_item(item: Item):  # 自动解析 JSON 请求体
    item_dict = item.dict()
    if item.tax:
        price_with_tax = item.price + item.tax
        item_dict.update({"price_with_tax": price_with_tax})
    return item_dict
```

> ✅ **优势**：
> - 自动 JSON 解析
> - 字段类型校验
> - 缺失字段报错
> - 文档中自动生成 Schema

---

### 4. 更复杂的请求体：多个参数

```python
from fastapi import FastAPI, Body

app = FastAPI()

@app.post("/items/{item_id}")
async def update_item(
    item_id: int,
    q: str = None,
    item: Item = None,                    # Pydantic 模型
    importance: int = Body(...)           # 使用 Body 显式标记
):
    result = {"item_id": item_id}
    if q:
        result.update({"q": q})
    if item:
        result.update({"item": item})
    if importance:
        result.update({"importance": importance})
    return result
```

> 💡 `Body(...)` 表示该参数必须在请求体中提供。

---

### 5. 使用 `Annotated` 类型（现代方式）

```python
from typing import Annotated
from fastapi import Body, FastAPI
from pydantic import BaseModel

app = FastAPI()

class User(BaseModel):
    username: str
    password: str

@app.post("/login/")
async def login(
    username: Annotated[str, Body(embed=True)],
    password: Annotated[str, Body(embed=True)]
):
    return {"username": username}
```

> ✅ `Annotated` 是 Python 3.9+ 推荐方式，替代旧式 `= Body(...)`

---

### 6. 数据验证与字段约束

```python
from pydantic import BaseModel, Field
from typing import List

class Item(BaseModel):
    name: str = Field(..., min_length=3, max_length=50, example="笔记本电脑")
    price: float = Field(..., gt=0, description="价格必须大于 0")
    tags: List[str] = []

@app.post("/items/")
async def create_item(item: Item):
    return item
```

> 📚 支持的约束：
> - `min_length`, `max_length`
> - `gt` (>) , `ge` (>=), `lt` (<), `le` (<=)
> - `regex`, `default`, `example` 等

---

### 7. 依赖注入（Dependency Injection）

```python
from fastapi import Depends, FastAPI, HTTPException

app = FastAPI()

# 共享依赖：数据库连接、认证逻辑等
async def common_params(q: str = None, skip: int = 0, limit: int = 10):
    return {"q": q, "skip": skip, "limit": limit}

@app.get("/items/")
async def read_items(commons: dict = Depends(common_params)):
    return commons

# 认证依赖示例
def verify_token(token: str):
    if token != "secret-token":
        raise HTTPException(status_code=403, detail="Invalid token")
    return True

@app.get("/secure-items/")
async def secure_items(token: str = Depends(verify_token)):
    return {"data": "敏感数据"}
```

> 🔑 优势：
> - 逻辑复用
> - 测试友好
> - 支持嵌套依赖

---

### 8. 异常处理（Custom Exceptions）

```python
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

app = FastAPI()

# 自定义异常类
class UnicornException(Exception):
    def __init__(self, name: str):
        self.name = name

# 全局异常处理器
@app.exception_handler(UnicornException)
async def unicorn_exception_handler(request: Request, exc: UnicornException):
    return JSONResponse(
        status_code=418,
        content={"message": f"Oops! {exc.name} did something!"}
    )

@app.get("/unicorns/{name}")
async def read_unicorn(name: str):
    if name == "yolo":
        raise UnicornException(name=name)
    return {"unicorn": name}
```

> ✅ 也可使用 `HTTPException` 抛出标准错误。

---

### 9. 中间件（Middleware）

```python
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware  # 常用中间件

app = FastAPI()

# CORS 中间件（允许跨域）
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # 允许前端域名
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 自定义中间件
@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = str(process_time)
    return response
```

---

### 10. 安全认证（Security）

```python
from fastapi import Depends, FastAPI, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from pydantic import BaseModel

app = FastAPI()

# 模拟用户数据库
fake_users_db = {
    "johndoe": {
        "username": "johndoe",
        "hashed_password": "fakehashedsecret"
    }
}

# 使用 OAuth2 的 Password 模式 + Bearer Token
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

def fake_hash_password(password: str):
    return "fakehashed" + password

@app.post("/token")
async def login(username: str, password: str):
    user = fake_users_db.get(username)
    if not user or fake_hash_password(password) != user["hashed_password"]:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return {"access_token": user["username"], "token_type": "bearer"}

@app.get("/users/me")
async def read_users_me(username: str = Depends(oauth2_scheme)):
    return {"username": username}
```

> 🔐 实际项目建议使用 `passlib` + `JWT` 实现安全令牌。

---

## 四、最佳实践

### 1. 项目结构建议

```
myapi/
├── main.py                 # FastAPI 应用入口
├── api/
│   ├── __init__.py
│   ├── v1/
│   │   ├── __init__.py
│   │   ├── endpoints/
│   │   │   ├── users.py
│   │   │   └── items.py
│   │   └── router.py       # 路由聚合
├── models/                 # Pydantic 模型
│   ├── user.py
│   └── item.py
├── schemas/                # 请求/响应模型
├── core/
│   ├── config.py           # 配置管理
│   └── security.py         # 认证逻辑
├── dependencies/           # 共享依赖
├── database/               # 数据库连接（SQLAlchemy/Asyncpg）
└── tests/                  # 单元测试
```

### 2. 环境配置（使用 `pydantic-settings`）

```python
# core/config.py
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    app_name: str = "My API"
    debug: bool = False
    database_url: str

    class Config:
        env_file = ".env"

settings = Settings()
```

### 3. 调试技巧

- 使用 `print()` 或 `logging` 输出调试信息
- 利用 IDE 断点调试（PyCharm/VSCode）
- 启用 `--reload` 实时查看代码变更
- 查看 Swagger UI 中的请求/响应示例

### 4. 部署简述

**开发环境**：
```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

**生产环境**（推荐组合）：
```bash
# 使用 Gunicorn 管理多个 Uvicorn 工作进程
gunicorn -k uvicorn.workers.UvicornWorker -w 4 -b 0.0.0.0:8000 main:app
```

> 🚀 可结合 Docker + Nginx + HTTPS 部署。

---

## 五、实战示例：简易待办事项 API

```python
# main.py
from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel
from typing import List, Optional

app = FastAPI(title="Todo API", version="1.0.0")

# 数据模型
class Todo(BaseModel):
    id: Optional[int] = None
    title: str
    completed: bool = False

# 模拟数据库
todos_db = []
todo_id_counter = 1

# 获取所有待办
@app.get("/todos/", response_model=List[Todo])
async def get_todos():
    return todos_db

# 创建待办
@app.post("/todos/", response_model=Todo, status_code=201)
async def create_todo(todo: Todo):
    global todo_id_counter
    todo.id = todo_id_counter
    todo_id_counter += 1
    todos_db.append(todo)
    return todo

# 获取单个待办
@app.get("/todos/{todo_id}", response_model=Todo)
async def get_todo(todo_id: int):
    for todo in todos_db:
        if todo.id == todo_id:
            return todo
    raise HTTPException(status_code=404, detail="Todo not found")
```

> ✅ 启动后访问 `/docs` 测试所有接口。

---

## 六、FastAPI 核心知识点速查表

| 类别 | 语法/用法 | 说明 |
|------|----------|------|
| **路由** | `@app.get("/path")` | 支持 `get`, `post`, `put`, `delete` 等 |
| **路径参数** | `{item_id}` | `item_id: int` 自动转换 |
| **查询参数** | 函数参数无类型或可选 | `q: str = None` |
| **请求体** | `item: Item` | Pydantic 模型自动解析 |
| **数据模型** | `class Item(BaseModel)` | 字段类型 + 验证 |
| **字段验证** | `Field(..., gt=0)` | 丰富校验规则 |
| **依赖注入** | `Depends(func)` | 共享逻辑，支持嵌套 |
| **异常处理** | `raise HTTPException()` | 返回标准错误响应 |
| **中间件** | `@app.middleware("http")` | 请求/响应拦截 |
| **安全** | `OAuth2PasswordBearer` | Bearer Token 认证 |
| **文档** | `/docs`, `/redoc` | 自动生成，无需配置 |

---

## 七、学习路径建议

1. ✅ **第1天**：完成官方 "Tutorial - User Guide" 前5节（Hello World → 路径/查询参数）
2. ✅ **第2天**：掌握 Pydantic 模型、请求体、数据验证
3. ✅ **第3天**：理解依赖注入、异常处理
4. ✅ **第4天**：学习安全认证（OAuth2, JWT）
5. ✅ **第5天**：实战项目（Todo API、博客系统）
6. ✅ **第6天**：集成数据库（SQLAlchemy + AsyncIO）
7. ✅ **第7天**：部署上线（Docker + Gunicorn）

> 📘 **推荐资源**：
> - [FastAPI 官方文档](https://fastapi.tiangolo.com/zh/)
> - GitHub 示例项目：`tiangolo/fastapi`
> - 扩展库：`fastapi-users`（用户管理）、`fastapi-cache`（缓存）

---

## 结语

FastAPI 是当前 Python Web 开发的“黄金标准”之一，尤其适合构建现代化的 RESTful API 或微服务。它将类型安全、高性能、开发效率完美结合。

**记住**：类型提示是 FastAPI 的灵魂。善用 `Pydantic` 和 `Annotated`，你会写出更健壮、更易维护的代码。

现在，启动你的第一个 FastAPI 项目吧！🚀



# Annotated

`Annotated` 是 Python 3.9+ 引入的一个来自 `typing` 模块的泛型工具，它在 **FastAPI** 中扮演了至关重要的角色，主要用于 **为类型添加元数据（metadata）**，同时保持类型提示的完整性。

---
`Annotated[T, metadata]` 的作用是：

> **告诉类型检查器：这个变量/参数的类型是 `T`，但还附带一些额外信息（metadata）**。

在 FastAPI 中，这些“额外信息”通常是用于 **参数校验、文档生成、依赖注入** 的声明，比如：

- 字符串长度限制
- 查询参数描述
- 路径参数约束
- 依赖项注入

---

### 🧩 基本语法

```python
from typing import Annotated
from fastapi import Query, Path

def func(param: Annotated[str, Query(description="这是一个查询参数", min_length=3))]):
    pass
```

结构：  
`Annotated[实际类型, 元数据]`  
- 实际类型：如 `str`, `int`, `list`，供编辑器和类型检查器使用  
- 元数据：如 `Query()`, `Path()`, `Depends()`，供 FastAPI 运行时使用

---

### ✅ 实际用法示例

#### 1. 替代旧式 `Query(...)` 写法（推荐新写法）

**旧写法（仍然有效）：**
```python
from fastapi import FastAPI, Query

app = FastAPI()

@app.get("/items/")
def read_items(q: str | None = Query(None, min_length=3, max_length=50)):
    return {"q": q}
```

**新写法（使用 `Annotated`，更清晰、更安全）：**
```python
from typing import Annotated
from fastapi import FastAPI, Query

app = FastAPI()

@app.get("/items/")
def read_items(q: Annotated[str | None, Query(min_length=3, max_length=50)] = None):
    return {"q": q}
```

> ✅ 优势：类型信息更明确，避免 `Query(...)` 干扰类型推断。

#### 2. 用于路径参数校验

```python
@app.get("/items/{item_id}")
def read_item(item_id: Annotated[int, Path(ge=1, description="物品ID必须是正整数")]):
    return {"item_id": item_id}
```

#### 3. 用于依赖注入

```python
from fastapi import Depends

def common_params(q: str | None = None, skip: int = 0, limit: int = 10):
    return {"q": q, "skip": skip, "limit": limit}

@app.get("/items/")
def read_items(common: Annotated[dict, Depends(common_params)]):
    return common
```

---

### ✅ 为什么推荐使用 `Annotated`？

| 优点 | 说明 |
|------|------|
| 🔍 更好的类型提示 | 编辑器能正确识别 `q: str | None`，而不是 `q: Query` |
| 📚 更清晰的代码结构 | 类型和元数据分离，逻辑更清晰 |
| 🛠️ 更安全的重构 | 避免类型误判导致的静态检查错误 |
| 📈 FastAPI 官方推荐 | 新版本教程中已全面采用 |

---

### 📌 总结

- `Annotated` 是 Python 类型系统的一个增强工具。
- 在 FastAPI 中，它让 **类型** 和 **校验/依赖/文档元数据** 分离，代码更清晰、更安全。
- 推荐在新项目中使用 `Annotated` 替代旧的 `Query(...)` 直接赋值写法。

> ✅ 简记：  
> **`Annotated[类型, 元数据]` = 类型不变 + 功能增强**

你现在可以放心地在 FastAPI 项目中使用它了！