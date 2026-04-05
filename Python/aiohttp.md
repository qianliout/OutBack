# 🐍 Python `aiohttp` 详细教程

`aiohttp` 是一个基于 `asyncio` 的异步 HTTP 客户端和服务器框架，适用于高性能、高并发的网络请求和 Web 服务开发。

> ✅ 支持：异步 GET/POST 请求、WebSocket、文件上传、流式传输、服务器端路由等  
> 🚀 优势：非阻塞、高并发、适合爬虫、API 调用、微服务等场景

---

## 📦 一、安装

```bash
pip install aiohttp
```

> 注意：`aiohttp` 依赖 `asyncio`，Python 3.7+ 推荐使用。

---

## 🧩 二、基本概念：`async` / `await`

在使用 `aiohttp` 前，必须理解异步编程的基本语法：

```python
import asyncio

async def hello():
    print("开始")
    await asyncio.sleep(1)  # 模拟耗时操作（非阻塞）
    print("1秒后")

# 运行异步函数
asyncio.run(hello())
```

> `aiohttp` 的所有网络操作都必须在 `async` 函数中使用 `await` 调用。

---

## 📡 三、aiohttp 作为 **HTTP 客户端**

### 1. 最简单的 GET 请求

```python
import aiohttp
import asyncio

async def fetch():
    async with aiohttp.ClientSession() as session:
        async with session.get('https://httpbin.org/get') as response:
            print(await response.text())

# 运行
asyncio.run(fetch())
```

#### 说明：
- `ClientSession()`：管理会话（推荐使用 `with` 上下文管理器）
- `session.get()`：发送 GET 请求
- `response.text()`：获取文本内容（也可用 `.json()` 获取 JSON）

---

### 2. 常见请求方式

```python
async def demo_requests():
    async with aiohttp.ClientSession() as session:
        # GET
        await session.get('https://api.example.com/data')

        # POST（表单）
        await session.post('https://api.example.com/login', data={'username': 'admin'})

        # POST（JSON）
        await session.post('https://api.example.com/data', json={'key': 'value'})

        # PUT / DELETE
        await session.put('https://api.example.com/item/1', json={'name': 'new'})
        await session.delete('https://api.example.com/item/1')
```

---

### 3. 添加请求头、参数、超时

```python
async def fetch_with_headers():
    headers = {
        'User-Agent': 'MyApp/1.0',
        'Authorization': 'Bearer token123'
    }
    params = {'page': 1, 'limit': 10}  # URL 参数：?page=1&limit=10
    timeout = aiohttp.ClientTimeout(total=10)  # 超时 10 秒

    async with aiohttp.ClientSession(headers=headers, timeout=timeout) as session:
        async with session.get('https://httpbin.org/get', params=params) as resp:
            print(await resp.json())
```

---

### 4. 处理响应数据

```python
async with session.get('https://httpbin.org/json') as resp:
    print(resp.status)           # 状态码：200
    print(resp.headers)          # 响应头
    print(await resp.text())     # 文本
    print(await resp.json())     # JSON（自动解析）
    print(await resp.read())     # 二进制（适合图片、文件）
```

---

### 5. 异常处理

```python
async def safe_fetch():
    async with aiohttp.ClientSession() as session:
        try:
            async with session.get('https://httpbin.org/status/404') as resp:
                resp.raise_for_status()  # 如果状态码 >= 400，抛出异常
                return await resp.json()
        except aiohttp.ClientResponseError as e:
            print(f"HTTP 错误: {e.status}")
        except aiohttp.ClientConnectorError as e:
            print(f"连接错误: {e}")
        except aiohttp.ClientTimeout as e:
            print(f"超时: {e}")
```

---

### 6. 并发请求（高并发利器）

```python
async def fetch_url(session, url):
    async with session.get(url) as resp:
        return await resp.text()

async def fetch_all():
    urls = [
        'https://httpbin.org/delay/1',
        'https://httpbin.org/delay/2',
        'https://httpbin.org/delay/1'
    ]
    async with aiohttp.ClientSession() as session:
        tasks = [fetch_url(session, url) for url in urls]
        results = await asyncio.gather(*tasks)
        for i, result in enumerate(results):
            print(f"请求 {i+1} 完成")

# 运行（3 个请求几乎同时发出，总耗时 ~2 秒）
asyncio.run(fetch_all())
```

> ⚡ 对比：同步请求要 1+2+1=4 秒，异步只需 ~2 秒！

---

## 🖥️ 四、aiohttp 作为 **Web 服务器**

`aiohttp` 也可以用来创建异步 Web 服务。

### 1. 最简单的服务器

```python
from aiohttp import web

async def handle(request):
    name = request.match_info.get('name', "Anonymous")
    text = f"Hello, {name}"
    return web.Response(text=text)

app = web.Application()
app.add_routes([web.get('/', handle), web.get('/{name}', handle)])

if __name__ == '__main__':
    web.run_app(app, port=8080)
```

> 访问：
- `http://localhost:8080` → Hello, Anonymous
- `http://localhost:8080/Alice` → Hello, Alice

---

### 2. 路由与请求处理

```python
async def hello(request):
    return web.Response(text="Hello World")

async def post_handler(request):
    data = await request.json()  # 接收 JSON
    return web.json_response({'received': data})

app = web.Application()
app.router.add_get('/hello', hello)
app.router.add_post('/api/data', post_handler)
```

---

### 3. 静态文件服务

```python
app = web.Application()
app.router.add_static('/static', path='static', name='static')
# 访问 /static/logo.png
```

---

### 4. 中间件（Middleware）

用于日志、认证等：

```python
async def logger_factory(app, handler):
    async def middleware(request):
        print(f"请求: {request.method} {request.path}")
        response = await handler(request)
        print(f"响应: {response.status}")
        return response
    return middleware

app = web.Application(middlewares=[logger_factory])
```

---

## 🔄 五、高级功能

### 1. 流式上传/下载（大文件）

```python
# 下载大文件（分块读取）
async def download_large_file():
    async with aiohttp.ClientSession() as session:
        async with session.get('https://example.com/large.zip') as resp:
            with open('large.zip', 'wb') as f:
                async for chunk in resp.content.iter_chunked(1024):
                    f.write(chunk)
```

### 2. WebSocket 客户端

```python
async def websocket_client():
    async with aiohttp.ClientSession() as session:
        async with session.ws_connect('ws://echo.websocket.org') as ws:
            await ws.send_str("Hello!")
            msg = await ws.receive()
            print(msg.data)
```

### 3. WebSocket 服务端

```python
async def websocket_handler(request):
    ws = web.WebSocketResponse()
    await ws.prepare(request)
    async for msg in ws:
        if msg.type == aiohttp.WSMsgType.TEXT:
            await ws.send_str(f"Echo: {msg.data}")
    return ws

app.router.add_get('/ws', websocket_handler)
```

---

## 🛠️ 六、最佳实践

| 建议 | 说明 |
|------|------|
| ✅ 使用 `ClientSession` 复用连接 | 减少握手开销，提升性能 |
| ✅ 并发用 `asyncio.gather()` | 高效并发请求 |
| ✅ 设置超时 | 避免请求卡死 |
| ✅ 异常处理 | 网络不稳定时更健壮 |
| ✅ 不要阻塞事件循环 | 避免在 `async` 函数中调用 `time.sleep()`、`requests.get()` 等同步操作 |

---

## 📚 七、常见问题

### ❓ 什么时候用 `aiohttp`？什么时候用 `requests`？

| 场景 | 推荐 |
|------|------|
| 单次请求、简单脚本 | `requests`（同步，简单） |
| 高并发、爬虫、API 聚合 | `aiohttp`（异步，并发强） |
| 需要 WebSocket | `aiohttp` |
| 构建 Web 服务 | `aiohttp` 或 `FastAPI`（推荐） |

> 🔁 小技巧：可以用 `requests` 写原型，`aiohttp` 做性能优化。

---

## ✅ 总结：aiohttp 核心能力

| 功能 | 是否支持 |
|------|----------|
| 异步 HTTP 请求 | ✅ |
| GET/POST/PUT/DELETE | ✅ |
| JSON、表单、文件上传 | ✅ |
| WebSocket | ✅ |
| Web 服务器 | ✅ |
| 路由、中间件 | ✅ |
| 流式传输 | ✅ |
| 高并发 | ✅✅✅ |

---

## 🎁 附：完整示例（并发爬虫）

```python
import aiohttp
import asyncio

async def fetch_title(session, url):
    try:
        async with session.get(url) as resp:
            text = await resp.text()
            # 简单提取标题（实际可用 BeautifulSoup）
            start = text.find('<title>') + 7
            end = text.find('</title>')
            return text[start:end]
    except:
        return "获取失败"

async def main():
    urls = [
        'https://httpbin.org/html',
        'https://example.com',
        'https://httpbin.org/json'
    ]
    async with aiohttp.ClientSession() as session:
        titles = await asyncio.gather(*(fetch_title(session, url) for url in urls))
        for url, title in zip(urls, titles):
            print(f"{url} -> {title}")

asyncio.run(main())
```

---

如果你已经理解，那太棒了！🎉  
如果你想进一步学习：

- 如何结合 `aiohttp` + `BeautifulSoup` 做异步爬虫
- 如何用 `aiohttp` 实现 MCP 协议服务器
- 如何部署 `aiohttp` 服务（配合 `gunicorn` + `uvloop`）

欢迎继续提问，我可以一步步带你实现！ 😊