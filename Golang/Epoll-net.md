# Golang net/http 与 epoll 底层原理详解

## 目录

1. [I/O 多路复用基础](#1-io-多路复用基础)
   - 1.1 [为什么需要 I/O 多路复用](#11-为什么需要-io-多路复用)
   - 1.2 [单点阻塞 I/O 模型的问题](#12-单点阻塞-io-模型的问题)
   - 1.3 [多路复用解决方案](#13-多路复用解决方案)
2. [epoll 核心原理](#2-epoll-核心原理)
   - 2.1 [epoll_create：创建内核事件表](#21-epoll_create创建内核事件表)
   - 2.2 [epoll_ctl：管理事件表](#22-epoll_ctl管理事件表)
   - 2.3 [epoll_wait：等待事件就绪](#23-epoll_wait等待事件就绪)
   - 2.4 [epoll 的优势](#24-epoll-的优势)
3. [Golang netpoll 框架原理](#3-golang-netpoll-框架原理)
   - 3.1 [整体架构设计](#31-整体架构设计)
   - 3.2 [核心数据结构 pollDesc](#32-核心数据结构-polldesc)
   - 3.3 [pollCache：pollDesc 对象池](#33-polldesc缓存池-pollcache)
   - 3.4 [poll_init：初始化 epoll 实例](#34-poll_init初始化-epoll-实例)
   - 3.5 [poll_open：注册文件描述符](#35-poll_open注册文件描述符)
   - 3.6 [poll_wait：阻塞等待 I/O](#36-poll_wait阻塞等待-io)
   - 3.7 [poll_close：关闭与回收](#37-poll_close关闭与回收)
   - 3.8 [net_poll：轮询与唤醒机制](#38-net_poll轮询与唤醒机制)
4. [Golang net/http 标准库实现](#4-golang-nethttp-标准库实现)
   - 4.1 [服务端核心数据结构](#41-服务端核心数据结构)
   - 4.2 [Handler 注册机制](#42-handler-注册机制)
   - 4.3 [Server 启动流程](#43-server-启动流程)
   - 4.4 [连接处理流程](#44-连接处理流程)
   - 4.5 [客户端核心数据结构](#45-客户端核心数据结构)
   - 4.6 [请求发送流程](#46-请求发送流程)
5. [关键设计思想总结](#5-关键设计思想总结)

---

## 1. I/O 多路复用基础

### 1.1 为什么需要 I/O 多路复用

在 Linux 系统中，一切皆为文件，socket 连接也不例外，可以抽象为文件描述符（file descriptor，简称 fd）。当服务端需要处理多个客户端连接时，如何高效地管理这些 fd 就成为了一个核心问题。

**多路复用**的核心概念：
- **多路**：存在多个待服务的目标（多个 socket 连接）
- **复用**：重复利用一个执行单元为多个目标提供服务

### 1.2 单点阻塞 I/O 模型的问题

最简单的做法是为每个连接分配一个独立的线程/进程：

```c
// 伪代码示例
for {
    // 阻塞等待连接
    conn = accept(listener_fd)
    // 为每个连接创建新线程
    go handleConnection(conn)
}
```

这种方法的问题：
- **资源消耗大**：每个线程都需要独立的栈空间和内核资源
- **上下文切换频繁**：大量线程切换带来性能开销
- **扩展性差**：难以支撑海量连接

### 1.3 多路复用解决方案演进

#### 阶段一：轮询 + 非阻塞 I/O

```c
// 设置非阻塞模式
setnonblocking(fd);

for {
    for each fd in connection_list {
        err = accept(fd, nonblock);
        if err == EAGAIN {
            continue; // 当前没有连接，继续轮询
        }
        handleConnection(conn);
    }
}
```

**问题**：CPU 空转浪费资源，即使没有连接到达也要不断循环询问。

#### 阶段二：I/O 多路复用

一个线程同时监听多个 fd，当任意一个 fd 就绪时返回：

```
┌─────────────────────────────────────────────────────┐
│                    用户态进程                         │
│  ┌──────────────────────────────────────────────┐   │
│  │           I/O 多路复用接口                    │   │
│  │    select / poll / epoll                     │   │
│  └──────────────────────────────────────────────┘   │
│                        │                            │
│                        ▼                            │
│  ┌──────────────────────────────────────────────┐   │
│  │              内核态                           │   │
│  │    ──── fd1 ──── fd2 ──── fd3 ──── ...      │   │
│  │         │        │        │                  │   │
│  │         ▼        ▼        ▼                  │   │
│  │    ┌─────────────────────────────────────┐   │   │
│  │    │          就绪事件列表                 │   │   │
│  │    └─────────────────────────────────────┘   │   │
│  └──────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────┘
```

---

## 2. epoll 核心原理

epoll（Event Poll）是 Linux 提供的 I/O 多路复用技术，它以事件回调机制实现，具有高效的事件通知能力。

### 2.1 epoll_create：创建内核事件表

```c
#include <sys/epoll.h>

int epoll_create(int size);
```

功能：开辟一片**内核空间**用于承载 epoll 事件表。

epoll 事件表的内部数据结构是**红黑树**：
- **key**：文件描述符 fd
- **value**：监听事件类型 + 自定义数据

**红黑树 vs 哈希表的选择**：
- 内存连续性：红黑树节点通过指针关联，可非连续分配
- 操作性能：红黑树 O(logN) 的常数系数较低，在 fd 数量收敛时性能接近 O(1)

### 2.2 epoll_ctl：管理事件表

```c
int epoll_ctl(int epfd, int op, int fd, struct epoll_event *event);
```

操作类型：
- `EPOLL_CTL_ADD`：添加 fd 并注册监听事件
- `EPOLL_CTL_MOD`：修改 fd 监听事件类型
- `EPOLL_CTL_DEL`：删除 fd

```c
struct epoll_event {
    uint32_t events;      // 事件类型
    epoll_data_t data;    // 用户数据（通常存储指向 pollDesc 的指针）
};
```

### 2.3 epoll_wait：等待事件就绪

```c
int epoll_wait(int epfd, struct epoll_event *events, int maxevents, int timeout);
```

功能：等待注册的事件就绪，返回就绪事件列表。

- **timeout = -1**：阻塞等待
- **timeout = 0**：非阻塞立即返回
- **timeout > 0**：超时等待（毫秒）

### 2.4 epoll 的优势

| 特性 | select | poll | epoll |
|------|--------|------|-------|
| fd 数量限制 | 受限（FD_SETSIZE） | 无硬性限制 | 无硬性限制 |
| 工作方式 | 每次轮询拷贝 | 每次轮询拷贝 | 通过红黑树增量管理 |
| 时间复杂度 | O(n) | O(n) | O(logN) |
| 事件返回 | 只返回数量 | 只返回数量 | 直接返回就绪列表 |

**epoll 适用场景**：fd 基数较大且活跃度不高

---

## 3. Golang netpoll 框架原理

### 3.1 整体架构设计

Golang 在 Linux 系统下依赖 epoll 作为核心基建，但其网络 I/O 模型并非简单调用 epoll 接口，而是设计了一套独特的 **netpoll** 框架。

核心流程：

```
┌─────────────────────────────────────────────────────────────────┐
│                        netpoll 完整流程                          │
├─────────────────────────────────────────────────────────────────┤
│  poll_init   → epoll_create 创建 epoll 实例                      │
│                    ↓                                            │
│  poll_open   → epoll_ctl(ADD) 将 fd 注册到 epoll 表             │
│                    ↓                                            │
│  poll_wait   → gopark 阻塞当前 goroutine                         │
│                    ↓                                            │
│  net_poll    → epoll_wait 获取就绪事件，goready 唤醒 goroutine   │
│                    ↓                                            │
│  poll_close  → epoll_ctl(DEL) 将 fd 从 epoll 表移除             │
└─────────────────────────────────────────────────────────────────┘
```

**关键设计**：golang 的 `poll_wait` 并不直接调用 `epoll_wait`，而是通过 `gopark` 实现 goroutine 粒度的阻塞；真正的 `epoll_wait` 调用发生在 GMP 调度器轮询 `net_poll` 流程中。

### 3.2 核心数据结构 pollDesc

```go
// runtime/netpoll.go
type pollDesc struct {
    link *pollDesc       // 指向 pollCache 中下一个 pollDesc
    fd   uintptr         // 关联的文件描述符

    // 读事件状态标识器
    // 可能值：pdReady(1)、pdWait(2)、g 实例指针、0
    rg atomic.Uintptr

    // 写事件状态标识器
    // 可能值：pdReady(1)、pdWait(2)、g 实例指针、0
    wg atomic.Uintptr
}

const (
    pdReady = 1  // I/O 已就绪
    pdWait  = 2  // goroutine 阻塞等待
)
```

**状态转换图**：

```
rg/wg 状态转换：
     0 ──────────────────────────────► pdReady
     │                                    ↑
     │   (cas)                            │ (cas)
     │                                    │
     ▼                                    │
     pdWait ──────────────────────────────┘
        │
        │ gopark + netpollblockcommit
        │
        ▼
    goroutine g 被挂起
        │
        │ net_poll 收到就绪事件
        │
        ▼
    goready 唤醒 g
```

### 3.3 pollDesc 缓存池 pollCache

为避免频繁分配/释放 pollDesc，golang 设计了 pollCache 对象池：

```go
// runtime/netpoll.go
type pollCache struct {
    lock  mutex
    first *pollDesc  // 单向链表队首
}

// 分配 pollDesc
func (c *pollCache) alloc() *pollDesc {
    lock(&c.lock)
    if c.first == nil {
        // 批量预分配（每次分配 4KB / 240 字节 ≈ 17 个）
        mem := persistentalloc(...)
        for i := uintptr(0); i < n; i++ {
            pd := (*pollDesc)(add(mem, i*pdSize))
            pd.link = c.first
            c.first = pd
        }
    }
    pd := c.first
    c.first = pd.link
    unlock(&c.lock)
    return pd
}

// 释放 pollDesc
func (c *pollCache) free(pd *pollDesc) {
    lock(&c.lock)
    pd.link = c.first
    c.first = pd
    unlock(&c.lock)
}
```

### 3.4 poll_init：初始化 epoll 实例

调用链：

```
net.Listen
    → socket()
    → netFD.listenStream()
    → fd.init()
    → pollDesc.init()
    → runtime_pollServerInit()
    → netpollGenericInit()
    → netpollinit()
    → epollcreate1()
```

```go
// runtime/netpoll_epoll.go
var (
    epfd int32  // 全局 epoll 实例
)

func netpollinit() {
    epfd = epollcreate1(_EPOLL_CLOEXEC)

    // 创建 pipe 用于接收中断信号（如程序退出）
    r, w, errno := nonblockingPipe()
    netpollBreakRd = uintptr(r)
    netpollBreakWr = uintptr(w)

    // 注册 pipe 读端到 epoll，监听中断事件
    var ev epollevent
    ev.events = _EPOLLIN
    *(**uintptr)(unsafe.Pointer(&ev.data)) = &netpollBreakRd
    epollctl(epfd, _EPOLL_CTL_ADD, r, &ev)
}
```

通过 `sync.Once` 保证全局只执行一次初始化。

### 3.5 poll_open：注册文件描述符

当 socket 或 conn 创建时，需要注册到 epoll：

```go
// runtime/netpoll.go
func poll_runtime_pollOpen(fd uintptr) (*pollDesc, int) {
    pd := pollcache.alloc()  // 从池中获取 pollDesc
    lock(&pd.lock)
    pd.fd = fd
    pd.rg.Store(0)  // 读事件状态初始化
    pd.wg.Store(0)  // 写事件状态初始化
    unlock(&pd.lock)

    // epoll_ctl(ADD) 注册读+写+边触发模式
    errno := netpollopen(fd, pd)
    return pd, int(errno)
}

func netpollopen(fd uintptr, pd *pollDesc) int32 {
    var ev epollevent
    ev.events = _EPOLLIN | _EPOLLOUT | _EPOLLRDHUP | _EPOLLET
    *(**pollDesc)(unsafe.Pointer(&ev.data)) = pd
    return -epollctl(epfd, _EPOLL_CTL_ADD, int32(fd), &ev)
}
```

**触发 poll_open 的时机**：
1. `net.Listen` 创建 listener socket
2. `listener.Accept` 获取新连接 conn

### 3.6 poll_wait：阻塞等待 I/O

当 goroutine 执行的 I/O 操作未就绪时，通过 `gopark` 进入用户态阻塞：

```go
// internal/poll/fd_poll_runtime.go
func (pd *pollDesc) wait(mode int, isFile bool) error {
    if pd.runtimeCtx == 0 {
        return errors.New("waiting for unsupported file type")
    }
    res := runtime_pollWait(pd.runtimeCtx, mode)
    return nil
}

// runtime/netpoll.go
func poll_runtime_pollWait(pd *pollDesc, mode int) int {
    for !netpollblock(pd, int32(mode), false) {
        // 自旋等待
    }
    return 0
}

func netpollblock(pd *pollDesc, mode int32, waitio bool) bool {
    gpp := &pd.rg
    if mode == 'w' {
        gpp = &pd.wg
    }

    for {
        // CAS：尝试将状态从 0 改为 pdReady
        if gpp.CompareAndSwap(pdReady, 0) {
            return true  // 已就绪
        }
        // CAS：尝试将状态从 0 改为 pdWait
        if gpp.CompareAndSwap(0, pdWait) {
            break  // 成功设置为等待状态
        }
    }

    // gopark 阻塞当前 goroutine
    gopark(netpollblockcommit, unsafe.Pointer(gpp), waitReasonIOWait, traceEvGoBlockNet, 5)

    // 被唤醒后，清除状态标识
    old := gpp.Swap(0)
    return old == pdReady
}

func netpollblockcommit(gp *g, gpp unsafe.Pointer) bool {
    // 将状态从 pdWait 改为当前 goroutine 指针
    r := atomic.Casuintptr((*uintptr)(gpp), pdWait, uintptr(unsafe.Pointer(gp)))
    if r {
        atomic.Xadd(&netpollWaiters, 1)
    }
    return r
}
```

**触发 poll_wait 的场景**：
- `listener.Accept`：socket 无连接到达
- `conn.Read`：conn 无数据可读
- `conn.Write`：conn 写缓冲区满

### 3.7 poll_close：关闭与回收

```go
func poll_runtime_pollClose(pd *pollDesc) {
    netpollclose(pd.fd)   // epoll_ctl(DEL)
    pollcache.free(pd)    // 归还到池中
}

func netpollclose(fd uintptr) int32 {
    var ev epollevent
    return -epollctl(epfd, _EPOLL_CTL_DEL, int32(fd), &ev)
}
```

### 3.8 net_poll：轮询与唤醒机制

net_poll 由 GMP 调度器触发，用于获取就绪的 I/O 事件并唤醒对应的 goroutine：

```go
// runtime/netpoll_epoll.go
func netpoll(delay int64) gList {
    var waitms int32
    if delay < 0 {
        waitms = -1      // 阻塞模式
    } else if delay == 0 {
        waitms = 0       // 非阻塞模式
    } else {
        waitms = int32(delay / 1e6)  // 超时模式
    }

    var events [128]epollevent
    n := epollwait(epfd, &events[0], int32(len(events)), waitms)

    var toRun gList
    for i := int32(0); i < n; i++ {
        ev := &events[i]
        if ev.events == 0 {
            continue
        }

        // 判断事件类型
        var mode int32
        if ev.events & (_EPOLLIN | _EPOLLRDHUP | _EPOLLHUP | _EPOLLERR) != 0 {
            mode += 'r'  // 读事件
        }
        if ev.events & (_EPOLLOUT | _EPOLLHUP | _EPOLLERR) != 0 {
            mode += 'w'  // 写事件
        }

        if mode != 0 {
            pd := *(**pollDesc)(unsafe.Pointer(&ev.data))
            netpollready(&toRun, pd, mode)
        }
    }
    return toRun
}

// runtime/netpoll.go
func netpollready(toRun *gList, pd *pollDesc, mode int32) {
    var rg, wg *g
    if mode == 'r' || mode == 'r' + 'w' {
        rg = netpollunblock(pd, 'r', true)
    }
    if mode == 'w' || mode == 'r' + 'w' {
        wg = netpollunblock(pd, 'w', true)
    }
    if rg != nil {
        toRun.push(rg)
    }
    if wg != nil {
        toRun.push(wg)
    }
}

func netpollunblock(pd *pollDesc, mode int32, bool) *g {
    gpp := &pd.rg
    if mode == 'w' {
        gpp = &pd.wg
    }
    for {
        old := gpp.Load()
        if old == 0 || old == pdReady {
            return nil
        }
        // CAS 将 goroutine 指针置为 pdReady
        if gpp.CompareAndSwap(old, pdReady) {
            return (*g)(unsafe.Pointer(old))
        }
    }
}
```

**net_poll 触发时机**：
- GMP 调度器常规调度流程（非阻塞，delay=0）
- sysmon 监控线程（非阻塞）
- GC 垃圾回收期间（非阻塞）
- P 大面积空闲时（阻塞模式，仅一个 P 留守）

---

## 4. Golang net/http 标准库实现

### 4.1 服务端核心数据结构

```go
// net/http/server.go
type Server struct {
    Addr    string      // 监听地址
    Handler Handler     // 路由处理器
}

type Handler interface {
    ServeHTTP(ResponseWriter, *Request)
}

type ServeMux struct {
    mu sync.RWMutex
    m  map[string]muxEntry  // 路由表
    es []muxEntry           // 按长度排序的路由条目（用于模糊匹配）
    hosts bool
}

type muxEntry struct {
    h Handler
    pattern string
}

// 默认多路复用器
var DefaultServeMux = &defaultServeMux
var defaultServeMux ServeMux
```

### 4.2 Handler 注册机制

```go
// 注册函数路由
func HandleFunc(pattern string, handler func(ResponseWriter, *Request)) {
    DefaultServeMux.HandleFunc(pattern, handler)
}

func (mux *ServeMux) HandleFunc(pattern string, handler func(ResponseWriter, *Request)) {
    mux.Handle(pattern, HandlerFunc(handler))
}

// HandlerFunc 实现了 Handler 接口
type HandlerFunc func(ResponseWriter, *Request)

func (f HandlerFunc) ServeHTTP(w ResponseWriter, r *Request) {
    f(w, r)
}

func (mux *ServeMux) Handle(pattern string, handler Handler) {
    mux.mu.Lock()
    defer mux.mu.Unlock()

    e := muxEntry{h: handler, pattern: pattern}
    mux.m[pattern] = e

    // 以 '/' 结尾的路由加入模糊匹配列表
    if pattern[len(pattern)-1] == '/' {
        mux.es = appendSorted(mux.es, e)
    }
}
```

**模糊匹配规则**：
- 仅以 `/` 结尾的 pattern 参与模糊匹配
- 匹配时选择最长前缀匹配的 pattern

### 4.3 Server 启动流程

```go
// 一键启动服务
func ListenAndServe(addr string, handler Handler) error {
    server := &Server{Addr: addr, Handler: handler}
    return server.ListenAndServe()
}

func (srv *Server) ListenAndServe() error {
    if addr := srv.Addr; addr == "" {
        addr = ":http"
    }
    ln, err := net.Listen("tcp", addr)
    if err != nil {
        return err
    }
    return srv.Serve(ln)
}

func (srv *Server) Serve(l net.Listener) error {
    ctx := context.WithValue(baseCtx, ServerContextKey, srv)

    for {
        rw, err := l.Accept()
        if err != nil {
            return err
        }

        connCtx := ctx
        c := srv.newConn(rw)
        go c.serve(connCtx)  // 每个连接一个 goroutine
    }
}
```

### 4.4 连接处理流程

```go
func (c *conn) serve(ctx context.Context) {
    c.r = &connReader{conn: c}
    c.bufr = newBufioReader(c.r)
    c.bufw = newBufioWriterSize(checkConnErrorWriter{c}, 4<<10)

    for {
        w, err := c.readRequest(ctx)
        if err != nil {
            // 处理错误
        }

        serverHandler{c.server}.ServeHTTP(w, w.req)
        w.cancelCtx()
    }
}

func (sh serverHandler) ServeHTTP(rw ResponseWriter, req *Request) {
    handler := sh.srv.Handler
    if handler == nil {
        handler = DefaultServeMux
    }
    handler.ServeHTTP(rw, req)
}

func (mux *ServeMux) ServeHTTP(w ResponseWriter, r *Request) {
    h, _ := mux.Handler(r)
    h.ServeHTTP(w, r)
}

func (mux *ServeMux) Handler(r *Request) (h Handler, pattern string) {
    return mux.handler(r.Host, r.URL.Path)
}

func (mux *ServeMux) match(path string) (h Handler, pattern string) {
    // 精确匹配
    if v, ok := mux.m[path]; ok {
        return v.h, v.pattern
    }

    // 模糊匹配：最长前缀
    for _, e := range mux.es {
        if strings.HasPrefix(path, e.pattern) {
            return e.h, e.pattern
        }
    }
    return nil, ""
}
```

### 4.5 客户端核心数据结构

```go
type Client struct {
    Transport RoundTripper  // 传输层
    Jar       CookieJar      // Cookie 管理
    Timeout   time.Duration  // 超时设置
}

type RoundTripper interface {
    RoundTrip(*Request) (*Response, error)
}

type Transport struct {
    idleConn     map[connectMethodKey][]*persistConn  // 空闲连接池
    DialContext  func(ctx context.Context, network, addr string) (net.Conn, error)
}

type persistConn struct {
    t          *Transport
    conn       net.Conn           // 底层连接
    reqch      chan requestAndChan    // 请求 channel
    writech    chan writeRequest      // 写请求 channel
    readLoop   func()               // 读守护协程
    writeLoop  func()               // 写守护协程
}
```

### 4.6 请求发送流程

```go
func Post(url, contentType string, body io.Reader) (*Response, error) {
    return DefaultClient.Post(url, contentType, body)
}

func (c *Client) Post(url, contentType string, body io.Reader) (*Response, error) {
    req, err := NewRequest("POST", url, body)
    if err != nil {
        return nil, err
    }
    req.Header.Set("Content-Type", contentType)
    return c.Do(req)
}

func (c *Client) Do(req *Request) (*Response, error) {
    return c.do(req)
}

func (c *Client) do(req *Request) (retres *Response, reterr error) {
    for {
        resp, didTimeout, err := c.send(req, c.deadline())
        if err == nil {
            return resp, nil
        }
    }
}

func (c *Client) send(req *Request, deadline time.Time) (*Response, func() bool, error) {
    if c.Jar != nil {
        for _, cookie := range c.Jar.Cookies(req.URL) {
            req.AddCookie(cookie)
        }
    }
    resp, didTimeout, err := send(req, c.transport(), deadline)
    if c.Jar != nil {
        if rc := resp.Cookies(); len(rc) > 0 {
            c.Jar.SetCookies(req.URL, rc)
        }
    }
    return resp, didTimeout, err
}

func (t *Transport) RoundTrip(req *Request) (*Response, error) {
    return t.roundTrip(req)
}

func (t *Transport) roundTrip(req *Request) (*Response, error) {
    for {
        pconn, err := t.getConn(treq, cm)
        resp, err = pconn.roundTrip(treq)
    }
}
```

**连接获取策略**：
1. 尝试从 `idleConn` 池中复用空闲连接
2. 无可用连接则异步创建新连接

```go
func (t *Transport) getConn(treq *transportRequest, cm connectMethod) (*persistConn, error) {
    w := &wantConn{
        cm:    cm,
        key:   cm.key(),
        ready: make(chan struct{}, 1),
    }

    // 尝试复用空闲连接
    if delivered := t.queueForIdleConn(w); delivered {
        return w.pc, nil
    }

    // 异步创建新连接
    t.queueForDial(w)

    // 等待连接就绪
    select {
    case <-w.ready:
        return w.pc, w.err
    }
}
```

---

## 5. 关键设计思想总结

### 5.1 goroutine 粒度的阻塞

**问题**：传统 epoll_wait 是线程级阻塞，但 Golang 需要 goroutine 级阻塞

**解决方案**：
- `poll_wait` 流程通过 `gopark` 阻塞 goroutine
- 真正的 `epoll_wait` 在 GMP 调度器的 `net_poll` 中调用
- 就绪后通过 `goready` 唤醒对应 goroutine

### 5.2 事件状态标识器设计

pollDesc 的 `rg/wg` 字段设计精妙：
- **0**：无动作
- **pdReady (1)**：I/O 已就绪
- **pdWait (2)**：goroutine 正在等待
- **goroutine***：指向等待的 goroutine

通过 CAS 实现无锁的状态转换。

### 5.3 对象池复用

pollCache 预分配 pollDesc 避免频繁 GC：
- 每次分配约 17 个 pollDesc（4KB / 240 字节）
- 通过单向链表管理
- 无需 GC 介入即可高效复用

### 5.4 边触发模式

golang 使用 `EPOLLET`（边触发）模式：
- 只通知状态变化一次
- 需要循环处理直到返回 EAGAIN
- 配合非阻塞 I/O 使用

### 5.5 GMP 调度协同

netpoll 融入 GMP 调度体系：
- 调度器轮询时触发 net_poll
- 非阻塞模式为主，避免影响调度
- 全局保留一个 P 以阻塞模式等待 I/O

---

## 参考资料

- [万字解析 golang netpoll 底层原理](https://mp.weixin.qq.com/s/_FTvpvLIWfYzgNhOJgKypA)
- [Golang HTTP 标准库实现原理](https://zhuanlan.zhihu.com/p/609258171)
