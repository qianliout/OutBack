---
title: "fasthttp 实战与源码笔记（Go 1.25+）"
source: "https://github.com/valyala/fasthttp"
created: 2026-04-11
updated: 2026-04-13
description: "面向 Go 1.25+，从源码角度系统拆解 fasthttp：建连、连接复用、TCP 交互、收发数据，以及与 net/http 的性能对比。"
tags:
  - golang
  - http
  - fasthttp
  - network
---

## 0. 入口 Demo
```go
func handler(ctx *fasthttp.RequestCtx) {
	fmt.Fprintf(ctx, "Hello, World!")
}

func main() {
	log.Fatal(fasthttp.ListenAndServe(":8080", handler))
}

```

---

## 1. 先说结论（源码导向版）

- `fasthttp` 的定位是**高性能边界场景**，不是所有项目默认首选。
- `fasthttp` 的“快”主要来自：**连接处理模型 + 对象池复用 + 热路径低分配 + 写回 flush 策略**。
- `fasthttp` 并不实现 TCP 协议本身（拥塞控制、滑动窗口等是内核职责），它是在应用层高效使用 `net.Conn`。
- 对于业务复杂、生态兼容优先的系统，`net/http` 仍然通常更优。

---

## 2. 关键源码路径索引（先看路径再读链路）

### 2.1 `fasthttp` 关键路径

- 仓库：`/Users/liuqianli/work/golang/src/github/fasthttp`
- 服务端主链路：
  - `server.go: ListenAndServe` / `(*Server).ListenAndServe`
  - `server.go: (*Server).Serve`
  - `server.go: acceptConn`
  - `workerpool.go: (*workerPool).Serve/getCh/workerFunc/release`
  - `server.go: (*Server).serveConn`
- 收包解析：
  - `header.go: (*RequestHeader).Read/readLoop/tryRead`
  - `http.go: (*Request).ReadBody/readBodyChunked`
- 发包回写：
  - `server.go: writeResponse`
  - `http.go: (*Response).Write`
- 复用机制：
  - `server.go: acquireCtx/releaseCtx`
  - `server.go: acquireReader/releaseReader`
  - `server.go: acquireWriter/releaseWriter`

### 2.2 `net/http` 对照路径（Go 1.25）

- 标准库路径：`/usr/local/go/src/net/http/server.go`
- 关键函数：
  - `(*Server).Serve`
  - `(*conn).serve`
  - `(*conn).readRequest`
  - `newBufioReader/putBufioReader`
  - `newBufioWriterSize/putBufioWriter`

---

## 3. 从源码看：如何建立连接（Accept 到进入处理循环）

### 3.1 监听与启动入口

`server.go` 中 `ListenAndServe` 只是构造 `Server` 并调用实例方法：

```go
func ListenAndServe(addr string, handler RequestHandler) error {
	s := &Server{Handler: handler}
	return s.ListenAndServe(addr)
}

func (s *Server) ListenAndServe(addr string) error {
	ln, err := net.Listen("tcp4", addr) // 底层仍是标准库 net.Listen
	if err != nil {
		return err
	}
	return s.Serve(ln)
}
```

要点：
- 真正开始接收连接的是 `(*Server).Serve`。
- fasthttp 没有自己造 socket 协议栈，而是建立在 `net.Listener/net.Conn` 上。

### 3.2 Accept 主循环与并发上限保护

```go
// [server.go]

func (s *Server) Serve(ln net.Listener) error {
	maxWorkersCount := s.getConcurrency()
	wp := &workerPool{
		WorkerFunc:      s.serveConn,
		MaxWorkersCount: maxWorkersCount,
	}
	wp.Start() // 

	for {
		c, err := acceptConn(s, ln, &lastPerIPErrorTime)
		if err != nil { ... }
		s.setState(c, StateNew) // 连接状态进入 New
		if !wp.Serve(c) {
			s.writeFastError(c, StatusServiceUnavailable, "The connection cannot be served because Server.Concurrency limit exceeded")
			c.Close()
		}
	}
}
```

要点：
- `acceptConn` 只做接入治理，不做 HTTP 解析。
- 超出并发上限直接 503，属于“过载快速失败”。

### 3.3 `acceptConn`：接入治理（keepalive / per-IP 限制）

```go
//[server.go]

func acceptConn(s *Server, ln net.Listener, lastPerIPErrorTime *time.Time) (net.Conn, error) {
	for {
		c, err := ln.Accept()
		if err != nil { ... }

		if tc, ok := c.(connKeepAliveer); ok && s.TCPKeepalive {
			_ = tc.SetKeepAlive(true)
			if s.TCPKeepalivePeriod > 0 {
				_ = tc.SetKeepAlivePeriod(s.TCPKeepalivePeriod)
			}
		}

		if s.MaxConnsPerIP > 0 {
			pic := wrapPerIPConn(s, c) // 每 IP 连接数限制
			if pic == nil {
				continue
			}
			c = pic
		}
		return c, nil
	}
}
```

要点：
- 与 TCP 交互点：`Accept`、`SetKeepAlive`、`SetKeepAlivePeriod`。
- 这是“连接准入”阶段，不是“请求处理”阶段。

---

## 4. 从源码看：如何复用连接（以及复用哪些对象）

### 4.1 worker 复用（不是每连接临时起 goroutine）

```go
// [workerpool.go]
func (wp *workerPool) Serve(c net.Conn) bool {
	ch := wp.getCh()
	if ch == nil {
		return false
	}
	ch.ch <- c // 只是加入chan
	return true
}

func (wp *workerPool) getCh() *workerChan {
	// 先尝试从 ready 栈拿空闲 worker（FILO）
	// 没有空闲且未达上限时才创建 worker
	var ch *workerChan
	createWorker := false

	wp.lock.Lock()
	ready := wp.ready
	n := len(ready) - 1
	if n < 0 {
		if wp.workersCount < wp.MaxWorkersCount {
			createWorker = true
			wp.workersCount++
		}
	} else {
		ch = ready[n] // 从末尾取
		ready[n] = nil
		wp.ready = ready[:n]
	}
	wp.lock.Unlock()

	if ch == nil {
		if !createWorker {
			return nil
		}
		vch := wp.workerChanPool.Get()
		ch = vch.(*workerChan)
		go func() {
			wp.workerFunc(ch)
			wp.workerChanPool.Put(vch)
		}()
	}
	return ch
}

func (wp *workerPool) workerFunc(ch *workerChan) {
	for c := range ch.ch {
		if c == nil { break }
		_ = wp.WorkerFunc(c) // 实际执行 s.serveConn(c)
		_ = c.Close()
		if !wp.release(ch) { break }
	}
}
```

关键点：
```go
// [workerpool.go]
var workerChanCap = func() int {
	if runtime.GOMAXPROCS(0) == 1 {
		return 0 // GOMAXPROCS=1 时使用阻塞通道
	}
	return 1 // GOMAXPROCS>1 时使用非阻塞通道
}()

```
- `ready` 使用 FILO，倾向复用“刚刚活跃”的 worker，有利于 cache locality。
- `workerChanCap` 会根据 `GOMAXPROCS` 调整（`workerpool.go` 顶部逻辑）。

### 4.2 连接复用（HTTP keep-alive）

`serveConn` 的核心不是“处理一次请求”，而是“同一连接循环处理多次请求”：

```go
// [server.go]
func (s *Server) serveConn(c net.Conn) error {
	defer s.serveConnCleanup()
	ctx := s.acquireCtx(c) // 从池中获取 ctx

	for { // 一个 TCP 连接上循环处理多个 HTTP 请求
		...
		err = ctx.Request.Header.Read(br)
		...
		err = ctx.Request.parseURI()
		...
		s.Handler(ctx)
		...
		err = writeResponse(ctx, bw)
		...
		if connectionClose { break }
		ctx.Request.Reset()  // 清理复用对象状态，防串数据
		ctx.Response.Reset() // 清理复用对象状态，防串数据
	}

	if hijackHandler == nil {
		s.releaseCtx(ctx) // 真实源码：尾部按条件释放，而不是 defer
	}
	return nil
}
```

关键点：
- keep-alive 的本质是 `for {}` 循环与 `connectionClose` 判定。
- 复用必须配合 `Reset()`，否则高并发下会出现数据污染。

### 4.3 对象池复用：`ctx`、`bufio.Reader`、`bufio.Writer`

```go
func (s *Server) acquireCtx(c net.Conn) (ctx *RequestCtx) {
	v := s.ctxPool.Get()
	if v == nil {
		ctx = new(RequestCtx)
		ctx.s = s
	} else {
		ctx = v.(*RequestCtx)
	}
	ctx.c = c
	return ctx
}

func (s *Server) releaseCtx(ctx *RequestCtx) {
	ctx.reset()        // 归还前必须清理状态
	s.ctxPool.Put(ctx) // 放回池中复用
}

func acquireReader(ctx *RequestCtx) *bufio.Reader {
	v := ctx.s.readerPool.Get()
	if v == nil {
		n := ctx.s.ReadBufferSize
		if n <= 0 {
			n = defaultReadBufferSize
		}
		return bufio.NewReaderSize(ctx.c, n) // 首次分配底层缓冲
	}
	r := v.(*bufio.Reader)
	r.Reset(ctx.c) // 复用 reader，仅替换底层连接
	return r
}

func acquireWriter(ctx *RequestCtx) *bufio.Writer {
	v := ctx.s.writerPool.Get()
	if v == nil {
		n := ctx.s.WriteBufferSize
		if n <= 0 {
			n = defaultWriteBufferSize
		}
		return bufio.NewWriterSize(ctx.c, n) // 首次分配底层缓冲
	}
	w := v.(*bufio.Writer)
	w.Reset(ctx.c) // 复用 writer，仅替换底层连接
	return w
}
```

关键点：
- `fasthttp` 的低分配优势很大一部分来自对象池复用。
- 这也是为什么 handler 中应避免把 `ctx` 内部引用泄露到生命周期之外。
- `bufio.Reader/Writer` 复用的核心意义不是“少写几行代码”，而是减少高并发下的内存分配与 GC 抖动：
  - 每次新建 `bufio.Reader/Writer` 都会携带一块缓冲区（默认 4KB 级别），连接多时会形成大量短命对象。
  - 通过 `Reset` 复用对象，避免频繁申请/回收缓冲，降低 `allocs/op` 与 GC 扫描压力。
  - 在 keep-alive 场景下，同一连接会处理多个请求；复用后的读写缓冲可持续服务该连接，减少热路径分配。
  - 写路径上复用 `bufio.Writer` 还能稳定批量写行为，配合条件 `Flush` 降低 syscall 次数与小包发送概率。
- 与 `ReduceMemoryUsage` 的关系：
  - `ReduceMemoryUsage=false`：更倾向保留 `br/bw`，吞吐更优、内存占用更高。
  - `ReduceMemoryUsage=true`：在合适时机释放 `br/bw` 回池，降低常驻内存，但可能增加后续重新获取成本。

---

## 5. 从源码看：如何与底层 TCP 交互

### 5.1 `fasthttp` 与 TCP 的职责边界

- 建连、重传、拥塞控制、滑动窗口：内核 TCP 栈负责。
- `fasthttp` 负责：
  - 通过 `net.Listen` 建立监听；
  - 通过 `ln.Accept()` 获取连接；
  - 通过 `net.Conn` 的 deadline / keepalive / read / write 管理连接行为。

### 5.2 直接交互点（源码可见）

在 `server.go` 可直接看到典型调用：
- `net.Listen("tcp4", addr)`
- `ln.Accept()`
- `c.SetReadDeadline(...)`
- `c.SetWriteDeadline(...)`
- `c.SetDeadline(zeroTime)`
- `tc.SetKeepAlive(true)`
- `tc.SetKeepAlivePeriod(...)`
- `c.Close()`

代码片段（`serveConn` 部分）：

```go
if s.ReadTimeout > 0 {
	if err = c.SetReadDeadline(time.Now().Add(s.ReadTimeout)); err != nil {
		break
	}
}

if writeTimeout > 0 {
	if err = c.SetWriteDeadline(time.Now().Add(writeTimeout)); err != nil {
		break
	}
}
```

解读：
- 这不是 TCP 算法实现，而是**应用层通过 socket 参数影响连接行为**。
- 超时与 keepalive 配置会影响吞吐、尾延迟、连接回收速度。

---

## 6. 从源码看：如何接收数据（读路径）

### 6.1 请求头读取：`Peek + ErrNeedMore` 增量解析

```go
// [header.go]
func (h *RequestHeader) Read(r *bufio.Reader) error {
	return h.readLoop(r, true)
}

func (h *RequestHeader) readLoop(r *bufio.Reader, waitForMore bool) error {
	n := 1
	for {
		err := h.tryRead(r, n)
		if err == nil {
			return nil
		}
		if !waitForMore || err != ErrNeedMore {
			return err
		}
		n = r.Buffered() + 1 // 缓冲区不够就扩展窥探窗口继续读
	}
}

func (h *RequestHeader) tryRead(r *bufio.Reader, n int) error {
	b, err := r.Peek(n)
	if len(b) == 0 { ... }
	b = mustPeekBuffered(r)
	headersLen, errParse := h.parse(b)
	if errParse != nil { ... }
	mustDiscard(r, headersLen) // 只丢弃已解析 header 字节，剩余字节留给后续解析
	return nil
}
```

要点：
- 这正是 TCP 字节流场景下应用层“恢复 HTTP 消息边界”的关键。
- 本质是协议分帧，不是“改造 TCP 粘包算法”。

### 6.2 请求体读取：定长 / chunked / identity

```go
// [http.go]
func (req *Request) ReadBody(r *bufio.Reader, contentLength, maxBodySize int) error {
	switch {
	case contentLength >= 0:
		body, err = readBody(r, contentLength, maxBodySize, body)
	case contentLength == -1:
		body, err = readBodyChunked(r, maxBodySize, body) // Transfer-Encoding: chunked
	default:
		body, err = readBodyIdentity(r, maxBodySize, body)
	}
}

func readBodyChunked(r *bufio.Reader, maxBodySize int, dst []byte) ([]byte, error) {
	for {
		chunkSize, err := parseChunkSize(r)
		if chunkSize == 0 { return dst, err }
		dst, err = appendBodyFixedSize(r, dst, chunkSize+2) // chunk 数据 + 末尾 CRLF
		if !bytes.Equal(dst[len(dst)-2:], []byte("\r\n")) {
			return dst, ErrBrokenChunk{...}
		}
		dst = dst[:len(dst)-2]
	}
}
```

要点：
- body 读取分支清晰，且全路径都能施加 `maxBodySize` 约束。
- chunked 每个 chunk 都做 CRLF 校验，防坏包与部分协议攻击面。

---

## 7. 从源码看：如何发送数据（写路径）

### 7.1 响应序列化

```go
// [http.go]
func writeResponse(ctx *RequestCtx, w *bufio.Writer) error {
	return ctx.Response.Write(w)
}

func (resp *Response) Write(w *bufio.Writer) error {
	body := resp.bodyBytes()
	resp.Header.SetContentLength(len(body))
	if err := resp.Header.Write(w); err != nil { return err } // 先写响应头
	_, err := w.Write(body)
	return err
}
```

### 7.2 flush 策略（性能关键）

`serveConn` 中的关键逻辑：

```go
// [server.go]
func (s *Server) serveConn(c net.Conn) error {

	if br == nil || br.Buffered() == 0 || connectionClose || (s.ReduceMemoryUsage && hijackHandler == nil) {
		err = bw.Flush() // 满足条件才 flush，减少 syscall 与小包发送
		if err != nil { break }
	}
}
```

要点：
- fasthttp 不追求“每次请求必 flush”，而是做条件 flush。
- pipeline 场景下可合并更多响应字节，降低系统调用与包数。

---

## 8. 与 `net/http` 源码对比：为什么 fasthttp 常更快

这一节只保留“差异摘要”，详细解释放到第 11 节，避免重复。

### 8.1 差异摘要（源码视角）

| 维度 | `net/http` | `fasthttp` |
|---|---|---|
| 连接并发模型 | `Serve` 中每连接 `go c.serve` | Accept 后进入 `workerPool` 分派 |
| 对象复用力度 | 有 bufio 池化 | 在此基础上更激进复用 `RequestCtx`读写对象 |
| 解析路径 | 通用语义优先 | `[]byte` 热路径 + 增量解析优先 |
| 写回策略 | 通用行为优先 | 条件 `Flush`，偏吞吐优化 |
| 设计取舍 | 兼容性/生态优先 | 高并发低分配优先 |

### 8.2 代码证据（`net/http`）

`/usr/local/go/src/net/http/server.go`：

```go
func (s *Server) Serve(l net.Listener) error {
	for {
		rw, err := l.Accept()
		...
		c := s.newConn(rw)
		go c.serve(connCtx) // 每连接 goroutine
	}
}

func (c *conn) serve(ctx context.Context) {
	c.r = &connReader{conn: c, rwc: c.rwc}
	c.bufr = newBufioReader(c.r)
	c.bufw = newBufioWriterSize(checkConnErrorWriter{c}, 4<<10)
	for {
		w, err := c.readRequest(ctx)
		...
	}
}
```

结论：
- `fasthttp` 的性能收益是**系统性设计结果**，不是单点黑魔法。
- 但这个收益要在高并发、协议层开销占比较高的场景才更明显。

---

## 9. 实战建议（结合源码）

- 先用 `net/http` 跑出基线，再用相同业务逻辑对照 `fasthttp`。
- 压测至少看：QPS、P99、allocs/op、GC pause、CPU profile。
- 如果瓶颈不在 HTTP 解析与分配，迁移到 `fasthttp` 可能收益有限。
- 使用 `fasthttp` 时重点审查：
  - 是否错误持有 `RequestCtx` 生命周期外引用；
  - 是否有不必要的字符串/字节转换；
  - 是否配置了合理的 timeout / 并发上限。

---

## 10. 最终结论

面试回答建议：按“6 点 + 1 句总结”回答，不要展开成大段。

### 11.1 六个核心优化点（可直接背）

1. 并发调度：`workerPool` 复用 worker，减少高并发下调度抖动。  
2. 对象池化：`RequestCtx`、`bufio.Reader/Writer` 复用，降低 `allocs/op` 与 GC 压力。  
3. 连接复用：`serveConn` 连接内 `for` 循环处理多请求，降低建连与切换成本。  
4. 解析热路径：字节级增量解析（`Peek + parse + Discard`），减少中间对象。  
5. 写回策略：条件 `Flush`，在 pipeline 场景减少 syscall 和小包。  
6. 过载保护：并发超限快速失败 + 超时/Body 上限，避免雪崩拖慢全局。  

### 11.2 一句总结（面试收尾）

`fasthttp` 的快来自系统性优化叠加，不是单点技巧；但代价是通用接口兼容性弱于 `net/http`，所以应基于压测证据做选型。

---

## 11. `serveConn` 长函数拆解（v1.70.0）

对应源码：`server.go:2224`

这个函数难读的根因是：它把“连接生命周期管理 + 请求循环 + 协议分支 + 内存复用 + 错误/收尾”都放在一个函数里。  
建议按下面 8 段阅读，而不是从头线性硬啃。

### 12.1 阶段 A：连接级初始化

```go
func (s *Server) serveConn(c net.Conn) error {
	defer s.serveConnCleanup() // 连接结束时减少 open/concurrency 计数
	s.concurrency.Add(1)

	proto, err := s.getNextProto(c)
	if err != nil {
		return err
	}
	if handler, ok := s.nextProtos[proto]; ok {
		// ALPN 命中其它协议时，转交给 next proto 处理器
		if s.ReadTimeout > 0 || s.WriteTimeout > 0 {
			if err := c.SetDeadline(zeroTime); err != nil {
				return err
			}
		}
		return handler(c)
	}
```

理解要点：
- 先做“协议分流”，HTTP/1.1 主路径只是其中一个分支。
- `serveConnCleanup` 是连接级资源计数收尾，不是请求级收尾。

### 12.2 阶段 B：连接上下文与循环变量准备

```go
	ctx := s.acquireCtx(c) // 从 ctxPool 获取
	ctx.connTime = connTime
	isTLS := ctx.IsTLS()
	var (
		br *bufio.Reader
		bw *bufio.Writer
		hijackHandler HijackHandler
		connectionClose bool
	)
	for {
		connRequestNum++
```

理解要点：
- `ctx/br/bw` 都是“连接内可复用对象”，不是每请求都新建。
- `for` 循环是 keep-alive 的核心语义：一个连接处理多个请求。

### 12.3 阶段 C：读路径前半段（超时 + header 读取）

```go
		if connRequestNum > 1 {
			if d := s.idleTimeout(); d > 0 {
				if err = c.SetReadDeadline(time.Now().Add(d)); err != nil {
					break
				}
			}
		}
		...
		err = ctx.Request.Header.Read(br) // 增量解析 header
		if err != nil { ... break }
```

理解要点：
- 第二个请求开始，先走 idle timeout 约束。
- header 读取失败会进入错误路径，不会继续走业务 handler。

### 12.4 阶段 D：请求级配置钩子 + URI/Body 解析

```go
		if onHdrRecv := s.HeaderReceived; onHdrRecv != nil {
			reqConf := onHdrRecv(&ctx.Request.Header) // 可覆盖超时/最大 body
			...
		}
		if err = ctx.Request.parseURI(); err != nil {
			bw = s.writeErrorResponse(bw, ctx, serverName, err)
			break
		}
		if s.StreamRequestBody {
			err = ctx.Request.readBodyStream(br, maxRequestBodySize, s.GetOnly, !s.DisablePreParseMultipartForm)
		} else {
			err = ctx.Request.readLimitBody(br, maxRequestBodySize, s.GetOnly, !s.DisablePreParseMultipartForm)
		}
```

理解要点：
- `HeaderReceived` 让你按请求动态改 read/write timeout 与 body 限制。
- URI 与 body 的解析都在 handler 前完成，失败时直接错误响应。

### 12.5 阶段 E：业务执行与连接关闭判定

```go
		connectionClose = s.DisableKeepalive || ctx.Request.Header.ConnectionClose()
		...
		if continueReadingRequest {
			s.Handler(ctx) // 业务处理
		}
		...
		connectionClose = connectionClose ||
			(s.MaxRequestsPerConn > 0 && connRequestNum >= uint64(s.MaxRequestsPerConn)) ||
			ctx.Response.Header.ConnectionClose() ||
			(s.CloseOnShutdown && s.stop.Load() == 1)
```

理解要点：
- `connectionClose` 是多条件叠加结果，不是单一 header 决定。
- 包含服务器策略（`MaxRequestsPerConn`、`CloseOnShutdown`）与请求/响应语义共同决策。

### 12.6 阶段 F：写路径（writeResponse + 条件 flush）

```go
		if err = writeResponse(ctx, bw); err != nil {
			break
		}
		if br == nil || br.Buffered() == 0 || connectionClose || (s.ReduceMemoryUsage && hijackHandler == nil) {
			err = bw.Flush() // 条件刷盘：兼顾吞吐和延迟
			if err != nil {
				break
			}
		}
```

理解要点：
- 不是每次请求都立即 flush，pipeline 场景会尽量合并发送。
- 这是 fasthttp 在高吞吐压测下性能表现较强的关键点之一。

### 12.7 阶段 G：hijack 分支（为何不能 `defer releaseCtx`）

```go
		if hijackHandler != nil {
			...
			go hijackConnHandler(ctx, hjr, c, s, hijackHandler)
			err = errHijacked
			break
		}
```

理解要点：
- `ctx` 被移交给 hijack 协程，当前函数不能立刻归还 `ctxPool`。
- 所以源码尾部是条件释放：

```go
if hijackHandler == nil {
	s.releaseCtx(ctx)
}
```

### 12.8 阶段 H：循环尾与函数收尾

```go
		s.setState(c, StateIdle)
		ctx.Request.Reset()
		ctx.Response.Reset() // 连接内下一次请求复用前重置
		...
	}
	if br != nil { releaseReader(s, br) }
	if bw != nil { releaseWriter(s, bw) }
	if hijackHandler == nil { s.releaseCtx(ctx) }
	return err
}
```

理解要点：
- `Reset` 是请求级清理；`releaseReader/releaseWriter/releaseCtx` 是连接级退出清理。
- 这套“请求内 Reset + 连接尾 Release”是 fasthttp 低分配模型的关键。

---

## 12. 为什么 `fasthttp` 要重写 HTTP 层

先说结论：`fasthttp` 主要复用的是 Go 标准库 `net`（`net.Listener`、`net.Conn`），而不是直接复用 `net/http` 的请求处理实现。  
`net/http` 已经很好，但它的设计目标与 `fasthttp` 不同，所以 `fasthttp` 选择重写 HTTP 层热路径。

### 13.1 分层要分清：复用 `net`，不等于复用 `net/http`

- `fasthttp` 在建连与收发字节层面依赖标准库 `net`。
- 但在 HTTP 层（请求头解析、body 读取、响应序列化、上下文模型）采用自有实现。
- 这意味着它没有重复造 TCP 轮子，而是重做了“HTTP 应用层引擎”。

### 13.2 目标函数不同：`net/http` 重通用，`fasthttp` 重极致性能

- `net/http` 目标：标准兼容、接口统一、生态广、可维护性强。
- `fasthttp` 目标：在高并发小包场景把分配、GC、syscall 压到更低。
- 目标不同，会直接导致实现策略不同。

### 13.3 为什么不能直接套 `net/http` 的实现

从工程上看，至少有四个约束：

1. 对象生命周期模型不同  
- `net/http` 是通用 `Request/ResponseWriter` 语义。
- `fasthttp` 是 `RequestCtx` + 池化复用语义（大量 `Reset` + `sync.Pool`）。
- 两者内存/生命周期假设不同，无法无损拼接。

2. 热路径分配策略不同  
- `fasthttp` 偏向 `[]byte` 与对象池，尽量减少临时对象和逃逸。
- 若直接沿用 `net/http` 路径，很难达到同等“低分配”目标。

3. 写回策略不同  
- `fasthttp` 在 `serveConn` 中做条件 `Flush`，对 pipeline/吞吐做了专项优化。
- 这是场景化取舍，不是通用框架默认策略。

4. 兼容性取舍不同  
- `net/http` 要优先保证标准接口兼容与生态可组合性。
- `fasthttp` 愿意牺牲一部分通用接口一致性换性能上限。

### 13.4 一句话总结

- `net/http` 像“通用底盘”，适合大多数业务。
- `fasthttp` 像“性能特化底盘”，为了速度重写 HTTP 层热路径。
- 所以看起来像“重复实现”，本质是“目标不同导致的架构分叉”。

---

## 13. `fasthttp` 支持 HTTP/2 吗？体现在哪里？

先说结论（基于你当前源码版本 `v1.70.0`）：  
- `fasthttp` 核心库**默认主链路不是内建 HTTP/2 server 实现**，主路径仍是 HTTP/1.x。  
- 但它提供了协议协商与连接接管扩展点，可让你接入自定义/第三方 HTTP/2 处理器。

### 13.1 在哪里体现“不是内建 HTTP/2 主链路”

1. README 明确说明能力边界  
- `README` 中有条目：为什么 `fasthttp` 不支持 HTTP/2.0，并指向外部项目。  
- 同时对比指出 `net/http` 从 Go 1.6 起支持 HTTP/2。  

2. `serveConn` 的分支行为  
- 在 `server.go:2224` 的 `serveConn` 里，先 `getNextProto`，如果命中注册的协议处理器就转交，否则走默认 HTTP/1.x 路径。  
- 这说明核心库并没有把 HTTP/2 作为默认内建解析链路。

### 13.2 在哪里体现“可以扩展接入 HTTP/2”

1. `NextProto` 扩展点  
- `Server.NextProto(key string, nph ServeHandler)` 可以把 ALPN 协商到的协议（如 `"h2"`）映射到自定义处理器。  
- `getNextProto` 会在 TLS 握手后读取 `NegotiatedProtocol`，再进入对应 handler。  

2. `Hijack` 扩展点  
- `RequestCtx.Hijack` 注释明确写了可用于实现升级协议，例如 WebSocket、HTTP/2。  
- 这属于“连接接管后自行处理协议”，不是 `fasthttp` 主链路直接处理 HTTP/2 帧。

### 13.3 面试回答建议

- 标准回答：`fasthttp` 默认聚焦 HTTP/1.x 性能，HTTP/2 在核心库里不是默认内建处理链路；它通过 `NextProto/Hijack` 提供扩展能力，通常结合第三方实现接入。  
- 对比补充：如果项目强依赖标准化 HTTP/2 能力且希望开箱即用，`net/http` 通常更省心。
