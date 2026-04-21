# 深入解析 Go HTTP 客户端

本文基于 Go `net/http` 和 `golang.org/x/net/http2` 源码，系统性解答 HTTP 客户端底层实现的核心问题。

## 1. 协议协商：客户端如何知道服务端使用的是 HTTP/1 还是 HTTP/2？

在建立 HTTPS 连接时，Go 客户端通过 **TLS 的 ALPN（Application-Layer Protocol Negotiation，应用层协议协商）** 机制来确定服务端支持的 HTTP 版本。

### 源码分析路径
源码位置：`/src/net/http/transport.go`

当客户端使用 `Transport` 发起请求时，底层的 `dialConn` 函数会建立 TCP 连接并进行 TLS 握手。握手完成后，会检查 TLS 状态中的 `NegotiatedProtocol`。

```go
// [src/net/http/transport.go] - func (t *Transport) dialConn(...)
// ...建立底层连接后...

// 1. 获取 TLS 握手后的状态
if s := pconn.tlsState; s != nil && s.NegotiatedProtocolIsMutual && s.NegotiatedProtocol != "" {
    // 2. 检查协商出的协议（比如 "h2"）是否存在于 TLSNextProto 映射表中
    // TLSNextProto 在 Transport 初始化时，默认会被注入 "h2" 的处理函数
    if next, ok := t.TLSNextProto[s.NegotiatedProtocol]; ok {
        // 3. 如果是 HTTP/2 (h2)，调用 next 函数将当前连接“升级”并切换到 http2.Transport
        alt := next(cm.targetAddr, pconn.conn.(*tls.Conn))
        if e, ok := alt.(erringRoundTripper); ok {
            return nil, e.RoundTripErr()
        }
        // 4. 返回包装了 HTTP/2 RoundTripper 的持久化连接
        return &persistConn{t: t, cacheKey: pconn.cacheKey, alt: alt}, nil
    }
}

// 5. 如果没有协商出 "h2"，或者不是 TLS 连接，则回退/默认使用 HTTP/1.1
// 初始化用于 HTTP/1.1 的读写缓冲区
pconn.br = bufio.NewReaderSize(pconn, t.readBufferSize())
pconn.bw = bufio.NewWriterSize(persistConnWriter{pconn}, t.writeBufferSize())

// 6. 开启 HTTP/1.1 专属的读写守护协程
go pconn.readLoop()
go pconn.writeLoop()
return pconn, nil
```

**详细流程总结：**
1. 客户端在 TLS 握手的 ClientHello 中携带自己支持的 ALPN 列表（`["h2", "http/1.1"]`）。
2. 服务端在 ServerHello 中返回选定的协议（如 `h2`）。
3. 握手完成后，`dialConn` 提取 `s.NegotiatedProtocol`。
4. 如果是 `h2`，则通过 `Transport.TLSNextProto` 映射表，接管该 TCP 连接，交由 `x/net/http2` 的 Client 处理。
5. 否则，按照 HTTP/1.1 处理，初始化 `bufio` 读写器，并启动 `readLoop` 和 `writeLoop`。

---

## 2. HTTP/2 Header 发送与连续性保证

在 HTTP/2 中，由于使用了多路复用，一条 TCP 连接上可能有多个并发的 Stream。为了保证 HPACK 动态表解码状态的正确性，**同一个请求的 Header 块（如果太大被切分成多个帧）在发送时，物理传输上必须是连续的，中间绝对不能插入其他 Stream 的帧。**

### 源码分析路径
源码位置：`/x/net/http2/transport.go`

`x/net/http2` 中的 `ClientConn.writeHeaders` 负责发送 Header。它会将压缩好的 Header 字节流进行分块，首块发送 `HEADERS` 帧，后续块发送 `CONTINUATION` 帧。并且，为了保证连续性，整个发送过程被一个写互斥锁（`wmu`）严格保护。

```go
// [golang.org/x/net/http2/transport.go]
func (cs *clientStream) encodeAndWriteHeaders(req *http.Request) error {
    cc := cs.cc
    // ... 对 Header 进行 HPACK 压缩编码，结果存入 hdrs ...
    hdrs := cc.hbuf.Bytes() 
    
    // 【关键锁】：获取写入互斥锁。
    // 这保证了在发送 HEADERS 和所有的 CONTINUATION 帧期间，
    // 其他 goroutine (其他 Stream) 绝对无法往底层的 TCP 连接写入任何帧！
    cc.wmu.Lock()
    defer cc.wmu.Unlock()

    // 调用内部方法写入帧
    err = cc.writeHeaders(cs.ID, endStream, int(cc.maxFrameSize), hdrs)
    // ...
}

// [golang.org/x/net/http2/transport.go]
func (cc *ClientConn) writeHeaders(streamID uint32, endStream bool, maxFrameSize int, hdrs []byte) error {
    first := true // 标记是否是第一个帧 (HEADERS 帧)
    
    for len(hdrs) > 0 && cc.werr == nil {
        chunk := hdrs
        // 如果剩余的 Header 数据大于单帧最大负载 (通常是 16KB)
        if len(chunk) > maxFrameSize {
            chunk = chunk[:maxFrameSize] // 截断出本次要发送的 payload
        }
        hdrs = hdrs[len(chunk):] // 推进游标
        
        // 只有当剩余数据长度为 0 时，才打上 END_HEADERS 标志位
        endHeaders := len(hdrs) == 0
        
        if first {
            // 第一块：发送 HEADERS 帧
            cc.fr.WriteHeaders(HeadersFrameParam{
                StreamID:      streamID,
                BlockFragment: chunk,
                EndStream:     endStream,
                EndHeaders:    endHeaders,
            })
            first = false
        } else {
            // 超过最大帧长被切分的后续块：发送 CONTINUATION 帧
            cc.fr.WriteContinuation(streamID, endHeaders, chunk)
        }
    }
    // 强制刷入底层 TCP 句柄
    cc.bw.Flush()
    return cc.werr
}
```

**连续性保证总结：**
- **分块逻辑**：通过判断 `maxFrameSize` 进行切片，第一片发 `HEADERS`，后续发 `CONTINUATION`，直到最后一片设置 `EndHeaders = true`。
- **并发隔离**：在准备写 Header 前，调用了 `cc.wmu.Lock()`（写入锁）。在整个循环发帧并 Flush 的过程中，锁不释放。这就从底层物理机制上保证了这些 Header/Continuation 帧在 TCP 流中是紧紧挨着的。

---

## 3. HTTP/1.1 数据分段发送机制（Chunked Encoding）

在 HTTP/1.1 中，**一次请求的数据是可以分多次发送的**。
这通常发生在客户端不知道 `Body` 总大小（即无法预先计算 `Content-Length`）时。此时，Go 会自动采用 `Transfer-Encoding: chunked`（分块传输编码）机制。

### 源码分析路径
源码位置：`/src/net/http/transfer.go` 与 `/src/net/http/internal/chunked.go`

```go
// [src/net/http/transfer.go]

func (t *transferWriter) writeBody(w io.Writer) (err error) {
    // ...
    if !t.ResponseToHEAD && t.Body != nil {
        var body = t.unwrapBody()
        
        // 场景 1：采用 Chunked 分块传输（当 ContentLength 未知时）
        if chunked(t.TransferEncoding) {
            // 包装一层 ChunkedWriter
            cw := internal.NewChunkedWriter(w)
            // 通过 io.Copy 循环从 body 读数据，并写入 cw
            _, err = t.doBodyCopy(cw, body)
            if err == nil {
                // 结束时，必须调用 Close 发送 "0\r\n\r\n" 标志结束
                err = cw.Close()
            }
            
        // 这里的ContentLength == -1是为了兼容老的go版本，防御性编程
        } else if t.ContentLength == -1 {
            // ...
        } else {
            // 限制精确的读取长度，直接拷贝到 TCP 缓冲
            ncopy, err = t.doBodyCopy(w, io.LimitReader(body, t.ContentLength))
            // ...
        }
    }
    // ...
}

// [src/net/http/internal/chunked.go]
// ChunkedWriter 对单次 Write 的处理
func (cw *chunkedWriter) Write(data []byte) (n int, err error) {
    if len(data) == 0 {
        return 0, nil
    }
    // 1. 写入分块的十六进制长度和 \r\n
    if _, err = fmt.Fprintf(cw.Wire, "%x\r\n", len(data)); err != nil {
        return 0, err
    }
    // 2. 写入实际的数据块 (data)
    if n, err = cw.Wire.Write(data); err != nil {
        return
    }
    // 3. 写入尾部的 \r\n
    if _, err = io.WriteString(cw.Wire, "\r\n"); err != nil {
        return
    }
    return n, nil
}
```

**处理逻辑与连续性保证：**
- **机制：** 如果使用 Chunked 编码，每次 `Write` 会被封装成 `<长度的十六进制>\r\n<实际数据>\r\n` 的格式。
- **连续性保证：** 在 HTTP/1.1 中，连接模型是阻塞式的（Pipeline 在 Go 客户端默认不支持且极少使用）。当前 Request 拿到了这个 TCP 连接的写入权（即 `io.Writer`），在 `writeBody` 完全结束之前，当前协程一直占有该连接，其他 Request 只能在空闲连接池中排队或去新建连接。因此，单次请求拆分的多个 Chunk 在同一条 TCP 链路上绝对是按序、连续的。

---

## 4. 连接复用与请求/响应匹配机制 (HTTP/1.1 vs HTTP/2)

客户端如何在同一条 TCP 连接上发出请求后，准确无误地把收到的响应匹配回对应的请求？HTTP/1.1 和 HTTP/2 的实现有着本质的区别。

### 4.1 HTTP/1.1 的匹配：基于顺序和 Channel 传递

HTTP/1.1 基于 `persistConn`（持久连接），它是“请求-响应”串行模型。匹配机制的核心在于**顺序一一对应**和**Channel 传递**。

源码位置：`/src/net/http/transport.go`

```go
// [src/net/http/transport.go]
type persistConn struct {
    // ...
    reqch   chan requestAndChan   // writeLoop 会从这里读取 request
    writech chan writeRequest     // roundTrip 会把请求丢进这里
    // ...
}

// 核心方法：执行一次往返请求
func (pc *persistConn) roundTrip(req *transportRequest) (resp *Response, err error) {
    // 1. 创建一个专属的接收通道，用于接收这个请求对应的 Response
    resc := make(chan responseAndError)
    
    // 2. 将包含 resc 的结构体放入 reqch，通知 readLoop 准备接收该请求的响应
    pc.reqch <- requestAndChan{
        treq: req,
        ch:   resc,
        // ...
    }
    
    // 3. 阻塞等待：监听 writeErrCh(写错误) 和 resc(读响应)
    for {
        select {
        case re := <-resc:
            // 成功拿到响应，返回给调用层！
            return handleResponse(re)
        // ...
        }
    }
}

// 守护协程：负责读响应
func (pc *persistConn) readLoop() {
    for alive {
        // 1. 从 reqch 读出一个期待的请求上下文（和客户端发送的顺序严格一致）
        rc := <-pc.reqch
        
        // 2. 从 TCP 流中读取并解析 HTTP 响应
        resp, err = pc.readResponse(rc, trace)
        
        // 3. 将解析出的响应，塞回给 rc.ch (即上面 roundTrip 建立的 resc)
        select {
        case rc.ch <- responseAndError{res: resp}:
        // ...
        }
    }
}
```
**HTTP/1.1 匹配逻辑：**
连接 `persistConn` 内部维护了 `readLoop` 和 `writeLoop` 两个守护协程。调用方将 Request 放入通道，同时附带一个专属的回调 Channel（`resc`）。由于 HTTP/1.1 是按顺序发、按顺序收，`readLoop` 每次解析完一个响应，就从队列里取出一个“期待的请求回调 Channel”，把响应塞进去。

### 4.2 HTTP/2 的匹配：基于 Stream ID 的多路复用

HTTP/2 支持乱序并发。响应在 TCP 流上是以交错的帧到达的，匹配的核心是 **Stream ID 映射表**。

源码位置：`golang.org/x/net/http2/transport.go`

```go
// [golang.org/x/net/http2/transport.go]
type ClientConn struct {
    // ...
    nextStreamID uint32 // 客户端生成的流ID必须是奇数，从1开始
    streams      map[uint32]*clientStream // 【核心数据结构】：管理当前连接上所有活跃的 Stream
    // ...
}

// 1. 客户端发起请求时，分配 Stream ID 并注册到 map
func (cc *ClientConn) addStreamLocked(cs *clientStream) {
    cs.ID = cc.nextStreamID
    cc.nextStreamID += 2     // 客户端递增，保证奇数
    cc.streams[cs.ID] = cs   // 注册到 map 中！
}

// 2. 底层读守护协程，解析来自服务端的帧
func (rl *clientConnReadLoop) processDataFrame(f *DataFrame) error {
    // 根据帧头中的 Stream ID，从 map 中精准找到对应的业务请求
    cs := rl.cc.streamByID(f.StreamID)
    if cs == nil {
        // 如果找不到，可能是已被取消或丢弃的流
        return nil 
    }
    
    // 找到了！将收到的 Body 数据写入该 Stream 专属的管道 bufPipe
    _, err := cs.bufPipe.Write(f.Data())
    return err
}

func (cc *ClientConn) streamByID(id uint32) *clientStream {
    cc.mu.Lock()
    defer cc.mu.Unlock()
    return cc.streams[id]
}
```
**HTTP/2 匹配逻辑：**
1. 客户端在发起请求前，为该请求分配一个递增的奇数 `Stream ID`，并将其与对应的 `clientStream` 结构体存入 `ClientConn.streams` (map) 中。
2. 发送请求帧时，打上该 `Stream ID`。
3. `ClientConn.readLoop` 不断从 TCP 读取服务端的返回帧，通过读取帧头（9个字节）中的 `StreamID` 字段，去 `map` 中查找对应的请求，把数据精准路由到该请求的接收管道中。彻底摆脱了 HTTP/1.1 的“排队”限制。
