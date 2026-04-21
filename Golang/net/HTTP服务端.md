# Go 源码 HTTP 协议与网络传输模型解析

> **核心导读**：TCP 是一种面向连接、可靠的字节流协议，它没有“消息边界”的概念。那么 HTTP 是如何在无边界的字节流中精准地切分出一个个独立的请求？请求行、Query、Header、Body 是如何逐一解析的？一个请求（Request）和响应（Response）又是如何死死绑定在同一个 TCP 连接上的？当 HTTP/1.1 面临队头阻塞时，HTTP/2 又是如何通过底层的二进制帧和 Stream ID 机制实现真正的多路复用？
> 
> 本文将从 Go 的 `net/http` 和 `golang.org/x/net/http2` 源码出发，逐行拆解这些最硬核的网络机制。

---

## 1. 理论与现实的碰撞：TCP 字节流与“粘包”

### 1.1 什么是 TCP 字节流？“粘包”是如何产生的？
TCP 在传输层只负责把字节按照顺序可靠地发给对端。它不管应用层发的是不是一个完整的 HTTP 请求。
当你调用 `Conn.Read()` 时，你可能读到：
- 半个 HTTP 请求（半包）。
- 一个半 HTTP 请求（粘包）。
- 两个完整的 HTTP 请求粘在一起。

这就引出了一个终极问题：**应用层必须自己定义一套规则，来在一维的字节流中画出一条条起止边界。**

### 1.2 HTTP 是如何在字节流中定位“起点”和“终点”的？
- **起点从哪来？**：在一个建立好的 TCP 连接上，**当前未被消费的第一个字节，就是下一个请求的起点**。
- **终点怎么定？**：
  - **HTTP/1.1** 采用**基于定界符（Delimiter）+ 长度**的混合策略：靠找 `\r\n` 定位请求行，找 `\r\n\r\n`（空行）定位 Header 结束；再靠 `Content-Length` 或 `Transfer-Encoding: chunked` 寻找 Body 的结束。
  - **HTTP/2** 采用**基于长度前缀（Length-prefixed）**的策略：每个消息被切成多个帧（Frame），每个帧固定 9 个字节的头部，前 3 个字节明确告诉你后面的负载（Payload）有多长，彻底抛弃了扫描定界符的低效做法。

---

## 2. HTTP/1.1 源码全景：从字节流到精准解析

Go 处理 HTTP/1.1 的核心模型是：**每个 TCP 连接分配一个 Goroutine，在这个 Goroutine 内跑一个串行的死循环（状态机），不断地从流中“解包 -> 处理 -> 回写”。**

### 2.1 起源：连接的建立与 `bufio.Reader` 的引入
一切始于 `Serve` 方法中的 `Accept` 循环。

```go
// [src/net/http/server.go]
func (s *Server) Serve(l net.Listener) error {
	for {
		rw, err := l.Accept() // 接收底层的 TCP 连接
		if err != nil { return err }
		
		c := s.newConn(rw)    // 包装成内部的 conn 对象
		go c.serve(connCtx)   // 每一个 TCP 连接启动一个专属 goroutine
	}
}
```

进入 `c.serve`，Go 为了解决“粘包”并提高性能，引入了最关键的容器：`bufio.Reader`。
**为什么必须用 `bufio`？**
因为我们要在字节流里找 `\r\n`，如果你每次只 `Read` 1个字节，系统调用开销太大。如果你一次 `Read` 4096 字节，很可能把**下一个请求的数据也读进来了**。
`bufio.Reader` 完美解决了这个问题：它一次性大块读取 TCP 数据到内存缓冲中。即使读多了，多出来的数据也安安静静地躺在 `bufio` 的 buffer 里，**当当前请求解析完后，buffer 里剩下的第一个字节，天然就是下一个请求的起点**。

```go
// [src/net/http/server.go]
func (c *conn) serve(ctx context.Context) {
	// c.rwc 就是底层的 net.Conn (TCP 连接)
	c.r = &connReader{conn: c, rwc: c.rwc} 
	// 将 TCP 连接包上 bufio，HTTP 解析的所有读取动作都必须针对 c.bufr！
	c.bufr = newBufioReader(c.r)
	c.bufw = newBufioWriterSize(checkConnErrorWriter{c}, 4<<10)

	for { 
		// 同一个 TCP 连接上的串行处理循环 (Keep-Alive)
		// 1. 精准地从字节流中切出一个 Request
		w, err := c.readRequest(ctx)
		if err != nil { return } // 只要某次切分错乱，直接退出并断开 TCP

		// 2. 路由分发与业务逻辑
		serverHandler{c.server}.ServeHTTP(w, w.req)
		
		// 3. 刷新响应，并清理残余的 Body
		w.finishRequest()
		if !w.shouldReuseConnection() { break }
	}
}
```

### 2.2 精准解析第一步：请求行与 URL Query 怎么解？
进入 `c.readRequest(c.bufr)`。首先要从字节流里剥离出请求行：`GET /search?q=go HTTP/1.1\r\n`。

```go
// [src/net/http/request.go]
func readRequest(b *bufio.Reader) (req *Request, err error) {
	// 使用 textproto.Reader 处理基于 \r\n 的文本协议
	tp := newTextprotoReader(b)
	req = new(Request)

	// 1. 读请求行：底层是在 bufio 里逐字节扫描，直到遇到 \n
	var s string
	if s, err = tp.ReadLine(); err != nil { return nil, err }
	
	// 2. 按空格切分："GET" | "/search?q=go" | "HTTP/1.1"
	req.Method, req.RequestURI, req.Proto, _ = parseRequestLine(s)
	
	// 3. Query 是怎么解析的？
	// URL 的 query 并不是独立的协议字段，它紧紧跟在 Path 后面。
	rawurl := req.RequestURI
	// ParseRequestURI 会按 '?' 拆分：Path="/search", RawQuery="q=go"
	if req.URL, err = url.ParseRequestURI(rawurl); err != nil {
		return nil, err
	}
	// ... 后面才会解析 Header
}
```
*注意：此时 `RawQuery` 只是个字符串 `"q=go"`。直到你在业务代码里调用 `r.ParseForm()` 或 `r.URL.Query()` 时，Go 才会调用 `url.ParseQuery` 把它按 `&` 和 `=` 拆分成 `map[string][]string`。*

### 2.3 精准解析第二步：Header 是怎么找到边界的？
请求行之后紧跟着一堆 Header。TCP 流里没有告诉你有多少个 Header，HTTP 的规矩是：**读到一个只有 `\r\n` 的空行，Header 就结束了**。

```go
// [src/net/textproto/reader.go]
func readMIMEHeader(r *Reader, maxMemory, maxHeaders int64) (MIMEHeader, error) {
	m := make(MIMEHeader, hint)
	for {
		// 读一行。readContinuedLineSlice 内部会处理折叠行（下一行以空格或 Tab 开头，属于同一个 Header）
		kv, err := r.readContinuedLineSlice(maxMemory, mustHaveFieldNameColon)
		
		// 【极其关键的定界判断】：如果读出的是空行，说明遇到了 \r\n\r\n
		if len(kv) == 0 {
			return m, err // 头部结束，成功返回！
		}
		
		// 按照冒号切分出 Key 和 Value
		k, v, ok := bytes.Cut(kv, colon)
		key, ok := canonicalMIMEHeaderKey(k) // 将 key 标准化为大写驼峰
		value := string(bytes.TrimLeft(v, " \t"))
		m[key] = append(m[key], value)
	}
}
```

### 2.4 精准解析第三步：Body 的边界怎么裁决？(Chunked 解析细节)
到了这里，流里剩下的全是 Body。怎么知道读多少？这就是 `readTransfer` 要干的活。

```go
// [src/net/http/transfer.go]
func readTransfer(msg any, r *bufio.Reader) (err error) {
	// ... 省略提取 Header 的逻辑
	
	switch {
	case t.Chunked:
		// 方式 1：Transfer-Encoding: chunked (自描述边界，优先级最高)
		t.Body = &body{src: internal.NewChunkedReader(r), hdr: msg, r: r, closing: t.Close}
		
	case realLength == 0:
		// 方式 2：没有任何 Body
		t.Body = NoBody
		
	case realLength > 0:
		// 方式 3：有明确的 Content-Length
		// io.LimitReader 就像一把剪刀，从流里精准剪出 realLength 个字节。
		// 对字节流来说，这就把它变成了“定长协议”[
]()		t.Body = &body{src: io.LimitReader(r, realLength), closing: t.Close}
		
	default:
		// 方式 4：啥都没有，但 Connection: close
		// 没法切分边界了，只能一直读，直到底层 TCP 断开 (EOF)
		if t.Close {
			t.Body = &body{src: r, closing: t.Close}
		}
	}
	return nil
}
```

**深入 Chunked 的底层解析：**
当客户端发的是 `chunked` 数据时，TCP 被切得稀碎，Go 怎么拼凑？
在 `internal/chunked.go` 中，每一次读都会先调 `beginChunk`：
```go
// [src/net/http/internal/chunked.go]
func (cr *chunkedReader) beginChunk() {
	// 1. 从流里读出一行：如 "A\r\n" (代表后面有 10 个字节)
	line, cr.err = readChunkLine(cr.r) 
	
	// 2. 去除 chunk 的可选扩展参数 (如 "A;ext=1\r\n")
	line, cr.err = removeChunkExtension(line)
	
	// 3. 将十六进制的 "A" 转成整数 10，存入 cr.n
	cr.n, cr.err = parseHexUint(line)
	
	// 4. 【核心定界】：如果读到的 chunk 大小是 0 (即 "0\r\n")
	if cr.n == 0 {
		cr.err = io.EOF // 宣告整个 Body 彻底结束！
	}
}
```
接下来，`chunkedReader.Read()` 就会严格按 `cr.n` 从底层的 `bufio.Reader` 中读取指定的字节数，读完后再读一个 `\r\n` 校验，然后继续循环，直到遇到 `0\r\n`。这就把无边界的流，通过“长度前缀块”的方式动态切分出来了。

### 2.5 Request 和 Response 是怎么绑定在同一个 TCP 上的？
当你调用 `w.Write([]byte("ok"))` 时，这句 "ok" 是怎么精确顺着那个 TCP 回去的？
在 `readRequest` 的末尾，Go 实例化了一个 `*response` 对象：
```go
// [src/net/http/server.go]
// readRequest 内部构造 w 的过程：
w := &response{
	conn:          c,        // 死死绑定当前处理循环传入的 conn (包含了底层的 net.Conn)
	req:           req,      // 绑定刚解包出来的 Request
	reqBody:       req.Body,
	handlerHeader: make(Header),
	contentLength: -1,
}
w.cw.res = w
// w.w 是一个写缓冲，它最终写入的是 checkConnErrorWriter{c}，后者内部包裹了 c.rwc (TCP 句柄)
w.w = newBufioWriterSize(&w.cw, bufferBeforeChunkingSize)
```
所以，传给 Handler 的 `w` 内部不仅握着 `req`，还握着底层那条唯一的 `TCP Conn` (`c.rwc`)。
当 Handler 执行 `w.Write(data)` 时，数据链路如下：
1. `response.Write(data)`
2. 交给内部缓冲 `w.w.Write(data)`（如果超出缓冲，会触发 Flush）
3. 触发底层的 `checkConnErrorWriter.Write`
4. 最终调用 `c.rwc.Write(p)`，也就是系统调用 `write(fd, ...)`，顺着建立的 TCP 链接原路返回。

### 2.6 多次请求如何复用同一个 TCP？(Keep-Alive 的本质)
在 `c.serve` 的 `for` 循环中，Handler 执行完毕后，会调用 `w.finishRequest()`。
- 如果你的业务代码没有把 Body 读完，`finishRequest` 会强制调用 `req.Body.Close()`，它会尽力从 `bufio.Reader` 中把剩下的 Body 字节“吃掉”（丢弃）。
- **为什么要吃掉？**因为必须保证当前 `bufio.Reader` 的指针，**刚刚好停在下一个请求的第一个字节上**。
- 如果剩的 Body 太多（比如好几兆），吃掉太浪费性能，Go 会设置 `early close` 标记（即：`Close` 时提前收手，未读到 EOF）。
- 这个“太多”在 Go 源码里有一个容忍上限：大约 `256KB`（`maxPostHandlerReadBytes = 256 << 10`）。超过这条线，就不再为了复用连接去硬吞完。
- 一旦命中 `early close`，`w.shouldReuseConnection()` 会返回不可复用：当前响应发完后，这条 TCP 连接会被关闭，而不是进入下一轮 `readRequest`。
- 最后判断 `w.shouldReuseConnection()`，如果没被标记关闭，`for` 循环进入下一轮，此时调用 `c.readRequest(ctx)` 再次去读，因为 `bufio` 里可能已经缓存了下一个请求的数据，直接复用当前 TCP 继续解包！
- **一旦某次边界错乱（少读/多读），后面的流就全毁了，Go 只能直接 `break` 关闭 TCP。**

---

## 3. HTTP/2 的降维打击：二进制帧与多路复用的源码深潜

HTTP/1.1 的上述机制虽然精妙，但太累了：**每次都要扫描 `\r\n` 找边界，且一个 TCP 只能排队一问一答（队头阻塞）**。
HTTP/2 的出现，是从“文本扫描”向“二进制定长”的降维打击。

### 3.1 帧结构（Frame）：固定 9 字节，彻底抛弃分隔符
HTTP/2 规定，所有通信全部拆分为一个个的**帧（Frame）**。每个帧都有一个**雷打不动、固定长度的 9 字节帧头**。
```
+-----------------------------------------------+
|                 长度 (24位)                    |  ← 3字节
+---------------+---------------+---------------+
|     类型 (8)  |    标志 (8)   |               |
+---------------+---------------+---------------+
|           流标识符 (31位)                      |  ← 4字节
+-----------------------------------------------+
|               帧负载 (可变长度)                 |
+-----------------------------------------------+
```

```go
// [golang.org/x/net/http2/frame.go] 帧头结构定义
type FrameHeader struct {
    Length   uint32  // 3字节：精确告诉你后面的 Payload 有多长！(最大约 16MB)
    Type     FrameType // 1字节：这是什么帧？(0x0=DATA, 0x1=HEADERS, 0x4=SETTINGS 等)
    Flags    Flags   // 1字节：标志位 (如 END_HEADERS 表示 Header 传完了)
    StreamID uint32  // 4字节：并发流的 ID (最高位保留)
}
```
**HTTP/2 的解析是如何吊打 HTTP/1.1 的？**
完全不用扫描 `\r\n`，直接定长读取：
```go
// [golang.org/x/net/http2/frame.go] 
func (fr *Framer) ReadFrame() (Frame, error) {
    // 1. 直接暴读 9 个字节，解析出帧头
    fh, err := fr.readFrameHeader()
    if err != nil { return nil, err }
    
    // 2. fh.Length 告诉你后面有多少负载，直接分配内存精准读取
    payload := make([]byte, fh.Length)
    if _, err := io.ReadFull(fr.r, payload); err != nil { return nil, err }
    
    // 3. 根据帧类型，转换成具体的结构体 (如 DataFrame, HeadersFrame)
    return fr.parseFrame(fh, payload)
}
```

### 3.2 HPACK：Header 帧的内容是怎么压缩的？
HTTP/1.1 每次都要把巨大的 Cookie 和 Header 当成纯文本发过去。HTTP/2 专门搞了 HPACK 算法。
HPACK 在 `HEADERS 帧` 的 Payload 里，装的不是文本，而是**高度压缩的二进制位运算指令**。

**HPACK 三大神器**：
1. **静态表（Static Table）**：内置 61 组最常用的 KV。比如 `:status: 200` 索引是 8，`:method: GET` 索引是 2。
2. **动态表（Dynamic Table）**：索引从 62 开始。连接双方在内存里维护一张字典。我发过一次大 Cookie 并存入动态表 62 号，下次我只发一个数字 `62`，你就知道是什么。
3. **Huffman 编码**：对字符串再压缩一波。

**源码级位运算解析示例：**
假设我们要发送 `server: nghttpx` 这个 Header。
- 查静态表，`server` 这个 key 的索引是 `54`（二进制 `110110`）。
- HPACK 规定，如果是静态表中已有的 Key，前 2 位固定为 `01`。
- 拼接起来：`01` + `110110` = `01110110`（十六进制 `0x76`）。
- **这就是魔法：用仅仅 1 个字节 `0x76`，就完整表达了 `server` 这个长达 6 字节的单词！**
- 接着后面的 `nghttpx` 字符串，首位设为 `1` 表示开启了 Huffman 编码，然后跟着长度，最后跟上被 Huffman 压得扁扁的二进制数据。最终把原本 17 字节的文本，压缩到了 8 字节（压缩率 47%）。

### 3.3 HTTP/2 是怎么多路复用的？底层并发模型大揭秘
**多路复用的本质：用 1 个 TCP 连接，承载无数个并发的逻辑 Stream。**

在 `StreamID` 的加持下，TCP 流里跑的数据变成了乱序的帧：
`[HEADERS帧, ID=1]` -> `[HEADERS帧, ID=3]` -> `[DATA帧, ID=1]` -> `[DATA帧, ID=3]`

**Go 的 HTTP/2 服务端是怎么处理这些交错的帧的？**
在 `golang.org/x/net/http2/server.go` 中，有一个主控循环：
```go
// [golang.org/x/net/http2/server.go]
func (sc *serverConn) serve() {
	// 启动一个专门读帧的 goroutine
	go sc.readFrames() 
	
	for {
		select {
		// 主控循环通过 channel 接收读帧协程发来的各种帧
		case f := <-sc.readFrameCh:
			sc.processFrameFromReader(f)
		// ...
		}
	}
}
```
当 `processFrameFromReader` 收到一个 `HEADERS 帧`（带着一个新的 StreamID 比如 1）时，它会：
1. 在内存里**创建一个新的 `*stream` 对象**，记录在 `sc.streams[1] = st` 的 Map 里。
2. 为这个 `stream` 分配一个独立的 `pipe`（类似带锁的缓冲区）。
3. **启动一个新的 Goroutine 去执行用户的 Handler**，并把这个 `stream` 包装成 `ResponseWriter` 和 `Request` 丢进去。

**Header 帧怎么知道结束？**
- Header 不是靠“预先知道总帧数”，而是靠标志位 `END_HEADERS` 判断结束。
- 如果首个 `HEADERS` 帧已经带 `END_HEADERS`，说明这个 Header Block 到此结束。
- 如果没带，就必须继续读取同一 `StreamID` 的 `CONTINUATION` 帧，直到某一帧带上 `END_HEADERS`。
- 所以本质上是：**按帧头 `Length` 切单帧边界，按 `END_HEADERS` 切整段 Header 边界**。


当后续收到 `[DATA帧, ID=1]` 时：
1. 主控循环从帧头剥离出 `StreamID=1`。
2. 去 Map 里找到 `sc.streams[1]`。
3. 把 Data 帧里的 Payload 直接写入那个 `stream` 的 `pipe` 里。
4. 正在执行 Handler 的那个 Goroutine，如果调了 `r.Body.Read()`，就会刚好从这个 `pipe` 里读出数据！

**Data 帧到底要读多少个？**
- 和 Header 不一样，Data 没有 `CONTINUATION` 的拼接语义；它就是一帧一帧独立处理。
- 服务端不会提前知道“总共有几帧”，而是持续处理当前 Stream 的 Data 帧，直到某一帧带上 `END_STREAM` 标记。
- 收到 `END_STREAM` 后，Go 会把该 Stream 的请求体收尾为 EOF，Handler 侧后续 `r.Body.Read()` 就会读到结束。
- 所以本质上是：**按帧头 `Length` 读单帧边界，按 `END_STREAM` 判定整段 Body 边界**。

**写回响应的复用：**
当两个并发的 Handler 分别调用 `w.Write()` 试图往客户端写数据时：
- 它们不能直接操作底层的 TCP `Conn`，因为会把数据写串。
- 它们会把自己的数据包装成 `DATA 帧`，打上各自的 `StreamID`，然后丢进一个调度队列（`sc.writeFrameAsync`）。
- 服务端有一个专门的 Write Goroutine，按照**流的优先级（Priority）和流量控制窗口（Flow Control Window）**，把这些帧有序地写回底层的 TCP 句柄中。

这就是 HTTP/2 的终极奥义：**把底层的 TCP 句柄完全抽象成了一条传输帧的物理高速公路，在上面跑满了无数个互不干扰的逻辑虚拟车道（Stream）。**


## 5. 附录：进阶 Q&A

### Q1：假如我发送的数据是 data1, data2, data3，TCP 会乱序到达（变成 data2, data3, data1）导致 HTTP 解析出错吗？

**结论是：绝对不会。HTTP 层永远不会看到乱序的数据。**

这是一个非常经典的疑问。在网络底层（IP层），数据包确实是通过不同的路由路径传输的，它们**完全有可能乱序到达**接收端的网卡。比如 data2 和 data3 先到了，data1 还没到。

但是，**TCP 协议的职责就是把这些乱序的包“熨平”**，它在操作系统内核层为你屏蔽了所有网络的不确定性。当 Go 语言在应用层调用 `Conn.Read()` 时，读到的数据**绝对是严格有序的**。

**TCP 是怎么做到这一点的呢？（内核层机制）**

1. **序列号 (Sequence Number)**
   当发送端 TCP 发送数据时，会给每一个字节都打上一个编号（序列号）。
   - data1 (Seq: 1 ~ 1000)
   - data2 (Seq: 1001 ~ 2000)
   - data3 (Seq: 2001 ~ 3000)

2. **内核接收缓冲区 (Receive Buffer)**
   接收端的操作系统内核维护了一个 TCP 接收缓冲区。
   假设网络中 data2 和 data3 先到达了网卡，内核检查它们的序列号（1001 和 2001），发现前面的 data1（Seq: 1）还没到。
   此时，内核会**把 data2 和 data3 暂存在缓冲区里，绝不把它们交给上层的 HTTP 应用程序（Go 程序）**。

3. **乱序重组与阻塞等待**
   上层的 Go 程序调用 `conn.Read(buf)` 试图读取数据时，会被操作系统**阻塞（挂起）**。
   直到姗姗来迟的 data1 到达网卡。内核看到 Seq: 1 到了，立刻把 data1、data2、data3 在缓冲区里拼成完整有序的 `data1, data2, data3`。
   此时，内核才会唤醒 Go 程序的 `Read` 操作，把这 3000 个字节按顺序拷贝到应用层的 `buf` 中。

**这就是为什么 HTTP 不需要处理乱序的根本原因**。HTTP 协议（无论是 HTTP/1.1 还是 HTTP/2）都是**假设底层的传输层是 100% 可靠且有序的**。HTTP 的代码逻辑里从来不需要写“等待前面的包”的代码，它只要一直 `Read` 就行了。

*(补充：但这种“等待丢失的包”的机制，正是 TCP 的致命弱点——这就是我们在正文中提到的 **TCP 层的队头阻塞**。为了等那个丢失的 data1，后面明明已经到达的 data2 和 data3 也不能被应用层使用，导致整个连接卡死。)*

---

### Q2：TCP 上的帧是乱序的，HTTP/2 是怎么组装分片 Header 和 Data 帧的？

这里有一个非常容易误解的点：**TCP 上的帧是乱序的（不同 Stream 之间交错），但同一个 Stream 内部的帧是严格有序的！**
如果一个请求的 Header 或 Body 太大，被切分成了 3 个帧发送，HTTP/2 是如何等待并组装的呢？

#### (1) Header 帧的分片组装 (HEADERS + CONTINUATION)
如果一个 Header 块超过了单帧最大长度（通常是 16KB），发送端会先发一个 `HEADERS 帧`，紧接着发送多个 `CONTINUATION 帧`（延续帧）。

**HTTP/2 规定：在发送连续的 Header/Continuation 帧期间，绝对不允许插入任何其他 Stream 的帧！必须一口气发完。**

**为什么协议要这么霸道地规定？**
因为 HPACK 解码器是**有状态的**（依赖动态表）。如果两个 Stream 的 Header 帧交错发送，HPACK 解码器的状态就会被污染，导致解码失败。所以整个 Header 块（无论切成多少帧）在物理传输上必须是连续的。

**接收端是怎么接收并组装的？**
因为发送端是一口气发完的，所以接收端在读取到没有 `END_HEADERS` 标志的 HEADERS 帧时，会让底层 Framer（帧解析器）进入**同步阻塞状态**，专门等待后续的 Continuation 帧，并且边收边进行 HPACK 解码。

```go
// [golang.org/x/net/http2/frame.go]

// 1. checkFrameOrder 会在读取每一帧的头部时进行状态机校验
func (fr *Framer) checkFrameOrder(fh FrameHeader) error {
	// 【硬核锁定校验】：如果上一个帧是未结束的 HEADERS 或 CONTINUATION 帧
	if fr.lastHeaderStream != 0 {
		// 校验1：下一个帧必须是 CONTINUATION 帧
		if fh.Type != FrameContinuation { return err }
		// 校验2：StreamID 必须跟刚才的 HeaderStream 相同！
		if fh.StreamID != fr.lastHeaderStream { return err }
	}
	
	// 更新状态：如果当前帧没有 END_HEADERS 标志，记录下 StreamID，进入“锁定状态”
	switch fh.Type {
	case FrameHeaders, FrameContinuation:
		if fh.Flags.Has(FlagHeadersEndHeaders) {
			fr.lastHeaderStream = 0 // 结束，解除锁定
		} else {
			fr.lastHeaderStream = fh.StreamID // 未结束，锁定当前 Stream
		}
	}
	return nil
}

// 2. readMetaFrame 负责把 HEADERS 和所有的 CONTINUATION 帧拼装起来，并进行 HPACK 解码
func (fr *Framer) readMetaFrame(hf *HeadersFrame) (Frame, error) {
	mh := &MetaHeadersFrame{HeadersFrame: hf}
	var hc headersOrContinuation = hf
	
	for {
		frag := hc.HeaderBlockFragment()
		
		// 边读取边进行 HPACK 解码，解码后的结果放入 mh.Fields
		hdec.Write(frag)
		
		// 如果当前帧带有 END_HEADERS 标志，直接退出循环
		if hc.HeadersEnded() {
			break
		}
		
		// 【阻塞等待】：如果没有 END_HEADERS 标志，在死循环中强制读取下一个帧！
		// 因为 checkFrameOrder 的存在，这里读到的必定是相同 StreamID 的 Continuation 帧
		f, err := fr.ReadFrame()
		hc = f.(*ContinuationFrame) 
	}
	
	return mh, nil // 返回组装完整的 MetaHeadersFrame 交给 serverConn 处理
}
```

#### (2) Data 帧的分片组装 (通过 pipe 动态消费)
与 Header 必须一口气组装完不同，**Body (DATA 帧) 是真正的流式处理，并且可以和其他 Stream 交错发送**。
假设 Stream 1 的 Body 分成了 3 个 DATA 帧，并且中间夹杂着 Stream 2 的帧。
Go 根本不需要“等待所有 DATA 帧到齐再组装”，它是通过底层的 `pipe` 实现边收边读的。

```go
// [golang.org/x/net/http2/server.go]
func (sc *serverConn) processData(f *DataFrame) error {
	id := f.Header().StreamID
	data := f.Data()
	
	// 1. 找到对应的 Stream 及其状态
	state, st := sc.state(id) 
	
	// 2. 各种前置校验：流是否空闲、是否已关闭、流量控制配额等
	// ... 
	
	if f.Length > 0 {
		if len(data) > 0 {
			// 3. 将这段 Payload 写入当前 Stream 专属的 pipe 中
			// st.body 本质是一个 pipe，写入后，如果业务层正在 Read()，数据会立刻被消费！
			st.bodyBytes += int64(len(data))
			wrote, err := st.body.Write(data)
			if err != nil {
				// 如果业务层提前关闭了请求体，直接忽略并返回流量配额
				sc.sendWindowUpdate(nil, int(f.Length)-wrote)
				return nil
			}
		}
		
		// 4. 流量控制：返还 padding 的流量配额
		pad := int32(f.Length) - int32(len(data))
		sc.sendWindowUpdate32(nil, pad)
		sc.sendWindowUpdate32(st, pad)
	}
	
	// 5. 如果这个帧带有 END_STREAM 标志，说明 Body 发完了
	if f.StreamEnded() {
		st.endStream() // 关闭 pipe，通知业务层的 Read 返回 io.EOF
	}
	return nil
}
```

**总结**：
- **Header 组装**：发送端通过 `for` 循环霸占 TCP 写锁一口气发完；接收端通过 `expectContinueStreamID` 锁，**同步阻塞式**拼接二进制碎片，拼完后一次性交由 HPACK 解码。
- **Data 组装**：通过每个 Stream 专属的 `pipe`，**异步流式**写入。主循环收到哪个 Stream 的 DATA 帧，就塞进哪个 Stream 的 pipe，由上层业务 Goroutine 按需消费。当收到 `END_STREAM` 标志时，宣告结束。

---

## 4. 总结与升华：网络协议的进化本质

从 HTTP/1.1 到 HTTP/2 的演进，本质上是**应用层在弥补 TCP 的固有缺陷**：
1. **TCP 没有边界**：HTTP/1.1 用低效的文本扫描（`\r\n\r\n`）去画边界；HTTP/2 用高效的定长二进制帧头（`Length`）去画边界。
2. **TCP 是严格串行的**：HTTP/1.1 只能强行开多个 TCP 连接去实现并发；HTTP/2 则在应用层自己搞了一套虚拟的 `Stream` 调度系统，实现了极致的单连接多路复用。

**未来的 HTTP/3**：
HTTP/2 的多路复用看似完美，但它依然基于 TCP。如果 TCP 传输中丢了一个包，操作系统的 TCP 协议栈会卡住整个连接等待重传，这导致 HTTP/2 的所有 Stream 都必须跟着一起等（**TCP 层的队头阻塞**）。
因此，HTTP/3 直接抛弃了 TCP，拥抱了基于 UDP 的 QUIC 协议。QUIC 在传输层内部实现了流级别的重传和控制，丢了 Stream 1 的包，绝不影响 Stream 2 的传输，这才是网络性能进化的下一个终点。
