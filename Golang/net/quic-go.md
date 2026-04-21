
# quic-go + HTTP/3 源码学习笔记（源码精讲版）

> 仓库：`github.com/quic-go/quic-go`  
> 目标：你只读这一份文档，也能从 `UDP` 收包一路理解到 `HTTP/3` 请求处理和响应写回。  
> 风格：每个主题都给 **代码位置 + 关键代码 + 中文注释 + 验证方法**。

---

## 1. 一张总图先建立全局感

端到端调用链（服务端）：

`UDP Socket -> Transport.listen -> Transport.handlePacket -> baseServer.handlePacketImpl -> Conn.run -> Conn.handleFrame -> streamsMap -> http3.Server.handleConn -> RawServerConn.handleRequestStream -> handler.ServeHTTP -> responseWriter`

建议你先把这个链路背下来，再看细节，不容易迷路。

---

## 2. 从 UDP 收包到连接路由

### 2.1 Transport 收包循环

代码位置：

- `transport.go:527` `(*Transport).listen`
- `transport.go:562` `(*Transport).handlePacket`

关键代码（含注释）：

```go
// transport.go
func (t *Transport) listen(conn rawConn) {
    for {
        // 从底层连接读取一个 UDP datagram（含 data + 元信息）
        p, err := conn.ReadPacket()

        // 临时错误（如某些系统调用瞬时异常）可继续读
        if nerr, ok := err.(net.Error); ok && nerr.Temporary() {
            t.logger.Debugf("Temporary error reading from conn: %w", err)
            continue
        }

        // 硬错误通常意味着 socket 不可用，关闭 transport
        if err != nil {
            t.close(err)
            return
        }

        // 进入 QUIC 包解析与 CID 分发逻辑
        t.handlePacket(p)
    }
}

func (t *Transport) handlePacket(p receivedPacket) {
    // 空包没有协议意义，直接丢弃
    if len(p.data) == 0 {
        return
    }

    // 先做快速判定，不像 QUIC 的数据走非 QUIC 分支
    if !wire.IsPotentialQUICPacket(p.data[0]) && !wire.IsLongHeaderPacket(p.data[0]) {
        t.handleNonQUICPacket(p)
        return
    }

    // 解析目标连接 ID，用于把包路由到正确 Conn
    connID, err := wire.ParseConnectionID(p.data, t.connIDLen)
    if err != nil {
        p.buffer.MaybeRelease()
        return
    }

    // 已有连接则直接投递，避免 server 慢路径
    if handler, ok := (*packetHandlerMap)(t).Get(connID); ok {
        handler.handlePacket(p)
        return
    }

    // 未知 CID 进入 server，可能是新连接 Initial 包
    t.server.handlePacket(p)
}
```

### 2.2 Server 识别 Initial 并建连

代码位置：

- `server.go:410` `(*baseServer).handlePacketImpl`
- `server.go:679` `(*baseServer).handleInitialImpl`

关键代码（含注释）：

```go
// server.go
func (s *baseServer) handlePacketImpl(p receivedPacket) bool {
    // 这里只处理可能触发新建连接的长头包
    hdr, _, _, err := wire.ParsePacket(p.data)
    if err != nil {
        return false
    }

    // 非 Initial 不参与新建连接流程
    if hdr.Type != protocol.PacketTypeInitial {
        return false
    }

    // 真正的建连逻辑在 handleInitialImpl
    if err := s.handleInitialImpl(p, hdr); err != nil {
        s.logger.Errorf("Error occurred handling initial packet: %s", err)
    }
    return true
}

func (s *baseServer) handleInitialImpl(p receivedPacket, hdr *wire.Header) error {
    // 创建 Conn，绑定发送通道、CID 管理器、TLS 等状态
    conn := s.newConn(
        ctx,
        cancel,
        newSendConn(s.conn, p.remoteAddr, p.info, s.logger),
        ...,
    )

    // 首个 Initial 包立即喂入连接，驱动握手状态机前进
    conn.handlePacket(p)

    // 把客户端 CID 和服务端新 CID 都登记进路由表
    if added := s.tr.AddWithConnID(hdr.DestConnectionID, connID, conn); !added {
        conn.closeWithTransportError(ConnectionRefused)
        return nil
    }

    // 启动连接主循环，后续包都由该 Conn 自主处理
    go conn.run()
    return nil
}
```

---

## 3. Conn 主循环：QUIC 传输层核心

代码位置：

- `connection.go:564` `(*Conn).run`
- `connection.go:1003` `(*Conn).handlePackets`
- `connection.go:1052` `(*Conn).handleOnePacket`
- `connection.go:1905` `(*Conn).handleFrame`

关键代码（含注释）：

```go
// connection.go
func (c *Conn) run() (err error) {
    // 连接级事件循环，统一处理收包、计时器、发包
runLoop:
    for {
        // 先批量处理收到的数据包，减少锁竞争和调度抖动
        processed, err := c.handlePackets()
        if err != nil {
            c.setCloseError(&closeError{err: err})
            break runLoop
        }

        // 根据 pacing / loss / ack / flow control 决定是否立即发送
        if err := c.triggerSending(monotime.Now()); err != nil {
            c.setCloseError(&closeError{err: err})
            break runLoop
        }

        // 若本轮无事可做，阻塞等待新包/定时器/发送调度
        if !processed {
            select {
            case <-c.notifyReceivedPacket:
            case <-c.timer.C:
            case <-c.sendingScheduled:
            case <-c.closeChan:
                break runLoop
            }
        }
    }
    return c.closeErr.Load().err
}

func (c *Conn) handleOnePacket(rp receivedPacket, datagramID qlog.DatagramID) (bool, error) {
    data := rp.data

    // 一个 UDP datagram 里可能合并多个 QUIC packet（coalesced）
    for len(data) > 0 {
        if wire.IsLongHeaderPacket(data[0]) {
            hdr, packetData, rest, err := wire.ParsePacket(data)
            if err != nil {
                return false, nil
            }

            // 长头包常见于握手阶段（Initial/Handshake/0-RTT）
            rp.data = packetData
            processed, err := c.handleLongHeaderPacket(rp, hdr, datagramID)
            if err != nil {
                return false, err
            }
            _ = processed
            data = rest
            continue
        }

        // 短头包通常是 1-RTT 应用数据
        processed, err := c.handleShortHeaderPacket(rp, false, datagramID)
        return processed, err
    }

    return true, nil
}
```

---

## 4. 加密与握手：TLS1.3 如何驱动 QUIC 状态机

代码位置：

- `internal/handshake/crypto_setup.go:191` `StartHandshake`
- `internal/handshake/crypto_setup.go:224` `HandleMessage`
- `internal/handshake/crypto_setup.go:309` `NextEvent`
- `connection.go:2013` `(*Conn).handleHandshakeEvents`

关键代码（含注释）：

```go
// connection.go
func (c *Conn) handleHandshakeEvents(now monotime.Time) error {
    for {
        // 从握手模块拉取事件（读密钥就绪、TP 到达、需写 CRYPTO 数据等）
        ev := c.cryptoStreamHandler.NextEvent()

        switch ev.Kind {
        case handshake.EventNoEvent:
            return nil

        case handshake.EventReceivedTransportParameters:
            // 应用对端传输参数（流上限、max_udp_payload_size 等）
            if err := c.handleTransportParameters(ev.TransportParameters); err != nil {
                return err
            }

        case handshake.EventReceivedReadKeys:
            // 新密钥到位后，重试此前“不可解密”缓存包
            c.undecryptablePacketsToProcess = append(c.undecryptablePacketsToProcess, c.undecryptablePackets...)
            c.undecryptablePackets = nil

        case handshake.EventWriteInitialData:
            // 把 TLS 产生的数据写入 QUIC CRYPTO stream
            if _, err := c.initialStream.Write(ev.Data); err != nil {
                return err
            }

        case handshake.EventWriteHandshakeData:
            if _, err := c.handshakeStream.Write(ev.Data); err != nil {
                return err
            }
        }
    }
}
```

---

## 5. 传输可靠性：ACK、丢包、重传

### 5.1 ACK 处理

代码位置：`internal/ackhandler/sent_packet_handler.go:378` `ReceivedAck`

```go
// sent_packet_handler.go
func (h *sentPacketHandler) ReceivedAck(ack *wire.AckFrame, encLevel protocol.EncryptionLevel, rcvTime monotime.Time) (bool, error) {
    // 遍历 ACK 区间，把已确认包从 outstanding 集合移除
    // 更新 RTT 估计（latest/smoothed/variance）
    // 通知拥塞控制器已确认字节，可能增大 cwnd
    ...
}
```

### 5.2 丢包检测

代码位置：

- `internal/ackhandler/sent_packet_handler.go:787` `detectLostPackets`
- `internal/ackhandler/sent_packet_handler.go:867` `OnLossDetectionTimeout`

```go
// sent_packet_handler.go
func (h *sentPacketHandler) detectLostPackets(now monotime.Time, encLevel protocol.EncryptionLevel) {
    // 按“时间阈值 + 包号阈值”综合判定丢包
    // 丢失的包会触发 frame 重传入队和拥塞事件
    ...
}

func (h *sentPacketHandler) OnLossDetectionTimeout(now monotime.Time) error {
    // PTO/loss 定时器触发时执行恢复动作（探测包/重传）
    ...
}
```

### 5.3 重传队列

代码位置：

- `internal/ackhandler/sent_packet_handler.go:1056` `queueFramesForRetransmission`
- `retransmission_queue.go:78` `GetFrame`

```go
// sent_packet_handler.go
func (h *sentPacketHandler) queueFramesForRetransmission(p *packet) {
    // 把丢失包里的可重传 frame 放回 retransmissionQueue
    // 并非所有帧都重传，需按协议语义过滤
    ...
}

// retransmission_queue.go
func (q *retransmissionQueue) GetFrame(encLevel protocol.EncryptionLevel, maxLen protocol.ByteCount, v protocol.Version) wire.Frame {
    // 根据加密级别和剩余空间，取一帧给 packer 重新发送
    ...
}
```

---

## 6. 流量控制（Flow Control）

### 6.1 Stream 级窗口

代码位置：

- `internal/flowcontrol/stream_flow_controller.go:49` `UpdateHighestReceived`
- `internal/flowcontrol/stream_flow_controller.go:101` `AddBytesRead`
- `internal/flowcontrol/stream_flow_controller.go:138` `GetWindowUpdate`

```go
// stream_flow_controller.go
func (c *streamFlowController) UpdateHighestReceived(offset protocol.ByteCount, final bool, now monotime.Time) error {
    // 更新“对端最多发送到哪里”，超过窗口则协议错误
    // 同时处理 FIN/FinalSize 一致性校验
    ...
}

func (c *streamFlowController) AddBytesRead(n protocol.ByteCount) (hasStreamWindowUpdate, hasConnWindowUpdate bool) {
    // 应用读走数据后，决定是否回发窗口更新帧
    ...
}
```

### 6.2 Connection 级窗口

代码位置：`internal/flowcontrol/connection_flow_controller.go:41` `IncrementHighestReceived`

```go
// connection_flow_controller.go
func (c *connectionFlowController) IncrementHighestReceived(increment protocol.ByteCount, now monotime.Time) error {
    // 连接总接收预算控制，防止多流叠加耗尽内存
    ...
}
```

---

## 7. 拥塞控制（CUBIC）

代码位置：

- `internal/congestion/cubic_sender.go:143` `OnPacketSent`
- `internal/congestion/cubic_sender.go:183` `OnPacketAcked`
- `internal/congestion/cubic_sender.go:199` `OnCongestionEvent`

```go
// cubic_sender.go
func (c *cubicSender) OnPacketAcked(ackedPacketNumber protocol.PacketNumber, ackedBytes protocol.ByteCount, priorInFlight protocol.ByteCount, eventTime time.Time) {
    // 收到 ACK 后推进拥塞窗口增长逻辑
    // 慢启动和拥塞避免阶段增长策略不同
    ...
}

func (c *cubicSender) OnCongestionEvent(packetNumber protocol.PacketNumber, lostBytes, priorInFlight protocol.ByteCount) {
    // 检测到拥塞时降低发送速率，进入恢复态
    ...
}
```

---

## 8. 重排序：乱序分片如何恢复为有序字节流

代码位置：

- `receive_stream.go:403` `handleStreamFrame`
- `frame_sorter.go:45` `Push`
- `frame_sorter.go:220` `Pop`

```go
// receive_stream.go
func (s *ReceiveStream) handleStreamFrame(frame *wire.StreamFrame, now monotime.Time) error {
    // STREAM frame 到达可能乱序，交给 frameSorter 做重组
    return s.handleStreamFrameImpl(frame, now)
}

// frame_sorter.go
func (s *frameSorter) Push(data []byte, offset protocol.ByteCount, doneCb func()) error {
    // 处理重复片段、重叠片段、乱序片段
    // 维护“缺口区间”，只在连续时才允许上层读取
    ...
}

func (s *frameSorter) Pop() (protocol.ByteCount, []byte, func()) {
    // 弹出当前可连续读取的数据段，保证流内严格有序
    ...
}
```

---

## 9. 超时与定时器调度

代码位置：

- `connection.go:847` `nextIdleTimeoutTime`
- `connection.go:854` `nextKeepAliveTime`
- `connection.go:862` `maybeResetTimer`

```go
// connection.go
func (c *Conn) maybeResetTimer() {
    // 统一比较各类 deadline：
    // 1) idle timeout
    // 2) keepalive
    // 3) ACK alarm
    // 4) loss detection timeout
    // 5) pacing deadline
    ...
    c.timer.Reset(monotime.Until(deadline))
}
```

如果这里理解错，后面会连带误解“为什么会突然重传/超时断连/发送暂停”。

---

## 10. 路径迁移（Path Migration）

### 10.1 服务端被动迁移

代码位置：

- `path_manager.go:66` `HandlePacket`
- `path_manager.go:150` `HandlePathResponseFrame`
- `path_manager.go:162` `SwitchToPath`

```go
// path_manager.go
func (pm *pathManager) HandlePacket(remoteAddr net.Addr, t monotime.Time, pathChallenge *wire.PathChallengeFrame, isNonProbing bool) (_ protocol.ConnectionID, _ []ackhandler.Frame, shouldSwitch bool) {
    // 检测到新地址后，为该路径发送 PATH_CHALLENGE
    // 只有验证通过且收到非 probing 包才允许 shouldSwitch=true
    ...
    frames = append(frames, ackhandler.Frame{
        Frame:   &wire.PathChallengeFrame{Data: p.pathChallenge},
        Handler: (*pathManagerAckHandler)(pm),
    })
    ...
}

func (pm *pathManager) HandlePathResponseFrame(f *wire.PathResponseFrame) {
    // 匹配上 challenge 才标记路径 validated
    ...
}
```

### 10.2 客户端主动迁移

代码位置：

- `path_manager_outgoing.go:38` `(*Path).Probe`
- `path_manager_outgoing.go:234` `NextPathToProbe`
- `path_manager_outgoing.go:294` `ShouldSwitchPath`

```go
// path_manager_outgoing.go
func (p *Path) Probe(ctx context.Context) error {
    // 主动探测新路径，直到 validated 或超时/取消
    ...
    nextProbeDur *= 2 // 指数退避，避免探测风暴
    ...
}

func (pm *pathManagerOutgoing) NextPathToProbe() (_ protocol.ConnectionID, _ ackhandler.Frame, _ *Transport, hasPath bool) {
    // 取出待探测路径并构造 PATH_CHALLENGE 帧
    ...
}
```

---

## 11. 连接管理（生命周期 + CID）

代码位置：

- `server.go:333` `accept`
- `server.go:357` `close`
- `connection.go:2175` `CloseWithError`
- `conn_id_manager.go:64` `Add`
- `conn_id_manager.go:254` `GetConnIDForPath`
- `conn_id_manager.go:279` `RetireConnIDForPath`

```go
// connection.go
func (c *Conn) CloseWithError(code ApplicationErrorCode, desc string) error {
    // 设置应用层关闭错误，触发连接退出流程
    c.closeLocal(&qerr.ApplicationError{ErrorCode: code, ErrorMessage: desc})

    // 阻塞等待 run() 主循环收尾完成
    <-c.ctx.Done()
    return nil
}

// conn_id_manager.go
func (h *connIDManager) RetireConnIDForPath(pathID pathID) {
    // 迁移或路径淘汰后回收 CID，避免无效 CID 长期占用
    ...
}
```

---

## 12. 与底层 UDP 交互：rawConn / oobConn / sendConn

### 12.1 能力抽象与分层

代码位置：

- `sys_conn.go:29` `type rawConn interface`
- `sys_conn.go:55` `wrapConn`

```go
// sys_conn.go
type rawConn interface {
    // 收一个包（包含接收时间、远端地址、ECN/OOB 信息）
    ReadPacket() (receivedPacket, error)

    // 发一个包（可带 GSO/ECN/OOB 控制信息）
    WritePacket(b []byte, addr net.Addr, packetInfoOOB []byte, gsoSize uint16, ecn protocol.ECN) (int, error)
    ...
}

func wrapConn(pc net.PacketConn) (rawConn, error) {
    ...
    if !ok {
        // 不支持 OOB 的 conn 退化为 basicConn
        return &basicConn{PacketConn: pc, supportsDF: supportsDF}, nil
    }

    // 支持 OOB 走高性能路径（ReadBatch/ECN/GSO）
    return newConn(c, supportsDF)
}
```

### 12.2 批量读与高性能发送

代码位置：

- `sys_conn_oob.go:162` `ReadPacket`
- `sys_conn_oob.go:174` `ReadBatch`
- `sys_conn_oob.go:247` `WritePacket`

```go
// sys_conn_oob.go
func (c *oobConn) ReadPacket() (receivedPacket, error) {
    // 底层使用 ReadBatch 一次拉多个包，降低 syscall 频率
    n, err := c.batchConn.ReadBatch(c.messages, 0)
    ...
}

func (c *oobConn) WritePacket(b []byte, addr net.Addr, packetInfoOOB []byte, gsoSize uint16, ecn protocol.ECN) (int, error) {
    // 发送时可携带 ECN 标记和 UDP_SEGMENT（GSO）
    ...
}
```

### 12.3 GSO 自动降级

代码位置：`send_conn.go:72` `(*sconn).Write`

```go
// send_conn.go
func (c *sconn) Write(p []byte, gsoSize uint16, ecn protocol.ECN) error {
    ai := c.remoteAddrInfo.Load()
    err := c.writePacket(p, ai.addr, ai.oob, gsoSize, ecn)

    if err != nil && isGSOError(err) {
        // GSO 失败后关闭 GSO 能力，改为逐段发送
        c.gotGSOError = true

        for len(p) > 0 {
            l := len(p)
            if l > int(gsoSize) {
                l = int(gsoSize)
            }
            if err := c.writePacket(p[:l], ai.addr, ai.oob, 0, ecn); err != nil {
                return err
            }
            p = p[l:]
        }
        return nil
    }

    return err
}
```

---

## 13. UDP 数据如何组装为 HTTP 请求

代码位置：

- `streams_map.go:309` `HandleStreamFrame`
- `http3/server.go:470` `handleConn`
- `http3/server_conn.go:99` `handleRequestStream`
- `http3/frames.go:54` `ParseNext`
- `http3/headers.go:184` `requestFromHeaders`

关键代码（含注释）：

```go
// http3/server.go
func (s *Server) handleConn(conn *quic.Conn) error {
    ...
    for {
        // 接收对端新开的双向请求流
        str, err := conn.AcceptStream(ctx)
        if err != nil {
            ...
        }

        // 每个请求流独立并发处理
        go func() {
            hconn.HandleRequestStream(str)
        }()
    }
}
```

```go
// http3/server_conn.go
func (c *RawServerConn) handleRequestStream(str *stateTrackingStream) {
    fp := &frameParser{closeConn: conn.CloseWithError, r: str, streamID: str.StreamID()}

    // 读取请求流第一帧
    frame, err := fp.ParseNext(qlogger)
    if err != nil {
        ...
        return
    }

    // RFC 要求请求流第一帧必须是 HEADERS
    hf, ok := frame.(*headersFrame)
    if !ok {
        conn.CloseWithError(quic.ApplicationErrorCode(ErrCodeFrameUnexpected), "expected first frame to be a HEADERS frame")
        return
    }

    // 读取 HEADERS payload（QPACK 编码块）
    headerBlock := make([]byte, hf.Length)
    if _, err := io.ReadFull(str, headerBlock); err != nil {
        ...
        return
    }

    // QPACK 解码并转换成标准库 Request
    decodeFn := decoder.Decode(headerBlock)
    req, err := requestFromHeaders(decodeFn, maxHeaderBytes, &hfs)
    if err != nil {
        ...
        return
    }

    ...
}
```

```go
// http3/headers.go
func requestFromHeaders(decodeFn qpack.DecodeFunc, sizeLimit int, headerFields *[]qpack.HeaderField) (*http.Request, error) {
    hdr, err := parseHeaders(decodeFn, true, sizeLimit, headerFields)
    if err != nil {
        return nil, err
    }

    // 这里会校验 :method/:path/:authority/:scheme 等伪头
    // 并且根据 CONNECT/Extended CONNECT 规则构造 URL 与 RequestURI
    ...

    req := &http.Request{
        Method:        hdr.Method,
        URL:           u,
        Proto:         "HTTP/3.0",
        ProtoMajor:    3,
        ProtoMinor:    0,
        Header:        hdr.Headers,
        ContentLength: hdr.ContentLength,
        Host:          hdr.Authority,
        RequestURI:    requestURI,
    }
    return req, nil
}
```

---

## 14. HTTP 请求处理与响应回写

### 14.1 进入业务 handler

代码位置：`http3/server_conn.go:222`

```go
// http3/server_conn.go
handler := c.requestHandler
if handler == nil {
    handler = http.DefaultServeMux
}

// 最终和 net/http 一样，调用 ServeHTTP
handler.ServeHTTP(r, req)
```

### 14.2 响应编码为 HEADERS + DATA

代码位置：

- `http3/response_writer.go:81` `WriteHeader`
- `http3/response_writer.go:209` `writeHeader`
- `http3/response_writer.go:166` `doWrite`
- `http3/stream.go:119` `Stream.Write`

```go
// http3/response_writer.go
func (w *responseWriter) writeHeader(status int) error {
    var headers bytes.Buffer
    enc := qpack.NewEncoder(&headers)

    // 先写 HTTP 状态伪头
    if err := enc.WriteField(qpack.HeaderField{Name: ":status", Value: strconv.Itoa(status)}); err != nil {
        return err
    }

    // 把普通 Header 编码进 QPACK
    for k, v := range w.header {
        ...
    }

    // 组装 HTTP/3 HEADERS frame = frame header + qpack block
    buf := make([]byte, 0, frameHeaderLen+headers.Len())
    buf = (&headersFrame{Length: uint64(headers.Len())}).Append(buf)
    buf = append(buf, headers.Bytes()...)

    // 裸写到 QUIC stream
    _, err := w.str.writeUnframed(buf)
    return err
}

func (w *responseWriter) doWrite(p []byte) (int, error) {
    if !w.headerWritten {
        if err := w.writeHeader(w.status); err != nil {
            return 0, err
        }
        w.headerWritten = true
    }

    // 每次 Write 都会先写 DATA frame 头，再写 payload
    df := &dataFrame{Length: uint64(len(p))}
    w.buf = w.buf[:0]
    w.buf = df.Append(w.buf)

    if _, err := w.str.writeUnframed(w.buf); err != nil {
        return 0, err
    }
    return w.str.writeUnframed(p)
}
```

---

## 15. 请求体、Trailer、Datagram

代码位置：

- `http3/stream.go:73` `(*Stream).Read`
- `http3/body.go:59` `(*body).Read`
- `http3/conn.go:252` `sendDatagram`
- `http3/conn.go:270` `receiveDatagrams`

```go
// http3/stream.go
func (s *Stream) Read(b []byte) (int, error) {
    if s.bytesRemainingInFrame == 0 {
        frame, err := s.frameParser.ParseNext(s.qlogger)
        if err != nil {
            return 0, err
        }

        switch f := frame.(type) {
        case *dataFrame:
            // 当前读取窗口切换到 DATA frame payload
            s.bytesRemainingInFrame = f.Length
        case *headersFrame:
            // 请求尾部 HEADERS 按 trailer 处理
            return 0, s.parseTrailer(s.datagramStream, f)
        default:
            // 在请求体阶段收到不该出现的帧，按协议错误处理
            s.conn.CloseWithError(quic.ApplicationErrorCode(ErrCodeFrameUnexpected), "")
            return 0, fmt.Errorf("peer sent an unexpected frame: %T", f)
        }
    }
    ...
}

// http3/body.go
func (r *body) Read(b []byte) (int, error) {
    // 严格执行 Content-Length，防止消息体长度作弊
    if err := r.checkContentLengthViolation(); err != nil {
        return 0, err
    }
    ...
}
```

---

## 16. 包解密补充：packet unpacker

代码位置：

- `packet_unpacker.go:55` `UnpackLongHeader`
- `packet_unpacker.go:109` `UnpackShortHeader`

```go
// packet_unpacker.go
func (u *packetUnpacker) UnpackLongHeader(hdr *wire.Header, data []byte) (*unpackedPacket, error) {
    // 根据包类型选择对应 opener（Initial/Handshake/0-RTT）
    ...

    // 先解 header，再解 payload，并恢复 packet number
    extHdr, decrypted, err := u.unpackLongHeaderPacket(opener, hdr, data)
    if err != nil {
        return nil, err
    }

    // 返回解密后的纯 QUIC frame payload
    return &unpackedPacket{hdr: extHdr, encryptionLevel: encLevel, data: decrypted}, nil
}
```

---

## 17. 如何用这份笔记学习（实操建议）

1. 先读第 `2/3/13/14` 章，跑通请求主链路。
2. 再读第 `5/6/7/8/9` 章，吃透可靠性与性能控制。
3. 最后读第 `10/12/16` 章，理解路径迁移与 UDP 底层能力。

建议验证命令：

```bash
curl --http3 -k -v https://localhost:8443/ping
```

建议观测：

- qlog：`PacketReceived / PacketLost / FrameParsed / ConnectionClosed`
- 日志：迁移、超时、GSO 降级、重传触发点

---

## 18. 总结

quic-go 的核心不是“UDP 上跑 HTTP”这么简单，而是把 **收包、解密、帧解析、可靠传输、流控拥塞、迁移和 HTTP/3 语义** 融为一个跨层状态机。你真正要会的是：看到现象，能定位到具体函数和状态转移。
