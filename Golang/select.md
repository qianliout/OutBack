# Go select 语句与源码解析
## 1. `select` 语句：Goroutine 的多路复用选择器

`select` 是 Go 语言独有的控制结构，它像一个专为 Channel 通信设计的 `switch` 语句，允许一个 Goroutine 同时等待多个通信操作。

### 1.1 基本语法与核心机制

```go
select {
case <-ch1:
    // ch1 可读
case data := <-ch2:
    // ch2 可读，data 是读取到的值
case ch3 <- "hello":
    // ch3 可写
default:
    // 所有 case 的 channel 都未就绪时，立即执行此分支
}
```

**核心工作机制：**

1.  **就绪评估**：`select` 会评估所有 `case` 中的 Channel 操作，找出“就绪”（可立即执行而无需阻塞）的 `case`。
2.  **随机选择**：
    *   如果只有一个 `case` 就绪，执行它。
    *   如果有多个 `case` 同时就绪，`select` 会**伪随机地选择一个**执行。这种随机性保证了公平，避免了饥饿问题。
    *   如果所有 `case` 都未就绪，`select` 会阻塞，直到其中一个 `case` 变得就绪。
3.  **`default` 分支**：`default` 分支的存在使 `select` 变为**非阻塞**的。如果没有任何 `case` 就绪，`select` 会立即执行 `default` 分支。
4.  **`nil` Channel**：对 `nil` Channel 的操作（读或写）将**永远不会被选中**，该 `case` 会被永久忽略。

### 1.2 常见并发模式

#### 1.2.1 超时控制

通过结合 `time.After`，可以轻松实现对一个操作的超时控制，这是 `select` 最经典的应用。

```go
func doSomething() string {
    time.Sleep(2 * time.Second)
    return "done"
}

func main() {
    resultChan := make(chan string, 1)
    go func() { resultChan <- doSomething() }()

    select {
    case res := <-resultChan:
        fmt.Println("Operation finished:", res)
    case <-time.After(1 * time.Second):
        fmt.Println("Timeout!") // 先发生
    }
}
// 输出: Timeout!
```

#### 1.2.2 退出信号

通过一个 `done` Channel，可以优雅地通知一个正在后台运行的 Goroutine 停止工作。

```go
func worker(done <-chan struct{}) {
    for {
        select {
        case <-done:
            fmt.Println("Worker: received stop signal, exiting.")
            return
        default:
            fmt.Println("Worker: working...")
            time.Sleep(500 * time.Millisecond)
        }
    }
}
```
这种模式虽然有效，但当 Goroutine 调用链很长时，逐层传递 `done` Channel 会变得非常繁琐。这正是 `context` 要解决的核心痛点之一。

#### 1.2.3 非阻塞的 Channel 操作

-   **非阻塞读**:
    ```go
    select {
    case data := <-ch:
        fmt.Println("Read data:", data)
    default:
        fmt.Println("Channel is empty, not blocking.")
    }
    ```

-   **非阻塞写**:
    ```go
    select {
    case ch <- "data":
        fmt.Println("Sent data successfully.")
    default:
        fmt.Println("Channel is full, not blocking.")
    }
    ```

### 1.3 `for-select` 循环陷阱

在 `for-select` 循环中直接使用 `time.After` 会导致**内存泄漏**，因为每次循环都会创建一个新的 `Timer` 对象。

```go
// 错误示例: 每次循环都创建新 Timer
for {
    select {
    case <-ch:
        // ...
    case <-time.After(1 * time.Second): // 泄漏的 Timer
        fmt.Println("Timed out")
        return
    }
}

// 正确示例: 复用 Timer
timer := time.NewTimer(1 * time.Second)
defer timer.Stop() // 确保资源被释放
for {
    select {
    case <-ch:
        // ...
        // 重置计时器
        if !timer.Stop() {
            <-timer.C
        }
        timer.Reset(1 * time.Second)
    case <-timer.C:
        fmt.Println("Timed out")
        return
    }
}
```

## 2. `select` 源码解读 (Go 1.25+)

当然。Go 中的 `select` 语句是并发模型的关键特性，其实现位于运行时（runtime）中，核心函数是 `selectgo`。下面我会从编译期到运行期逐步拆解，并辅以源码（基于 Go 1.25+ 的简化版）解释。

### 2.1 编译期转换

当你写下：

```go
select {
case ch1 <- v1:
    ...
case v2 := <-ch2:
    ...
default:
    ...
}
```

编译器（`cmd/compile/internal/walk/select.go`）会将其转换为：

1. 创建 `[]scase` 数组。**在 Go 1.25+ 中，编译器会对 case 进行重新排序：所有的 send case 放在数组前段，recv case 放在后段。**
2. 调用 `runtime.selectgo`，该函数返回被选中的 case 索引（根据源码中原始的 case 顺序对应）。
3. 根据索引跳转到对应的代码块。

`scase` 定义在 `runtime/select.go` 中，在近期的 Go 版本中已被大幅精简：

```go
type scase struct {
    c    *hchan         // 对应 channel
    elem unsafe.Pointer // 接收或发送的数据元素地址
}
```

注意：旧版本中的 `kind`（类型）、`pc` 等字段已被移除。现在 case 的类型（发送/接收）由它在数组中的**索引位置**决定（是否小于 `nsends`）。而 `default` case 不再作为一种独立的 scase 存在，而是转换为一个 `block` 的布尔值标志传递。

### 2.2 `selectgo` 的核心流程

函数签名：

```go
func selectgo(cas0 *scase, order0 *uint16, pc0 *uintptr, nsends, nrecvs int, block bool) (int, bool)
```

参数：
- `cas0`：`[]scase` 的首地址。
- `order0`：两倍于 `ncases` 长度的 `uint16` 数组，前半段用于随机轮询顺序，后半段用于加锁顺序。
- `nsends`, `nrecvs`：发送 case 和接收 case 的数量。
- `block`：表示如果没有就绪的 channel 时是否需要阻塞。如果有 `default` case，这里会传入 `false`。
- 返回值：选中的 case 索引；如果是从 channel 接收，第二个返回值表示是否成功接收（对应 `v, ok <-ch` 的 `ok`）。

整体流程分为 **三个阶段**：

#### 2.2.1 阶段1：生成轮询顺序 & 加锁

为了保证公平性，`selectgo` 会 **随机** 遍历所有 case。通过两个辅助切片 `pollorder` 和 `lockorder` 实现：

- `pollorder`：随机遍历顺序，用于第一轮检查就绪状态
- `lockorder`：按 channel 地址排序的加锁顺序，用于避免死锁

**轮询顺序生成源码** (`selectgo` 内部)：

```go
// generate permuted order
norder := 0
allSynctest := true
for i := range scases {
    cas := &scases[i]

    // Omit cases without channels from the poll and lock orders.
    if cas.c == nil {
        cas.elem = nil // allow GC
        continue
    }

    if cas.c.bubble != nil {
        if getg().bubble != cas.c.bubble {
            fatal("select on synctest channel from outside bubble")
        }
    } else {
        allSynctest = false
    }

    if cas.c.timer != nil {
        cas.c.timer.maybeRunChan(cas.c)
    }

    j := cheaprandn(uint32(norder + 1))
    pollorder[norder] = pollorder[j]
    pollorder[j] = uint16(i)
    norder++
}
pollorder = pollorder[:norder]
lockorder = lockorder[:norder]
```

**加锁顺序排序** (heap sort 按 channel 地址)：

```go
// sort the cases by Hchan address to get the locking order.
// simple heap sort, to guarantee n log n time and constant stack footprint.
for i := range lockorder {
    j := i
    // Start with the pollorder to permute cases on the same channel.
    c := scases[pollorder[i]].c
    for j > 0 && scases[lockorder[(j-1)/2]].c.sortkey() < c.sortkey() {
        k := (j - 1) / 2
        lockorder[j] = lockorder[k]
        j = k
    }
    lockorder[j] = pollorder[i]
}
// ... 下滤过程 ...
```

**sellock 源码**（按 lockorder 顺序加锁，避免死锁）：

```go
func sellock(scases []scase, lockorder []uint16) {
    var c *hchan
    for _, o := range lockorder {
        c0 := scases[o].c
        if c0 != c {
            c = c0
            lock(&c.lock)
        }
    }
}
```

#### 2.2.2 阶段2：第一轮轮询（无阻塞）

对所有 case 按 `pollorder` 顺序检查是否有就绪的 channel：

```go
// pass 1 - look for something already waiting
for _, casei := range pollorder {
    casi = int(casei)
    cas = &scases[casi]
    c = cas.c

    if casi >= nsends { // 接收 case
        sg = c.sendq.dequeue()
        if sg != nil {
            goto recv
        }
        if c.qcount > 0 {
            goto bufrecv
        }
        if c.closed != 0 {
            goto rclose
        }
    } else { // 发送 case
        if c.closed != 0 {
            goto sclose
        }
        sg = c.recvq.dequeue()
        if sg != nil {
            goto send
        }
        if c.qcount < c.dataqsiz {
            goto bufsend
        }
    }
}
```

如果第一轮没有发现任何就绪的 case：
- 如果 `block` 为 `false`（有 `default` case） → `selunlock` 后直接返回 `casi = -1`
- 否则（`block` 为 `true`），进入 **阻塞等待**

#### 2.2.3 阶段3：阻塞 & 等待唤醒

这是最复杂的部分。将当前 goroutine 包装成 `sudog`，挂载到所有 case 对应的 channel 等待队列上：

**pass 2 - enqueue on all chans**：

```go
// pass 2 - enqueue on all chans
if gp.waiting != nil {
    throw("gp.waiting != nil")
}
nextp = &gp.waiting
for _, casei := range lockorder {
    casi = int(casei)
    cas = &scases[casi]
    c = cas.c
    sg := acquireSudog()
    sg.g = gp
    sg.isSelect = true
    sg.elem = cas.elem
    sg.releasetime = 0
    if t0 != 0 {
        sg.releasetime = -1
    }
    sg.c = c
    // Construct waiting list in lock order.
    *nextp = sg
    nextp = &sg.waitlink

    if casi < nsends {
        c.sendq.enqueue(sg) // 发送挂到 sendq
    } else {
        c.recvq.enqueue(sg) // 接收挂到 recvq
    }

    if c.timer != nil {
        blockTimerChan(c)
    }
}

// wait for someone to wake us up
gp.param = nil
gp.parkingOnChan.Store(true)
gopark(selparkcommit, nil, waitReason, traceBlockSelect, 1)
gp.activeStackChans = false
```

**唤醒后的 pass 3 处理**（清理未成功的等待队列）：

```go
sellock(scases, lockorder)

gp.selectDone.Store(0)
sg = (*sudog)(gp.param)
gp.param = nil

// pass 3 - dequeue from unsuccessful chans
// otherwise they stack up on quiet channels
casi = -1
cas = nil
caseSuccess = false
sglist = gp.waiting
// Clear all elem before unlinking from gp.waiting.
for sg1 := gp.waiting; sg1 != nil; sg1 = sg1.waitlink {
    sg1.isSelect = false
    sg1.elem = nil
    sg1.c = nil
}
gp.waiting = nil

for _, casei := range lockorder {
    k = &scases[casei]
    if k.c.timer != nil {
        unblockTimerChan(k.c)
    }
    if sg == sglist {
        // sg has already been dequeued by the G that woke us up.
        casi = int(casei)
        cas = k
        caseSuccess = sglist.success
        if sglist.releasetime > 0 {
            caseReleaseTime = sglist.releasetime
        }
    } else {
        c = k.c
        if int(casei) < nsends {
            c.sendq.dequeueSudoG(sglist)
        } else {
            c.recvq.dequeueSudoG(sglist)
        }
    }
    sgnext = sglist.waitlink
    sglist.waitlink = nil
    releaseSudog(sglist)
    sglist = sgnext
}
```

### 2.3 关键细节

#### 2.3.1 公平性

`select` 每次执行时会通过 `cheaprandn` **随机化** 轮询顺序，防止某个 case 被饿死。

#### 2.3.2 空 select

```go
select {}
```

会编译为对 `runtime.block` 的调用，该函数执行 `gopark(nil, nil, waitReasonSelectNoCases, ...)` 让 goroutine 永远挂起。

#### 2.3.3 对 nil channel 的处理

`selectgo` 遍历初始化 `pollorder` 时，如果遇到 `cas.c == nil` 的 channel 会直接跳过。这意味着 nil channel 既不会被轮询，也不会被加锁阻塞。如果所有 channel 都是 nil：
- 有 default (`block == false`)：直接执行 default。
- 无 default (`block == true`)：当前 goroutine 会被永久挂起。

#### 2.3.4 内存优化

`selectgo` 的 `order0` 数组由编译器在栈上分配，避免频繁堆分配。`scase` 数组也是栈上分配。由于移除了 `kind` 等字段，现在的 `scase` 更加轻量，仅占两个指针大小。

### 2.4 `selectgo` 完整源码（Go 1.25+）

以下是 `/usr/local/go/src/runtime/select.go` 中 `selectgo` 函数的完整源码，逐段分析：

**函数签名与变量初始化：**

```go
func selectgo(cas0 *scase, order0 *uint16, pc0 *uintptr, nsends, nrecvs int, block bool) (int, bool) {
    gp := getg()
    if debugSelect {
        print("select: cas0=", cas0, "\n")
    }

    cas1 := (*[1 << 16]scase)(unsafe.Pointer(cas0))
    order1 := (*[1 << 17]uint16)(unsafe.Pointer(order0))

    ncases := nsends + nrecvs
    scases := cas1[:ncases:ncases]
    pollorder := order1[:ncases:ncases]
    lockorder := order1[ncases:][:ncases:ncases]
    // NOTE: pollorder/lockorder's underlying array was not zero-initialized by compiler.

    var pcs []uintptr
    if raceenabled && pc0 != nil {
        pc1 := (*[1 << 16]uintptr)(unsafe.Pointer(pc0))
        pcs = pc1[:ncases:ncases]
    }
    casePC := func(casi int) uintptr {
        if pcs == nil {
            return 0
        }
        return pcs[casi]
    }

    var t0 int64
    if blockprofilerate > 0 {
        t0 = cputicks()
    }
    // ... 轮询顺序生成、heap sort、sellock ...

    var (
        sg     *sudog
        c      *hchan
        k      *scase
        sglist *sudog
        sgnext *sudog
        qp     unsafe.Pointer
        nextp  **sudog
    )
```

**bufrecv（从缓冲区接收）标签：**

```go
bufrecv:
    // can receive from buffer
    if raceenabled {
        if cas.elem != nil {
            raceWriteObjectPC(c.elemtype, cas.elem, casePC(casi), chanrecvpc)
        }
        racenotify(c, c.recvx, nil)
    }
    if msanenabled && cas.elem != nil {
        msanwrite(cas.elem, c.elemtype.Size_)
    }
    if asanenabled && cas.elem != nil {
        asanwrite(cas.elem, c.elemtype.Size_)
    }
    recvOK = true
    qp = chanbuf(c, c.recvx)
    if cas.elem != nil {
        typedmemmove(c.elemtype, cas.elem, qp)
    }
    typedmemclr(c.elemtype, qp)
    c.recvx++
    if c.recvx == c.dataqsiz {
        c.recvx = 0
    }
    c.qcount--
    selunlock(scases, lockorder)
    goto retc
```

**bufsend（发送到缓冲区）标签：**

```go
bufsend:
    // can send to buffer
    if raceenabled {
        racenotify(c, c.sendx, nil)
        raceReadObjectPC(c.elemtype, cas.elem, casePC(casi), chansendpc)
    }
    if msanenabled {
        msanread(cas.elem, c.elemtype.Size_)
    }
    if asanenabled {
        asanread(cas.elem, c.elemtype.Size_)
    }
    typedmemmove(c.elemtype, chanbuf(c, c.sendx), cas.elem)
    c.sendx++
    if c.sendx == c.dataqsiz {
        c.sendx = 0
    }
    c.qcount++
    selunlock(scases, lockorder)
    goto retc
```

**recv（从 sleeping sender 接收）标签：**

```go
recv:
    // can receive from sleeping sender (sg)
    recv(c, sg, cas.elem, func() { selunlock(scases, lockorder) }, 2)
    if debugSelect {
        print("syncrecv: cas0=", cas0, " c=", c, "\n")
    }
    recvOK = true
    goto retc
```

**rclose（从已关闭 channel 接收）标签：**

```go
rclose:
    // read at end of closed channel
    selunlock(scases, lockorder)
    recvOK = false
    if cas.elem != nil {
        typedmemclr(c.elemtype, cas.elem)
    }
    if raceenabled {
        raceacquire(c.raceaddr())
    }
    goto retc
```

**send（发送给 sleeping receiver）标签：**

```go
send:
    // can send to a sleeping receiver (sg)
    if raceenabled {
        raceReadObjectPC(c.elemtype, cas.elem, casePC(casi), chansendpc)
    }
    if msanenabled {
        msanread(cas.elem, c.elemtype.Size_)
    }
    if asanenabled {
        asanread(cas.elem, c.elemtype.Size_)
    }
    send(c, sg, cas.elem, func() { selunlock(scases, lockorder) }, 2)
    if debugSelect {
        print("syncsend: cas0=", cas0, " c=", c, "\n")
    }
    goto retc
```

**retc（返回）标签：**

```go
retc:
    if caseReleaseTime > 0 {
        blockevent(caseReleaseTime-t0, 1)
    }
    return casi, recvOK

sclose:
    // send on closed channel
    selunlock(scases, lockorder)
    panic(plainError("send on closed channel"))
}
```

**selunlock 源码（倒序解锁）：**

```go
func selunlock(scases []scase, lockorder []uint16) {
    // We must be very careful here to not touch sel after we have unlocked
    // the last lock, because sel can be freed right after the last unlock.
    for i := len(lockorder) - 1; i >= 0; i-- {
        c := scases[lockorder[i]].c
        if i > 0 && c == scases[lockorder[i-1]].c {
            continue // will unlock it on the next iteration
        }
        unlock(&c.lock)
    }
}
```

### 2.5 总结

- `select` 的核心是 `selectgo`，在 Go 1.25+ 中，编译器通过物理排序（先 send 后 recv）移除了 `scase.kind`，使结构更轻量。
- `default` 不再作为单独的 case 处理，而是化简为了 `block` 布尔标志。
- 没有就绪且无 default 时，当前 goroutine 会被挂载到所有相关的 channel 等待队列中，直到某个 case 就绪被唤醒。
- 这种实现使得 `select` 能够高效处理多路 I/O 复用，是 CSP 模型在 Go 中的重要支撑。

如果你希望我展开某个特定部分（比如 `sudog` 的数据结构、`gopark` 的调度机制、唤醒后的精确查找算法），我也可以深入源码逐行讲解。