src/runtime/proc.go:449


```go
// Puts the current goroutine into a waiting state and calls unlockf on the
// system stack.
//
// If unlockf returns false, the goroutine is resumed.
//
// unlockf must not access this G's stack, as it may be moved between
// the call to gopark and the call to unlockf.
//
// Note that because unlockf is called after putting the G into a waiting
// state, the G may have already been readied by the time unlockf is called
// unless there is external synchronization preventing the G from being
// readied. If unlockf returns false, it must guarantee that the G cannot be
// externally readied.
//
// Reason explains why the goroutine has been parked. It is displayed in stack
// traces and heap dumps. Reasons should be unique and descriptive. Do not
// re-use reasons, add new ones.
//
// gopark should be an internal detail,
// but widely used packages access it using linkname.
// Notable members of the hall of shame include:
//   - gvisor.dev/gvisor
//   - github.com/sagernet/gvisor
//
// Do not remove or change the type signature.
// See go.dev/issue/67401.
//
//go:linkname gopark
func gopark(unlockf func(*g, unsafe.Pointer) bool, lock unsafe.Pointer, reason waitReason, traceReason traceBlockReason, traceskip int) {
	if reason != waitReasonSleep {
		checkTimeouts() // timeouts may expire while two goroutines keep the scheduler busy
	}
	mp := acquirem()
	gp := mp.curg
	status := readgstatus(gp)
	if status != _Grunning && status != _Gscanrunning {
		throw("gopark: bad g status")
	}
	mp.waitlock = lock
	mp.waitunlockf = unlockf
	gp.waitreason = reason
	mp.waitTraceBlockReason = traceReason
	mp.waitTraceSkip = traceskip
	releasem(mp)
	// can't do anything that might move the G between Ms here.
	mcall(park_m)
}
```

# gopark 函数详解

`gopark` 是 Go 运行时中 **goroutine 调度**的核心函数之一，负责将当前 goroutine 从运行状态转换为等待状态。

## 函数签名

```go
func gopark(unlockf func(*g, unsafe.Pointer) bool, 
            lock unsafe.Pointer, 
            reason waitReason, 
            traceReason traceBlockReason, 
            traceskip int)
```

## 参数解析

| 参数            | 类型                              | 含义                       |
| ------------- | ------------------------------- | ------------------------ |
| `unlockf`     | `func(*g, unsafe.Pointer) bool` | 解锁函数，唤醒时调用，返回 false 则不唤醒 |
| `lock`        | `unsafe.Pointer`                | 关联的锁对象指针                 |
| `reason`      | `waitReason`                    | 等待原因，用于调试和追踪             |
| `traceReason` | `traceBlockReason`              | 追踪块原因                    |
| `traceskip`   | `int`                           | 堆栈追踪时跳过的帧数               |

---

## 逐段解析

### 1. 超时检查

```go
if reason != waitReasonSleep {
    checkTimeouts() // timeouts may expire while two goroutines keep the scheduler busy
}
```

- 如果不是睡眠原因，调用 `checkTimeouts()` 检查是否有超时到期
- 这确保即使有两个 goroutine 一直占用调度器，超时机制也能正常工作

### 2. 获取当前 goroutine 和调度信息

```go
mp := acquirem()
gp := mp.curg
status := readgstatus(gp)
```

- `acquirem()`: 获取当前的 M (Machine) 和 G (Goroutine) 关联
- `mp.curg`: 指向当前运行的 goroutine
- `readgstatus(gp)`: 读取 goroutine 状态

### 3. 状态校验

```go
if status != _Grunning && status != _Gscanrunning {
    throw("gopark: bad g status")
}
```

goroutine 必须是以下状态才能 park：
- `_Grunning`: 正在运行
- `_Gscanrunning`: GC 扫描中运行

### 4. 保存等待状态到 M

```go
mp.waitlock = lock
mp.waitunlockf = unlockf
gp.waitreason = reason
mp.waitTraceBlockReason = traceReason
mp.waitTraceSkip = traceskip
releasem(mp)
```

将所有等待相关的信息保存到当前的 M 结构体中：
- `waitlock`: 关联的锁
- `waitunlockf`: 解锁函数（唤醒时调用）
- `waitreason`: 等待原因（会在 stack traces 和 heap dumps 中显示）

### 5. 调用 mcall 切换到调度器

```go
// can't do anything that might move the G between Ms here.
mcall(park_m)
```

**关键点**：
- `mcall` 是从 Go 代码切换到 C 代码（调度器）的汇编调用
- `park_m` 是调度器中的 C 函数，负责将 G 从 M 上剥离
- 调用 `mcall` 后，当前 goroutine 会被挂起，调度器选择下一个 goroutine 运行

---

## 核心概念图

```
┌─────────────────────────────────────────────────────────────┐
│                      User Code                              │
│                  (goroutine running)                        │
└─────────────────────┬───────────────────────────────────────┘
                      │
                      │ gopark()
                      ▼
┌─────────────────────────────────────────────────────────────┐
│  1. acquirem() 获取 M                                        │
│  2. 保存 unlockf, lock, reason 到 M                          │
│  3. releasem(mp)                                             │
│  4. mcall(park_m) ──────► 切换到调度器 C 代码                  │
└─────────────────────────────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────┐
│                    park_m (C 函数)                          │
│  • 将 G 从 M 的当前运行位置剥离                                │
│  • 将 G 状态设置为 _Gwaiting                                 │
│  • 调用 schedule() 选择下一个 G 运行                          │
└─────────────────────────────────────────────────────────────┘
```

---

## 唤醒流程

当 goroutine 被唤醒时（如 I/O 就绪、锁释放）：

```
┌─────────────────────────────────────────────────────────────┐
│                  外部事件触发 ready                           │
│            (netpoll,锁释放,channel操作等)                     │
└─────────────────────┬───────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────┐
│  ready(gp) 或 goready(gp)                                    │
│  • 将 G 状态改为 _Grunnable                                  │
│  • 放入运行队列                                               │
└─────────────────────────────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────┐
│  调度器选择此 G 后，调用 execute()                            │
│  • 调用 unlockf() 确认可以唤醒                               │
│  • 如果 unlockf 返回 false，G 保持等待状态                    │
│  • 如果返回 true，G 继续执行 gopark 后的代码                  │
└─────────────────────────────────────────────────────────────┘
```

---

## waitReason 枚举（部分）

| 值 | 含义 |
|------|------|
| `waitReasonIOWait` | 等待 I/O |
| `waitReasonChanReceive` | 等待 channel 接收 |
| `waitReasonChanSend` | 等待 channel 发送 |
| `waitReasonSemacquire` | 信号量等待 |
| `waitReasonSleep` | 睡眠 |
| `waitReasonFinalizer` | Finalizer 等待 |

---

## 设计亮点

1. **不可移动约束**: 注释明确指出 `gopark` 调用后不能做任何可能移动 G 的操作，因为 G 可能正在被 GC 移动
2. **与 GC 的协作**: `waitTraceSkip` 参数用于在 GC 时正确追踪堆栈，跳过 gopark 相关的内部帧
3. **外部同步要求**: 文档指出 `unlockf` 必须保证在没有外部同步的情况下 G 不会被 ready，否则 `unlockf` 返回 false
4. **linkname 机制**: 注释表明此函数被外部包（如 gvisor）通过 `linkname` 直接访问，这是 Go 运行时的特殊豁免

---

## 唤醒与重新执行：从 waiting 到 runnable

`gopark` 本身**只负责把 goroutine 挂起**（交出 CPU 控制权并进入 `_Gwaiting` 状态）。它并不知道自己什么时候会被唤醒，也没有“自动恢复”的定时器。

goroutine 的重新执行，必须依赖**外部事件**调用 `goready`（或内部的 `ready` 函数）来完成状态转换和重新调度。

### 1. 谁在什么时机触发唤醒？

根据导致 `gopark` 的 `waitReason` 不同，唤醒的触发方也不同：

- **网络 I/O (`waitReasonIOWait`)**：
  - 当底层 socket 真正可读/可写时，内核 epoll 触发事件。
  - Go 运行时的调度系统（GMP 模型）会负责收集这些就绪的 fd。具体有两个途径：一是常规 M 在执行调度循环（`schedule` / `findrunnable`）时主动调用 `netpoll` 检查；二是后台监控线程 `sysmon` 会定期调用 `netpoll` 兜底，防止网络事件长时间得不到处理。
  - `netpoll` 查找到 fd 绑定的 goroutine，并将其批量放入运行队列（内部调用 `goready`）。
- **定时器 (`waitReasonSleep` 等)**：
  - timer 到期时，调度器会在 `checkTimers` 阶段找到超时的 timer，取出绑定的 goroutine 并调用 `goready`。
- **Channel 阻塞 (`waitReasonChanReceive/Send`)**：
  - 比如你因读空 channel 而 park，当另一个 goroutine 往该 channel 写入数据时，发送方会直接取出等待队列里的接收方 goroutine，调用 `goready` 唤醒它。
- **锁等待 (`waitReasonSemacquire` 等)**：
  - 比如 `Mutex.Lock()` 阻塞，当持有锁的 goroutine 调用 `Unlock()` 时，会唤醒等待队列队头的 goroutine。

### 2. 唤醒的源码路径：goready 与 ready

当外部事件发生时，会调用 `runtime.goready`，其核心逻辑在 `ready` 函数中：

```go
// src/runtime/proc.go
func ready(gp *g, traceskip int, next bool) {
	status := readgstatus(gp)
	// 1. 必须是从 waiting 状态唤醒
	if status&^_Gscan != _Gwaiting {
		throw("bad g->status in ready")
	}

	// 2. 将状态从 _Gwaiting 改为 _Grunnable
	casgstatus(gp, _Gwaiting, _Grunnable)

	// 3. 将 goroutine 放入运行队列 (本地 P 队列或全局队列)
	runqput(mp.p.ptr(), gp, next)

	// 4. 尝试唤醒一个空闲的 P/M 来执行它
	wakep()
}
```

**关键误区**：`ready` 执行完后，goroutine **并没有立刻上 CPU 执行**。它只是从“不可运行”变成了“排队等待运行”（`_Grunnable`）。

### 3. 真正重新执行：调度器接管

当 goroutine 被放入 `runq`（运行队列）后，它的命运就交给了 Go 的调度器：

1. 某个工作线程（M）绑定的处理器（P）正在执行 `schedule()` 循环。
2. `schedule()` 从本地队列或全局队列中弹出了这个 `_Grunnable` 的 goroutine。
3. 调用 `execute(gp, ...)` 函数。
4. `execute` 会恢复该 goroutine 的上下文（PC 指令寄存器、SP 栈指针等）。
5. **goroutine 从当初调用 `gopark` 的下一条指令处继续向下执行**。

### 4. 完整生命周期闭环

```text
[业务代码]
   │
   ▼
1. 遇到阻塞 (如 Read socket 拿不到数据)
   │
   ▼
2. gopark()
   ├─ 保存上下文到 M
   ├─ 状态: _Grunning -> _Gwaiting
   └─ 释放 M，执行 schedule() 找下一个 G 运行
   
   ... (当前 G 沉睡，让出 CPU) ...

3. 外部事件 (如 epoll 返回 socket 可读)
   │
   ▼
4. goready(gp)
   ├─ 状态: _Gwaiting -> _Grunnable
   └─ 放入 P 的 run queue (排队)

   ... (当前 G 在队列中等待) ...

5. 调度器 schedule() 选中该 G
   │
   ▼
6. execute(gp)
   ├─ 状态: _Grunnable -> _Grunning
   └─ 恢复 PC/SP，从 gopark() 下一行代码继续执行
   │
   ▼
[业务代码继续] (例如再次尝试 Read socket，此时已有数据)
```