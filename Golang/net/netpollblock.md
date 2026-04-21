src/runtime/netpoll.go

```go
// returns true if IO is ready, or false if timed out or closed
// waitio - wait only for completed IO, ignore errors
// Concurrent calls to netpollblock in the same mode are forbidden, as pollDesc
// can hold only a single waiting goroutine for each mode.
func netpollblock(pd *pollDesc, mode int32, waitio bool) bool {
	gpp := &pd.rg
	if mode == 'w' {
		gpp = &pd.wg
	}

	// set the gpp semaphore to pdWait
	for {
		// Consume notification if already ready.
		if gpp.CompareAndSwap(pdReady, pdNil) {
			return true
		}
		if gpp.CompareAndSwap(pdNil, pdWait) {
			break
		}

		// Double check that this isn't corrupt; otherwise we'd loop
		// forever.
		if v := gpp.Load(); v != pdReady && v != pdNil {
			throw("runtime: double wait")
		}
	}

	// need to recheck error states after setting gpp to pdWait
	// this is necessary because runtime_pollUnblock/runtime_pollSetDeadline/deadlineimpl
	// do the opposite: store to closing/rd/wd, publishInfo, load of rg/wg
	if waitio || netpollcheckerr(pd, mode) == pollNoError {
		gopark(netpollblockcommit, unsafe.Pointer(gpp), waitReasonIOWait, traceBlockNet, 5)
	}
	// be careful to not lose concurrent pdReady notification
	old := gpp.Swap(pdNil)
	if old > pdWait {
		throw("runtime: corrupted polldesc")
	}
	return old == pdReady
}
```

我来详细解释这个 `netpollblock` 函数。这是 Go 运行时中网络轮询的核心函数之一。

## 函数概述

```go
func netpollblock(pd *pollDesc, mode int32, waitio bool) bool
```

**功能**: 阻塞当前 goroutine 直到 I/O 就绪，或超时，或关闭。

**参数**:
- `pd`: poll 描述符，包含 I/O 状态信息
- `mode`: `'r'` 表示读就绪，`'w'` 表示写就绪
- `waitio`: `true` 表示只等待已完成的 I/O（忽略错误）

---

## 逐段解析

### 1. 选择对应的等待队列

```go
gpp := &pd.rg
if mode == 'w' {
    gpp = &pd.wg
}
```

`pollDesc` 有两个等待队列：
- `rg`: 读等待队列（read goroutine）
- `wg`: 写等待队列（write goroutine）

根据 mode 选择对应的指针。

### 2. CAS 循环设置 pdWait 状态

```go
for {
    // 如果已经就绪，直接消费通知
    if gpp.CompareAndSwap(pdReady, pdNil) {
        return true
    }
    // 尝试将自己设置为等待状态
    if gpp.CompareAndSwap(pdNil, pdWait) {
        break
    }
    // 双重检查：防止损坏
    if v := gpp.Load(); v != pdReady && v != pdNil {
        throw("runtime: double wait")
    }
}
```

这里使用 **CAS (Compare-And-Swap)** 实现自旋锁：

| 状态 | 含义 |
|------|------|
| `pdNil` | 空闲状态 |
| `pdReady` | I/O 已就绪 |
| `pdWait` | 有 goroutine 正在等待 |

**三种情况**：
1. **已是 pdReady** → 有人已唤醒你，直接返回 true
2. **是 pdNil** → 成功抢占到等待位置，退出循环
3. **其他状态** → 损坏，panic

### 3. 实际阻塞 goroutine

```go
if waitio || netpollcheckerr(pd, mode) == pollNoError {
    gopark(netpollblockcommit, unsafe.Pointer(gpp), waitReasonIOWait, traceBlockNet, 5)
}
```

- 如果 `waitio=true` 或没有错误 → 调用 `gopark` **真正阻塞**
- `gopark` 会将当前 goroutine 挂起，让出调度器
- `netpollblockcommit` 是唤醒时的回调函数

### 4. 唤醒后清理

```go
old := gpp.Swap(pdNil)
if old > pdWait {
    throw("runtime: corrupted polldesc")
}
return old == pdReady
```

- 唤醒后，将状态重置为 `pdNil`
- 检查旧状态是否合法
- 返回 `true` 表示 I/O 就绪，`false` 表示超时/关闭

---

## 状态转换图

```
                    ┌─────────────┐
                    │   pdNil     │ (空闲)
                    └──────┬──────┘
                           │ CAS(pdNil, pdWait)
                           ▼
                    ┌─────────────┐
         ┌─────────│   pdWait    │─────────┐
         │         └─────────────┘         │
         │ goroutine 被 park               │ 被唤醒
         │                                 │
         ▼                                 ▼
┌─────────────────┐               ┌─────────────┐
│ gopark 阻塞中    │               │   pdReady   │
└─────────────────┘               └──────┬──────┘
                                         │ CAS(pdReady, pdNil)
                                         ▼
                                  返回 true
```

---

## 核心设计思想

1. **无锁化设计**: 使用 CAS 原子操作避免锁竞争
2. **单等待者**: 每个 pollDesc 每种模式只能有一个等待 goroutine
3. **延迟唤醒**: 通过 gopark 实现 goroutine 调度级别的阻塞
4. **错误检查**: 阻塞前检查是否有错误，避免无效等待

这是一个精心设计的底层同步原语，用于在 Go 运行时层面实现高效的网络 I/O 多路复用。