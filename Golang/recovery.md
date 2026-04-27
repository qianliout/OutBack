# Go 协程 panic 与 recover 三种场景笔记

本文结合 `main.go` 里的三个实验场景，解释下面三个问题：
1. 二级协程 panic 了，主协程会怎么样
2. 主协程能不能 recover 二级协程的 panic
3. 二级协程自己 recover 后，程序会怎么样

## 一、实验代码结构

运行方式：

```bash
go run . case1
go run . case2
go run . case3
```

代码核心结构是：
1. `main` 根据参数选择 `case1/case2/case3`
2. 每个 case 都是 `main` 启动一级协程
3. 一级协程再启动二级协程
4. panic 都发生在二级协程中

这样可以避免“main 直接启动 panic 协程”的争议，专注观察协程层级下的行为。

## 二、场景 1：二级协程 panic，无 recover

命令：

```bash
go run . case1
```

关键代码逻辑：

```go
go func() { // level1
    go func() { // level2
        panic("...")
    }()
}()
```

现象：
1. 控制台打印二级协程即将 panic
2. 进程直接退出，常见是 `exit status 2`
3. `main` 后续逻辑通常来不及正常收尾

原因：
1. panic 发生在某个 goroutine 时，运行时会先在当前 goroutine 的调用栈里做栈展开
2. 栈展开过程中会执行该 goroutine 已注册的 defer
3. 如果最终没有被 recover，panic 会升级为未处理致命错误
4. Go 运行时会终止整个进程，而不是只杀掉这个 goroutine

一句话总结：
二级协程的未恢复 panic，最终会导致整个程序崩溃。

## 三、场景 2：主协程写了 recover，但 panic 在二级协程

命令：

```bash
go run . case2
```

关键代码逻辑：

```go
defer func() {
    if r := recover(); r != nil {
        fmt.Println("main recovered:", r)
    }
}()

go func() { // level1
    go func() { // level2
        panic("...")
    }()
}()
```

现象：
1. 依然崩溃，依然是 `exit status 2`
2. 主协程里的 recover 不会打印“成功恢复二级协程 panic”

原因：
1. recover 只对当前 goroutine 的 panic 生效
2. main 的 defer recover 只能接住 main 自己调用栈上的 panic
3. 二级协程是独立的调用栈，main 无法跨 goroutine 抓它的 panic

一句话总结：
recover 不能跨 goroutine 工作，主协程 recover 不住子协程 panic。

## 四、场景 3：二级协程 panic，但二级协程自己 recover

命令：

```bash
go run . case3
```

关键代码逻辑：

```go
go func() { // level1
    go func() { // level2
        defer func() {
            if r := recover(); r != nil {
                fmt.Println("level2 recovered:", r)
            }
        }()
        panic("...")
    }()
}()
```

现象：
1. 可以看到二级协程打印 recover 成功
2. 进程不崩溃，main 能继续执行并正常退出

原因：
1. panic 在二级协程发生
2. 二级协程自己的 defer recover 在同一个 goroutine 调用栈中执行
3. panic 被消费后，不再向运行时升级为未处理致命错误
4. 因此程序整体保持存活

一句话总结：
谁 panic，最好谁 recover，至少要在同一个 goroutine 内 recover。

## 五、底层原理展开

panic/recover 的关键规则可以压缩成四条：
1. panic 是沿当前 goroutine 的栈向上展开，不会跳到其他 goroutine 栈
2. defer 在栈展开时按后进先出执行
3. recover 必须在 defer 函数中调用才有效
4. recover 只能捕获当前 goroutine 正在展开中的 panic

可以理解为每个 goroutine 都有自己的“故障域”：
1. 故障默认只在本 goroutine 栈里传播
2. 如果本域内没人兜底，最终会触发进程级崩溃

## 六、工程实践建议

建议模式：
1. 每个长期运行的 worker goroutine 在入口统一加 defer recover
2. recover 后记录日志、打点、上报，再决定是否重启 worker
3. 不要把“主协程 recover 一切”当成兜底方案
4. 对外提供服务时，把异常隔离在 goroutine 边界内

一个常见模板：

```go
go func() {
    defer func() {
        if r := recover(); r != nil {
            fmt.Printf("worker panic recovered: %v\n", r)
        }
    }()
    // worker body
}()
```

## 七、三种场景对照表

| 场景 | panic 位置 | recover 位置 | 结果 |
| --- | --- | --- | --- |
| case1 | 二级协程 | 无 | 进程崩溃 |
| case2 | 二级协程 | 主协程 | 进程崩溃 |
| case3 | 二级协程 | 二级协程自身 | 进程存活 |

最终结论：
1. 子协程 panic 不会被主协程自动吸收
2. recover 不能跨 goroutine
3. 在 panic 所在 goroutine 内 recover 才能真正止损
