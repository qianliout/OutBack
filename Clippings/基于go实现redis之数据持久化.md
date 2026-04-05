---
title: "基于go实现redis之数据持久化"
source: "https://mp.weixin.qq.com/s/tOyZQ2UPdRrBRlQDCKgWIQ"
author:
  - "[[小徐先生1212]]"
published:
created: 2026-04-04
description: "祝贺，至此，整个【基于go实现redis】系列内容全部完成！\x0d\x0a本篇为基于go实现redis之数据持久化篇： 介绍 goredis 关于 aof 持久化机制的实现以及有关于aof重写策略的执行细节"
tags:
  - "clippings"
---
原创 小徐先生1212 *2024年5月5日 13:26*

## 0 前言

欢迎回来，由我们一起继续推进技术分享专题—— **【基于go实现redis之数据持久化】**.

此前我已于 github 开源项目——goredis，由于我个人水平有限，如有实现不到位之处，欢迎批评指正：https://github.com/xiaoxuxiansheng/goredis

![图片](https://mmbiz.qpic.cn/sz_mmbiz_png/3ic3aBqT2ibZuNLYow0WdMKbNdSeBQxjSCgpAm7Ym8S1Sq0WgyqsgzU31Cq7ujZauEibW8gVTgbxWe1EaQRgo1z1Q/640?wx_fmt=png&from=appmsg&tp=webp&wxfrom=5&wx_lazy=1#imgIndex=0)

本系列正是围绕着开源项目goredis展开，共分为四篇内容，本文为该系列的完结篇——数据持久化篇：

- • [**基于go实现redis之主干框架（已完成）：** 在宏观视角下纵览 goredis 整体架构，梳理各模块间的关联性](http://mp.weixin.qq.com/s?__biz=MzkxMjQzMjA0OQ==&mid=2247485070&idx=1&sn=e6fc425dee04746648dfb0bc4c3b2313&chksm=c10c4850f67bc14620a8824cac0734755d5404ad6a8f3d541507b2ec14860832ea5fe6198d74&scene=21#wechat_redirect)
- • [**基于go实现redis之指令分发（已完成）：** 聚焦介绍 goredis 服务端如何启动和运行，并在接收客户端请求后实现指令协议的解析和分发](http://mp.weixin.qq.com/s?__biz=MzkxMjQzMjA0OQ==&mid=2247485081&idx=1&sn=68319157e3e6cd3133738ceda4e41872&chksm=c10c4847f67bc151e6379391b4deffca482423429e41e52bf0f159aaeb5413668fab7030a940&scene=21#wechat_redirect)
- • [**基于go实现redis之存储引擎（已完成）：** 聚焦介绍数据存储层中单协程无锁化执行框架，各类基本数据类型的底层实现细节，以及过期数据的惰性和定期回收机制](http://mp.weixin.qq.com/s?__biz=MzkxMjQzMjA0OQ==&mid=2247485098&idx=1&sn=72a09e4d91fa3c5937cd56086ba7a80a&chksm=c10c4874f67bc1623860e30901fbd1e8b4b4961035df5d2b5c44021fb734ea73f0bae41c76c8&scene=21#wechat_redirect)
- • **基于go实现redis之数据持久化（本篇）：** 介绍goredis关于aof持久化机制的实现以及有关于aof重写策略的执行细节

（此外，这里需要特别提及一下，在学习过程中，很大程度上需借助了 **hdt3213 系列博客和项目** 的帮助，在此致敬一下：https://www.cnblogs.com/Finley/category/1598973.html）

## 1 redis 持久化机制

redis 中的数据均存储在内存，而 **内存** 属于一种 **易失性存储介质** ，一旦进程或者机器崩溃，都会导致数据的丢失. 为了提高数据存储的稳定性， **redis 建立了两种将内存数据溢写到磁盘的持久化机制，包括 rdb 和 aof 两种**.

## 1.1 rdb

![图片](data:image/svg+xml,%3C%3Fxml version='1.0' encoding='UTF-8'%3F%3E%3Csvg width='1px' height='1px' viewBox='0 0 1 1' version='1.1' xmlns='http://www.w3.org/2000/svg' xmlns:xlink='http://www.w3.org/1999/xlink'%3E%3Ctitle%3E%3C/title%3E%3Cg stroke='none' stroke-width='1' fill='none' fill-rule='evenodd' fill-opacity='0'%3E%3Cg transform='translate(-249.000000, -126.000000)' fill='%23FFFFFF'%3E%3Crect x='249' y='126' width='1' height='1'%3E%3C/rect%3E%3C/g%3E%3C/g%3E%3C/svg%3E)

**rdb，全称 redis database** ，指的是一种将 **内存数据序列化生成快照文件的持久化机制**.

**rdb 持久化方式注重的是结果，** 由于每轮 rdb 都涉及到全量数据的操作，因此从单次持久化过程来看显得比较耗费性能，但是从结果来看，rdb 是一种比较节省和高效的持久化方式：一方面，rdb 文件可以 **对文件内容高度压缩，从而实现存储空间的节省** ；另一方面， **快照形式的持久化内容能保证不会存在冗余的内容** ，在加载还原内存数据时也会更加的高效.

## 1.2 aof

![图片](data:image/svg+xml,%3C%3Fxml version='1.0' encoding='UTF-8'%3F%3E%3Csvg width='1px' height='1px' viewBox='0 0 1 1' version='1.1' xmlns='http://www.w3.org/2000/svg' xmlns:xlink='http://www.w3.org/1999/xlink'%3E%3Ctitle%3E%3C/title%3E%3Cg stroke='none' stroke-width='1' fill='none' fill-rule='evenodd' fill-opacity='0'%3E%3Cg transform='translate(-249.000000, -126.000000)' fill='%23FFFFFF'%3E%3Crect x='249' y='126' width='1' height='1'%3E%3C/rect%3E%3C/g%3E%3C/g%3E%3C/svg%3E)

**aof，全称 append only file，是一种通过记录增量变化信息，从而回溯出全量数据的持久化策略.**

aof 的优势是，每次以指令的粒度进行持久化，因此从单次行为来看比较轻便和高效，但是这种方式也不可避免地存在两个缺陷：

- • **持久化数据存在冗余：aof 持久化方式注重的是过程.** 针对同一笔 kv 数据，不论执行多少次变更操作都会被 aof 事无巨细地记录下来，比如用户执行了 1000 次 set a b 指令，从结果来看，操作结果仅影响了一条记录，但是 aof 持久化方式却需要保留 1000 条变更记录，尽管其中 999 条都是无用信息
- • **数据还原流程低效：** 在还原内存数据时，需要通过遍历执行指令的方式，完整回溯一遍内存数据库的变更时间线，是一种很低效的数据加载方式.

受限于笔者水平，对 redis rdb 持久化流程的认知水平还有所不足，本次 goredis 项目实现中，仅涉及到对 aof 持久化机制的实现.

## 2 goredis 持久化模块定位

## 2.1 持久化模块接口定义

![图片](data:image/svg+xml,%3C%3Fxml version='1.0' encoding='UTF-8'%3F%3E%3Csvg width='1px' height='1px' viewBox='0 0 1 1' version='1.1' xmlns='http://www.w3.org/2000/svg' xmlns:xlink='http://www.w3.org/1999/xlink'%3E%3Ctitle%3E%3C/title%3E%3Cg stroke='none' stroke-width='1' fill='none' fill-rule='evenodd' fill-opacity='0'%3E%3Cg transform='translate(-249.000000, -126.000000)' fill='%23FFFFFF'%3E%3Crect x='249' y='126' width='1' height='1'%3E%3C/rect%3E%3C/g%3E%3C/g%3E%3C/svg%3E)

在 goredis 项目中，定义出一个持久化模块 persister，该模块包含两个明确的核心作用：

- • **数据持久化：** 内存数据库发生变化时，持久化记录增量变更内容
- • **数据重加载：** 在服务重启时，能够读取持久化内容还原出内存中的数据

goredis 针对于持久化模块定义了一个抽象 interface，对应声明了 Reloader 和 PersistCmd 两个方法，代码位于 handler/persister.go：

```
// 持久化模块
type Persister interface {
    // 获取一个 reader，用于读取之前持久化的内容
    Reloader() (io.ReadCloser, error) 
    // 写入增量持久化指令的入口
    PersistCmd(ctx context.Context, cmd [][]byte) 
    // 关闭持久化模块
    Close() 
}
```

## 2.2 增量指令持久化

![图片](data:image/svg+xml,%3C%3Fxml version='1.0' encoding='UTF-8'%3F%3E%3Csvg width='1px' height='1px' viewBox='0 0 1 1' version='1.1' xmlns='http://www.w3.org/2000/svg' xmlns:xlink='http://www.w3.org/1999/xlink'%3E%3Ctitle%3E%3C/title%3E%3Cg stroke='none' stroke-width='1' fill='none' fill-rule='evenodd' fill-opacity='0'%3E%3Cg transform='translate(-249.000000, -126.000000)' fill='%23FFFFFF'%3E%3Crect x='249' y='126' width='1' height='1'%3E%3C/rect%3E%3C/g%3E%3C/g%3E%3C/svg%3E)

在 DB 的数据执行层中，针对数据的操作分为读操作和写操作两类：

- • **读操作：** 如 get，sismember 等指令，并不会引起底层数据状态的变化，因此无需持久化
- • **写操作：** 如 hset、sadd 等，则会通过 persister 统一对操作指令进行持久化

以 hset、sadd 两个写操作指令的代码实例加以说明：

```
func (k *KVStore) HSet(cmd *database.Command) handler.Reply {
    // 1 ...执行数据写入 hashmap 操作

    // 2 对 hset 指令进行持久化 
    k.persister.PersistCmd(cmd.Ctx(), cmd.Cmd()) // 持久化
    // 3 ...返回响应结果
}
```
```
// set
func (k *KVStore) SAdd(cmd *database.Command) handler.Reply {
    // 1 ...执行数据写入 set 操作
    // 2 持久化 sadd 指令
    k.persister.PersistCmd(cmd.Ctx(), cmd.Cmd()) 
    // 3 ...返回响应结果
}
```

值得一提的是， **涉及到过期时间的写操作** 在持久化时需要一些特殊的处理技巧，在 goredis 当前的实现中涉及到 set、expire、expireAt 三个指令.

持久化过期时间的要点在于， **不能以一个相对的过期时间 ttl（time to live）作为持久化目标** ，否则可能导致过期信息的失真：比如在 11:00 执行指令 expire key 3600，那么预期 key 应该于 12:00 过期；倘若在 11:30 加载持久化内容，就会把过期时间错误推断为 12:30

基于以上，goredis 在持久化过期时间时， **统一采用的是一个过期时间点绝对值** ，无论是 set ex 还是 expire 指令，在持久化内容上都需要转为 expireAt 的形式.

下面以 Expire 指令持久化流程的代码示例加以说明：

```
// 执行 expire 指令.
func (k *KVStore) Expire(cmd *database.Command) handler.Reply {
    // 1 读取执行参数
    args := cmd.Args()
    // 2 读取操作的 key
    key := string(args[0])
    // 3 获取过期时间 ttl
    ttl, err := strconv.ParseInt(string(args[1]), 10, 64)
    
    // 4 根据 ttl 推算出过期时间点
    expireAt := lib.TimeNow().Add(time.Duration(ttl) * time.Second)
    // 5 构造出要持久化的 expireAt 指令
    _cmd := [][]byte{[]byte(database.CmdTypeExpireAt), []byte(key), []byte(lib.TimeSecondFormat(expireAt))}
    // 6 在 expireAt 方法中，完成过期时间设置操作，并且完成指令持久化
    return k.expireAt(cmd.Ctx(), _cmd, key, expireAt)
}
```
```
func (k *KVStore) expireAt(ctx context.Context, cmd [][]byte, key string, expireAt time.Time) handler.Reply {
    // 1 执行过期时间设置操作
    k.expire(key, expireAt)
    // 2 持久化过期指令
    k.persister.PersistCmd(ctx, cmd) // 持久化
    // ...
}
```

## 2.3 持久化数据恢复

![图片](data:image/svg+xml,%3C%3Fxml version='1.0' encoding='UTF-8'%3F%3E%3Csvg width='1px' height='1px' viewBox='0 0 1 1' version='1.1' xmlns='http://www.w3.org/2000/svg' xmlns:xlink='http://www.w3.org/1999/xlink'%3E%3Ctitle%3E%3C/title%3E%3Cg stroke='none' stroke-width='1' fill='none' fill-rule='evenodd' fill-opacity='0'%3E%3Cg transform='translate(-249.000000, -126.000000)' fill='%23FFFFFF'%3E%3Crect x='249' y='126' width='1' height='1'%3E%3C/rect%3E%3C/g%3E%3C/g%3E%3C/svg%3E)

在 goredis 服务重试时，会通过持久化模块 persister 获取指令加载器 reloader，从中依次读取此前完成持久化的指令. 在具体技术实现上，goredis 会 **将 reloader 包装成一个虚拟连接，模拟服务端接收到客户端请求指令的流程，依次将指令分发到存储引擎层** ，最终实现内存数据的恢复：

```
// 启动指令分发器
func (h *Handler) Start() error {
    // 1 加载持久化指令，还原出内存中的数据
    reloader, err := h.persister.Reloader()
    if err != nil {
        return err
    }
    defer reloader.Close()
    // 2 把持久化指令 reloader 包装成一个 fake 的连接，执行 handler 分发指令主流程
    h.handle(SetLoadingPattern(context.Background()), newFakeReaderWriter(reloader))
    return nil
}
```
```
func (h *Handler) handle(ctx context.Context, conn io.ReadWriter) {
    // 借助 parser 将连接转为 chan 形式，并持续接收解析到的指令
    stream := h.parser.ParseStream(conn)
    for {
        select {
        // ...
        // 接收到来自 chan 的指令，分发到存储引擎层执行
        case droplet := <-stream:
            if err := h.handleDroplet(ctx, conn, droplet); err != nil {
                // ...
            }
        }
    }
}
```

## 3 goredis aof 持久化实现

介绍完上游模块是如何对 persister 展开使用后，下面就打开 persister 模块的黑盒子，解释 goredis 中针对 aof 持久化机制的实现细节.

## 3.1 类定义

![图片](data:image/svg+xml,%3C%3Fxml version='1.0' encoding='UTF-8'%3F%3E%3Csvg width='1px' height='1px' viewBox='0 0 1 1' version='1.1' xmlns='http://www.w3.org/2000/svg' xmlns:xlink='http://www.w3.org/1999/xlink'%3E%3Ctitle%3E%3C/title%3E%3Cg stroke='none' stroke-width='1' fill='none' fill-rule='evenodd' fill-opacity='0'%3E%3Cg transform='translate(-249.000000, -126.000000)' fill='%23FFFFFF'%3E%3Crect x='249' y='126' width='1' height='1'%3E%3C/rect%3E%3C/g%3E%3C/g%3E%3C/svg%3E)

**aofPersister** 是对 persister 的具体实现，其实例在启动时，会 **伴生启动一个异步守护协程，持续接收来自数据执行层投递的增量指令并为之完成持久化操作.**

有关 aof 持久化模块的类定义如下：

- • 通过 ctx 实现守护协程的生命周期控制
- • 通过 buffer channel 接收来自数据执行层 executor 投递的写指令
- • 通过 aofFile 文件持久化记录写指令内容
- • 此外基于一系列配置参数，定义 aof 落盘和指令重写的策略
```
// aof 持久化模块实现类
type aofPersister struct {
    // 生命周期控制
    ctx    context.Context
    cancel context.CancelFunc

    // 持久化指令接收 channel. 同时带有缓冲区的作用
    buffer                 chan [][]byte
    // aof 持久化存储文件
    aofFile                *os.File
    // aof 文件名称
    aofFileName            string
    // aof 持久化策略. 
    appendFsync            appendSyncStrategy
    // 每持久化多少条 aof 指令后，进行一次 aof 重写
    autoAofRewriteAfterCmd int64
    // 记录距离上一次 aof 重写后，当前已经执行了多少条 aof 指令
    aofCounter             atomic.Int64
    // 互斥锁. 保护 aof 文件并发安全
    mu   sync.Mutex
    once sync.Once
}
```

## 3.2 持久化策略

与 redis aof 机制相对应，在 goredis 的实现中，将 aof 持久化策略同样划分为三种等级：

```
// aof 持久化等级 always | everysec | no
type appendSyncStrategy string

const (
    alwaysAppendSyncStrategy   appendSyncStrategy = "always" // 每条指令都进行持久化落盘
    everysecAppendSyncStrategy appendSyncStrategy = "everysec" // 每秒批量执行一次持久化落盘
    noAppendSyncStrategy       appendSyncStrategy = "no" // 不主动进行指令的持久化落盘，由设备自行决定落盘节奏
)
```

在构造 persister 实例时，会读取 redis.conf 中的配置信息，决定是否启用 aof 持久化策略，以及对应的持久化策略级别：

```
// thinker 为收拢了持久化配置参数的 interface
func NewPersister(thinker Thinker) (handler.Persister, error) {
    // 不启用 aof 持久化
    if !thinker.AppendOnly() {
        return newFakePersister(nil), nil
    }
     // 启用 aof 持久化
    return newAofPersister(thinker)
}

type Thinker interface {
    // 是否启用 aof 持久化
    AppendOnly() bool
    // aof 文件名称
    AppendFileName() string
    // aof 持久化策略级别
    AppendFsync() string
    // 每持久化多少条 aof 指令后进行一次重写
    AutoAofRewriteAfterCmd() int
}
```
```
// 构造 aof 持久化模块
func newAofPersister(thinker Thinker) (handler.Persister, error) {
    // aof 文件名称
    aofFileName := thinker.AppendFileName()
    // 打开 aof 文件
    aofFile, err := os.OpenFile(aofFileName, os.O_APPEND|os.O_CREATE|os.O_RDWR, 0600)
    
    // 构造持久化模块实例
    a := aofPersister{
        ctx:         ctx,
        cancel:      cancel,
        buffer:      make(chan [][]byte, 1<<10),
        aofFile:     aofFile,
        aofFileName: aofFileName,
    }

    // 判断是否启用了 aof 指令重写策略
    if autoAofRewriteAfterCmd := thinker.AutoAofRewriteAfterCmd(); autoAofRewriteAfterCmd > 1 {
        a.autoAofRewriteAfterCmd = int64(autoAofRewriteAfterCmd)
    }

    // 设置 aof 持久化策略级别
    switch thinker.AppendFsync() {
    case alwaysAppendSyncStrategy.string():
        a.appendFsync = alwaysAppendSyncStrategy
    case everysecAppendSyncStrategy.string():
        a.appendFsync = everysecAppendSyncStrategy
    default:
        a.appendFsync = noAppendSyncStrategy // 默认策略
    }

    // 启动持久化模块常驻运行协程
    pool.Submit(a.run)
    return &a, nil
}
```

## 3.3 核心方法

![图片](data:image/svg+xml,%3C%3Fxml version='1.0' encoding='UTF-8'%3F%3E%3Csvg width='1px' height='1px' viewBox='0 0 1 1' version='1.1' xmlns='http://www.w3.org/2000/svg' xmlns:xlink='http://www.w3.org/1999/xlink'%3E%3Ctitle%3E%3C/title%3E%3Cg stroke='none' stroke-width='1' fill='none' fill-rule='evenodd' fill-opacity='0'%3E%3Cg transform='translate(-249.000000, -126.000000)' fill='%23FFFFFF'%3E%3Crect x='249' y='126' width='1' height='1'%3E%3C/rect%3E%3C/g%3E%3C/g%3E%3C/svg%3E)

aofPersister 的伴生写成会持续运行 run 方法，通过 for + select 的运行框架，持续监听 buffer channel，接收来自上游数据执行层 executor 投递的增量写指令，并对其完成持久化操作：

```
func (a *aofPersister) run() {
    // 倘若持久化策略级别为每秒落盘一次，则额外启动一个协程负责该事项
    if a.appendFsync == everysecAppendSyncStrategy {
        pool.Submit(a.fsyncEverySecond)
    }

    for {
        select {
        // ...
        // 接收到来自上游投递的指令，对其进行持久化
        case cmd := <-a.buffer:
            // 指令写入 aof 文件
            a.writeAof(cmd)
            // 更新已聚合的 aof 指令计数器
            a.aofTick()
        }
    }
}
```

倘若 redis.conf 中定义的持久化策略级别为 everySec，则会额外启动一个协程负责每过一秒对 aof 文件执行一次 fsync 操作，确保其中的持久化指令安全落盘.

（此处需要补充设定：通过 file.Write 操作，本质上只会将持久化操作提交到设备 io 队列中，其具体的执行时机是不确定的. 倘若希望持久化操作能够立即得到执行并且能够明确感知到执行结果，则需要执行文件 fsync 操作）

```
func (a *aofPersister) fsyncEverySecond() {
    ticker := time.NewTicker(time.Second)
    for {
        select {
        // ...
        case <-ticker.C:
            if err := a.fsync(); err != nil {
                // log
            }
        }
    }
}
```

当 aofPersister 从 channel 中获取到拟持久化的写指令后，则会将其格式化成 multi bulk reply 的形式，并调用 file.Write 方法将落盘操作提交到设备 io 队列中. 但是倘若 redis.conf 中定义的持久化级别为 always，则会在此处立即执行一次 fsync 操作，确保指令当即被落盘.

```
func (a *aofPersister) writeAof(cmd [][]byte) {
    // 1 加锁保证并发安全
    a.mu.Lock()
    defer a.mu.Unlock()

    // 2 将指令封装为 multi bulk reply 形式
    persistCmd := handler.NewMultiBulkReply(cmd)
    // 3 指令 append 写入到 aof 文件
    if _, err := a.aofFile.Write(persistCmd.ToBytes()); err != nil {
        // log
        return
    }

    // 4 除非持久化策略等级为 always，否则不需要立即执行 fsync 操作，强制进行指令落盘(性能较差)
    if a.appendFsync != alwaysAppendSyncStrategy {
        return
    }

    // 5 fsync 操作，指令强制落盘
    if err := a.fsyncLocked(); err != nil {
        // log
    }
}
```

而在 goredis 重启时机下的内存数据重加载流程中，指令分发器 handler 会调用 persister 的 Reloader 方法，将 aof 持久化文件包装成一个虚拟连接，然后持续读取其中已完成持久化的指令，对内存数据进行加载：

```
func (a *aofPersister) Reloader() (io.ReadCloser, error) {
    file, err := os.Open(a.aofFileName)
    if err != nil {
        return nil, err
    }
    _, _ = file.Seek(0, io.SeekStart)
    return file, nil
}
```

## 4 goredis aof 指令重写流程

1.2 小节中有提到，aof 持久化方式存在的一大弊端，就是存在数据冗余问题. 一笔 kv 数据的终态可能是经历了一系列变更指令的执行后才得到的结果，但是从结果导向来看，在持久化时其实只需要通过一笔指令记录其终态结果，其中繁杂的变更过程并不需要被关心.

综上， **在 goredis 中建立了一套 aof 指令“瘦身”机制. 该机制的实现思路是，针对内存数据“拷贝”出一份副本，然后从结果出发将其映射成一条简单的 aof 指令，从而实现冗余指令的去重.**

这个流程虽然描述起来逻辑清晰，但在具体实现过程中，需要兼顾保证持久化主流程与重写流程的并行执行，并兼顾持久化内容的正确性和稳定性，还是有很多技术细节值得探讨.

## 4.1 重写流程启动

![图片](data:image/svg+xml,%3C%3Fxml version='1.0' encoding='UTF-8'%3F%3E%3Csvg width='1px' height='1px' viewBox='0 0 1 1' version='1.1' xmlns='http://www.w3.org/2000/svg' xmlns:xlink='http://www.w3.org/1999/xlink'%3E%3Ctitle%3E%3C/title%3E%3Cg stroke='none' stroke-width='1' fill='none' fill-rule='evenodd' fill-opacity='0'%3E%3Cg transform='translate(-249.000000, -126.000000)' fill='%23FFFFFF'%3E%3Crect x='249' y='126' width='1' height='1'%3E%3C/rect%3E%3C/g%3E%3C/g%3E%3C/svg%3E)

在 aofPersister 每执行一条指令的持久化动作时，都会通过计数器进行记录，当执行的指令数量达到一定阈值后，就会异步启动一次 aof 重写流程：

```
// 记录执行的 aof 指令次数
func (a *aofPersister) aofTick() {
    // 如果阈值 <= 1，代表不启用 aof 指令重写策略
    if a.autoAofRewriteAfterCmd <= 1 {
        return
    }

    // 累加指令执行计数器
    if ticked := a.aofCounter.Add(1); ticked < int64(a.autoAofRewriteAfterCmd) {
        return
    }

    // 执行 aof 指令持久化次数达到重写阈值，进行重写，并将计数器清零
    _ = a.aofCounter.Add(-a.autoAofRewriteAfterCmd)
    pool.Submit(func() {
        if err := a.rewriteAOF(); err != nil {
            // log
        }
    })
}
```

## 4.2 重写详细步骤

![图片](data:image/svg+xml,%3C%3Fxml version='1.0' encoding='UTF-8'%3F%3E%3Csvg width='1px' height='1px' viewBox='0 0 1 1' version='1.1' xmlns='http://www.w3.org/2000/svg' xmlns:xlink='http://www.w3.org/1999/xlink'%3E%3Ctitle%3E%3C/title%3E%3Cg stroke='none' stroke-width='1' fill='none' fill-rule='evenodd' fill-opacity='0'%3E%3Cg transform='translate(-249.000000, -126.000000)' fill='%23FFFFFF'%3E%3Crect x='249' y='126' width='1' height='1'%3E%3C/rect%3E%3C/g%3E%3C/g%3E%3C/svg%3E)

aof 指令重写流程会分为三个核心步骤：

- • **重写前准备** ：记录此刻 aof 文件的大小，并拷贝生成一份 aof 临时文件副本（此阶段需要加锁，短暂停止 aof 持久化流程）
- • **重写进行时** ：拷贝生成一份内存数据副本，并映射成最直观的 aof 指令落在 aof 临时文件副本中（此流程可以与 aof 持久化流程并发进行）
- • **重写收尾：** 将原 aof 文件后继部分（在重写进行时，会有新的指令在并发地执行持久化操作）追加到 aof 临时文件副本中，然后使用临时文件副本覆盖原 aof 文件（此阶段需要加锁，暂停 aof 持久化流程）
```
// 重写 aof 文件
func (a *aofPersister) rewriteAOF() error {
    // 1 重写前处理. 需要短暂加锁
    tmpFile, fileSize, err := a.startRewrite()
    if err != nil {
        return err
    }

    // 2 aof 指令重写. 与主流程并发执行
    if err = a.doRewrite(tmpFile, fileSize); err != nil {
        return err
    }

    // 3 完成重写. 需要短暂加锁
    return a.endRewrite(tmpFile, fileSize)
}
```

在 **重写前准备阶段** 中，包含如下详细步骤：

- • **加锁：** 暂停正常的 aof 持久化流程
- • **获取当前 aof 文件大小，目的是对 aof 文件做个切割，** 以便后续能区分出 aof 原文件中哪部分指令是本轮重写流程所涉及的，哪部分数据是重写开始后新生成的，针对后者会在本轮重写中进行忽略，留待后续轮次再作处理（这样设计的目的，本质上是为了能够在重写进行时和正常的 aof 持久化流程并发进行）
- • **创建一个 aof 临时文件副本，** 用于承载重写后的指令内容
- • **解锁：** 恢复正常的 aof 持久化流程
```
func (a *aofPersister) startRewrite() (*os.File, int64, error) {
    a.mu.Lock()
    defer a.mu.Unlock()

    if err := a.aofFile.Sync(); err != nil {
        return nil, 0, err
    }

    fileInfo, _ := os.Stat(a.aofFileName)
    fileSize := fileInfo.Size()

    // 创建一个临时的 aof 文件
    tmpFile, err := os.CreateTemp("./", "*.aof")
    if err != nil {
        return nil, 0, err
    }

    return tmpFile, fileSize, nil
}
```

**重写进行时阶段** ：

- • 读取 aof 原文件中，在重写阶段开始前的所有 aof 指令，构造出一份内存数据的副本
- • **遍历内存数据副本中的每组 kv 数据，将其转换成最直观的 aof 指令，写入到 aof 临时文件副本中** （最核心的重写动作就在于此）
```
func (a *aofPersister) doRewrite(tmpFile *os.File, fileSize int64) error {
    forkedDB, err := a.forkDB(fileSize)
    if err != nil {
        return err
    }

    // 将 db 数据转为 aof cmd
    forkedDB.ForEach(func(key string, adapter database.CmdAdapter, expireAt *time.Time) {
        _, _ = tmpFile.Write(handler.NewMultiBulkReply(adapter.ToCmd()).ToBytes())

        if expireAt == nil {
            return
        }

        expireCmd := [][]byte{[]byte(database.CmdTypeExpireAt), []byte(key), []byte(lib.TimeSecondFormat(*expireAt))}
        _, _ = tmpFile.Write(handler.NewMultiBulkReply(expireCmd).ToBytes())
    })

    return nil
}
```
```
func (a *aofPersister) forkDB(fileSize int64) (database.DataStore, error) {
    file, err := os.Open(a.aofFileName)
    if err != nil {
        return nil, err
    }
    file.Seek(0, io.SeekStart)
    logger := log.GetDefaultLogger()
    reloader := readCloserAdapter(io.LimitReader(file, fileSize), file.Close)
    fakePerisister := newFakePersister(reloader)
    tmpKVStore := datastore.NewKVStore(fakePerisister)
    executor := database.NewDBExecutor(tmpKVStore)
    trigger := database.NewDBTrigger(executor)
    h, err := handler.NewHandler(trigger, fakePerisister, protocol.NewParser(logger), logger)
    if err != nil {
        return nil, err
    }
    if err = h.Start(); err != nil {
        return nil, err
    }
    return tmpKVStore, nil
}
```

在重写收尾阶段，则包含如下步骤：

- • **加锁：** 暂停正常的 aof 持久化流程
- • **将重写阶段开始后写入到 aof 原文件中的持久化指令拷贝追加到 aof 临时文件副本中**. 至此，临时文件副本拥有最完整的持久化内容，分为两部分：重写阶段开始前的部分（已完成重写）；重写阶段开始后的部分（未重写，仅拷贝追加）
- • **使用 aof 临时文件副本覆盖 aof 原文件**
- • **解锁：** 恢复正常的 aof 持久化流程
```
func (a *aofPersister) endRewrite(tmpFile *os.File, fileSize int64) error {
    a.mu.Lock()
    defer a.mu.Unlock()

    // copy commands executed during rewriting to tmpFile
    /* read write commands executed during rewriting */
    src, err := os.Open(a.aofFileName)
    if err != nil {
        return err
    }
    defer func() {
        _ = src.Close()
        _ = tmpFile.Close()
    }()

    if _, err = src.Seek(fileSize, 0); err != nil {
        return err
    }

    // 把老的 aof 文件中后续内容 copy 到 tmp 中
    if _, err = io.Copy(tmpFile, src); err != nil {
        return err
    }

    // 关闭老的 aof 文件，准备废弃
    _ = a.aofFile.Close()
    // 重命名 tmp 文件，作为新的 aof 文件
    if err := os.Rename(tmpFile.Name(), a.aofFileName); err != nil {
        // log
    }

    // 重新开启
    aofFile, err := os.OpenFile(a.aofFileName, os.O_APPEND|os.O_CREATE|os.O_RDWR, 0600)
    if err != nil {
        panic(err)
    }
    a.aofFile = aofFile
    return nil
}
```

至此，一轮 aof 重写流程完成.

## 5 展望

至此，整个【基于go实现redis】系列内容全部完成，在此对本系列内容做个总结：

- • [**基于go实现redis之主干框架（已完成）：** 在宏观视角下纵览 goredis 整体架构，梳理各模块间的关联性](http://mp.weixin.qq.com/s?__biz=MzkxMjQzMjA0OQ==&mid=2247485070&idx=1&sn=e6fc425dee04746648dfb0bc4c3b2313&chksm=c10c4850f67bc14620a8824cac0734755d5404ad6a8f3d541507b2ec14860832ea5fe6198d74&scene=21#wechat_redirect)
- • [**基于go实现redis之指令分发（已完成）：** 聚焦介绍 goredis 服务端如何启动和运行，并在接收客户端请求后实现指令协议的解析和分发](http://mp.weixin.qq.com/s?__biz=MzkxMjQzMjA0OQ==&mid=2247485081&idx=1&sn=68319157e3e6cd3133738ceda4e41872&chksm=c10c4847f67bc151e6379391b4deffca482423429e41e52bf0f159aaeb5413668fab7030a940&scene=21#wechat_redirect)
- • [**基于go实现redis之存储引擎（已完成）：** 聚焦介绍数据存储层中单协程无锁化执行框架，各类基本数据类型的底层实现细节，以及过期数据的惰性和定期回收机制](http://mp.weixin.qq.com/s?__biz=MzkxMjQzMjA0OQ==&mid=2247485098&idx=1&sn=72a09e4d91fa3c5937cd56086ba7a80a&chksm=c10c4874f67bc1623860e30901fbd1e8b4b4961035df5d2b5c44021fb734ea73f0bae41c76c8&scene=21#wechat_redirect)
- • **基于go实现redis之数据持久化（已完成）：** 介绍 goredis 关于 aof 持久化机制的实现以及有关于aof重写策略的执行细节

**微信扫一扫赞赏作者**

继续滑动看下一个

小徐先生的编程世界

向上滑动看下一个