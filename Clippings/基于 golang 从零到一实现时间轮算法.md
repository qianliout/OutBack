---
title: "基于 golang 从零到一实现时间轮算法"
source: "https://mp.weixin.qq.com/s/0HwuYTTe9cdT46advEZe0w"
author:
  - "[[小徐先生1212]]"
published:
created: 2026-04-04
description: "本期和大家探讨了如何基于 golang 从零到一实现时间轮算法，通过原理结合源码，详细展示了单机版和 redis 分布式版时间轮的实现方式."
tags:
  - "clippings"
---
原创 小徐先生1212 *2023年9月23日 21:47*

## 0 前言

近期计划攻坚一个新的专题系列——如何基于 golang 从零到一实现 redis.

这里选择的学习素材是 hdt3213 大佬于 github 上开源的 godis 项目. 在大佬的实现中，充分利用了 golang 的特性，将 redis 存储层由单线程模型转为并发模型，其中在实现数据的 expire 机制时，采用的是单机时间轮模型进行过期数据的删除操作.

godis 项目开源地址：http://github.com/HDT3213/godis

godis 时间轮代码：http://github.com/HDT3213/godis/blob/master/lib/timewheel/timewheel.go

实际上，在我之前分享个人项目——分布式定时器 xtimer 中也有到了时间轮算法，这部分内容具有一定共性. 借此机会，本期单独对时间轮算法的原理和实践内容进行梳理，并基于 golang 从零到一开源实现出一个单机版和 redis 分布式版的时间轮，供大家一起交流探讨.

个人开源的时间轮项目地址为：http://github.com/xiaoxuxiansheng/timewheel

![图片](https://mmbiz.qpic.cn/sz_mmbiz_jpg/3ic3aBqT2ibZtLAAxUVPIlAQBr7yPZJ8wAq4oomEzLl4iblrLTRcmlsCEJAdjvTDzavJLhia5TEct66AcpxjViadYEA/640?wx_fmt=jpeg&tp=webp&wxfrom=5&wx_lazy=1#imgIndex=0)

本期内容的目录大纲如下：

![图片](data:image/svg+xml,%3C%3Fxml version='1.0' encoding='UTF-8'%3F%3E%3Csvg width='1px' height='1px' viewBox='0 0 1 1' version='1.1' xmlns='http://www.w3.org/2000/svg' xmlns:xlink='http://www.w3.org/1999/xlink'%3E%3Ctitle%3E%3C/title%3E%3Cg stroke='none' stroke-width='1' fill='none' fill-rule='evenodd' fill-opacity='0'%3E%3Cg transform='translate(-249.000000, -126.000000)' fill='%23FFFFFF'%3E%3Crect x='249' y='126' width='1' height='1'%3E%3C/rect%3E%3C/g%3E%3C/g%3E%3C/svg%3E)

## 1 时间轮原理

![图片](data:image/svg+xml,%3C%3Fxml version='1.0' encoding='UTF-8'%3F%3E%3Csvg width='1px' height='1px' viewBox='0 0 1 1' version='1.1' xmlns='http://www.w3.org/2000/svg' xmlns:xlink='http://www.w3.org/1999/xlink'%3E%3Ctitle%3E%3C/title%3E%3Cg stroke='none' stroke-width='1' fill='none' fill-rule='evenodd' fill-opacity='0'%3E%3Cg transform='translate(-249.000000, -126.000000)' fill='%23FFFFFF'%3E%3Crect x='249' y='126' width='1' height='1'%3E%3C/rect%3E%3C/g%3E%3C/g%3E%3C/svg%3E)

时间轮是定时任务调度系统中常用到的一种经典的算法模型. 有关时间轮算法的详细概念介绍，可以参见论文:《Hashed and Hierarchical Time Wheels: Data Structures for the Efficient Implementation of a Time Facility》http://www.cs.columbia.edu/~nahum/w6998/papers/sosp87-timing-wheels.pdf

接下来我也会从个人角度出发，谈谈我对于时间轮算法的一些浅显理解.

## 1.1 时间轮概念

聊时间轮之前，先聊聊时间的概念.

首先时间是一维、单向的，我们可以用一条一维的时间轴将其具象化. 我们对时间轴进行刻度拆分，每个刻度对应一个时间范围，那么刻度拆分得越细，则表示的时间范围越精确.

![图片](data:image/svg+xml,%3C%3Fxml version='1.0' encoding='UTF-8'%3F%3E%3Csvg width='1px' height='1px' viewBox='0 0 1 1' version='1.1' xmlns='http://www.w3.org/2000/svg' xmlns:xlink='http://www.w3.org/1999/xlink'%3E%3Ctitle%3E%3C/title%3E%3Cg stroke='none' stroke-width='1' fill='none' fill-rule='evenodd' fill-opacity='0'%3E%3Cg transform='translate(-249.000000, -126.000000)' fill='%23FFFFFF'%3E%3Crect x='249' y='126' width='1' height='1'%3E%3C/rect%3E%3C/g%3E%3C/g%3E%3C/svg%3E)

然而，我们知道时间是没有尽头的，因此这条一维时间轴的长度是无穷大的. 倘若我们想要建立一个数据结构去表达这条由一系列刻度聚合形成的时间轴，这个数据结构所占用的空间也是无穷无尽的.

那么，我们应该如何优化这个问题呢？此时，大家不妨低头看一眼自己的手表. 手表或时钟这类日常生活中用来关联表达时间的工具，采用的是首尾衔接的环状结构来替代无穷长度的一维时间轴，每当钟表划过一圈，刻度从新回到零值，但是已有的时间进度会传承往下.

本期所聊的时间轮算法采用的思路正是与之类似，下面我们梳理一下核心流程：

- • 建立一个环状数据结构
- • 每个刻度对应一个时间范围
- • 创建定时任务时，根据距今的相对时长，推算出需要向后推移的刻度值
- • 倘若来到环形数组的结尾，则重新从起点开始计算，但是记录时把执行轮次数加1
- • 一个刻度可能存在多笔定时任务，所以每个刻度需要挂载一个定时任务链表

接下来，我们建立时间轮的扫描机制，就如同钟表中的指针一般，按照固定的时间节奏，沿着环形数组周而复始地持续向下扫描. 每当来到一个刻度时，则取出链表中轮次为 0 的定时任务进行执行. 这就是时间轮算法的核心思路.

![图片](data:image/svg+xml,%3C%3Fxml version='1.0' encoding='UTF-8'%3F%3E%3Csvg width='1px' height='1px' viewBox='0 0 1 1' version='1.1' xmlns='http://www.w3.org/2000/svg' xmlns:xlink='http://www.w3.org/1999/xlink'%3E%3Ctitle%3E%3C/title%3E%3Cg stroke='none' stroke-width='1' fill='none' fill-rule='evenodd' fill-opacity='0'%3E%3Cg transform='translate(-249.000000, -126.000000)' fill='%23FFFFFF'%3E%3Crect x='249' y='126' width='1' height='1'%3E%3C/rect%3E%3C/g%3E%3C/g%3E%3C/svg%3E)

## 1.2 多级时间轮

接下来聊一聊时间轮中的等级制度与多级时间轮的概念.

首先捋一捋，时间轮每个周期轮次中，使用的数据结构容量与所表达的时间范围之间的关系.

我们把时间轮中的每个刻度记为一个 slot，每个 slot 表示的时间范围记为 t.

假设时间轮中总共包含 2m 个 slot，求问如何组织我们的时间轮数据结构，能够使得时间轮每个轮次对应表达的时间范围尽可能的长. （一个轮次对应的时间范围越长，在时间流逝过程中轮次的迭代速度就越慢，于是每个 slot 对应的定时任务链表长度就越短，执行定时任务时的检索效率就越高.）

这里最简单的方式就是进行采用一维纵向排列的方式，那么能够表达的时间范围就是 2m \* t，某个刻度对应的时间值就记为 {slot\_i}.

![图片](data:image/svg+xml,%3C%3Fxml version='1.0' encoding='UTF-8'%3F%3E%3Csvg width='1px' height='1px' viewBox='0 0 1 1' version='1.1' xmlns='http://www.w3.org/2000/svg' xmlns:xlink='http://www.w3.org/1999/xlink'%3E%3Ctitle%3E%3C/title%3E%3Cg stroke='none' stroke-width='1' fill='none' fill-rule='evenodd' fill-opacity='0'%3E%3Cg transform='translate(-249.000000, -126.000000)' fill='%23FFFFFF'%3E%3Crect x='249' y='126' width='1' height='1'%3E%3C/rect%3E%3C/g%3E%3C/g%3E%3C/svg%3E)

另一种思路是，我们在时间轮中建立一种等级秩序.

比如我们将 2m 个 slot 拆成两个等级——level1 和 level2. 最终我们通过 {level1\_slot}\_{level2\_slot} 的方式进行时间的表达.

我们给 level2 分配 m 个 slot，其中每个 slot 对应的时间范围同样为 t. 而 level1 同样也分配 m 个 slot，但是此时其中每个 slot 对应的时间范围应该为 m \* t，因为在 level1 中的 slot 确定时，level2 中还有 m 种 slot 的组合方式.

如此一来，这种组织方式下，时间轮单个轮次所能表达的时间范围就是 m \* m \* t.

这里探讨的核心不是具体某级时间轮的时间范围结果，而是抛出了一种多级时间轮的思路，从一到二是质变，从二到三、从三到四就仅仅是量变的问题，可以继续复刻相同的思路.

![图片](data:image/svg+xml,%3C%3Fxml version='1.0' encoding='UTF-8'%3F%3E%3Csvg width='1px' height='1px' viewBox='0 0 1 1' version='1.1' xmlns='http://www.w3.org/2000/svg' xmlns:xlink='http://www.w3.org/1999/xlink'%3E%3Ctitle%3E%3C/title%3E%3Cg stroke='none' stroke-width='1' fill='none' fill-rule='evenodd' fill-opacity='0'%3E%3Cg transform='translate(-249.000000, -126.000000)' fill='%23FFFFFF'%3E%3Crect x='249' y='126' width='1' height='1'%3E%3C/rect%3E%3C/g%3E%3C/g%3E%3C/svg%3E)

回过头来看，我们会发现日常使用的时间表达式正是采用了这样一种多级时间轮的等级制度，比如当前的时刻为：2023-09-23 15:50:00. 这本质上是一种通过 {year}-{month}-{date}-{hour}-{minute}-{second} 组成的 6 级时间轮等级结构.

后续在本文第 3 章探讨如何基于 redis zset 实现时间轮的话题中，我们会进一步利用这种多级时间轮的思路，将每个任务首先基于前 5 级 {year}-{month}-{date}-{hour}-{minute} 的等级表达式进行分钟级时间片的纵向拆分，最终在 1 分钟范围内进行定时任务的有序组织，保证在每次插入、删除和检索任务时，处理的数据量级能够维持在分钟级的数量，最大化地提高时间轮结果的处理性能.

## 2 单机版实现

聊完原理部分，下面我们一起进入实战环节.

在本章中，我们会使用 golang 标准库的定时器工具 time ticker 结合环状数组的设计思路，实现一个单机版的单级时间轮.

## 2.1 核心类

### 2.1.1 时间轮

在对时间轮的类定义中，核心字段如下图所示：

![图片](data:image/svg+xml,%3C%3Fxml version='1.0' encoding='UTF-8'%3F%3E%3Csvg width='1px' height='1px' viewBox='0 0 1 1' version='1.1' xmlns='http://www.w3.org/2000/svg' xmlns:xlink='http://www.w3.org/1999/xlink'%3E%3Ctitle%3E%3C/title%3E%3Cg stroke='none' stroke-width='1' fill='none' fill-rule='evenodd' fill-opacity='0'%3E%3Cg transform='translate(-249.000000, -126.000000)' fill='%23FFFFFF'%3E%3Crect x='249' y='126' width='1' height='1'%3E%3C/rect%3E%3C/g%3E%3C/g%3E%3C/svg%3E)

在几个核心字段中：

- • slots——类似于时钟的表盘
- • curSlot——类似于时钟的指针
- • ticker 是使用 golang 标准库的定时器工具，类似于驱动指针运转的齿轮

在创建时间轮实例时，会通过一个异步的常驻 goroutine 执行定时任务的检索、添加、删除等操作，并通过几个 channel 进行 goroutine 的执行逻辑和生命周期的控制：

- • stopc：用于停止 goroutine
- • addTaskCh：用于接收创建定时器指令
- • removeTaskCh：用于接收删除定时任务的指令
```
// 单机版时间轮
type TimeWheel struct {
    // 单例工具，保证时间轮停止操作只能执行一次
    sync.Once
    // 时间轮运行时间间隔
    interval     time.Duration
    // 时间轮定时器
    ticker       *time.Ticker
    // 停止时间轮的 channel
    stopc        chan struct{}
    // 新增定时任务的入口 channel  
    addTaskCh    chan *taskElement
    // 删除定时任务的入口 channel
    removeTaskCh chan string
    // 通过 list 组成的环状数组. 通过遍历环状数组的方式实现时间轮
    // 定时任务数量较大，每个 slot 槽内可能存在多个定时任务，因此通过 list 进行组装
    slots        []*list.List
    // 当前遍历到的环状数组的索引
    curSlot      int
    // 定时任务 key 到任务节点的映射，便于在 list 中删除任务节点
    keyToETask   map[string]*list.Element
}
```

此处有几个技术细节需要提及：

**首先：** 所谓环状数组指的是逻辑意义上的. 在实际的实现过程中，会通过一个定长数组结合循环遍历的方式，来实现这个逻辑意义上的“环状”性质.

**其次：** 数组每一轮能表达的时间范围是固定的. 每当在添加添加一个定时任务时，需要根据其延迟的相对时长推算出其所处的 slot 位置，其中可能跨遍历轮次的情况，这时候需要额外通过定时任务中的 cycle 字段来记录这一信息，避免定时任务被提前执行.

**最后：** 时间轮中一个 slot 可能需要挂载多笔定时任务，因此针对每个 slot，需要采用 golang 标准库 container/list 中实现的双向链表进行定时任务数据的存储.

![图片](data:image/svg+xml,%3C%3Fxml version='1.0' encoding='UTF-8'%3F%3E%3Csvg width='1px' height='1px' viewBox='0 0 1 1' version='1.1' xmlns='http://www.w3.org/2000/svg' xmlns:xlink='http://www.w3.org/1999/xlink'%3E%3Ctitle%3E%3C/title%3E%3Cg stroke='none' stroke-width='1' fill='none' fill-rule='evenodd' fill-opacity='0'%3E%3Cg transform='translate(-249.000000, -126.000000)' fill='%23FFFFFF'%3E%3Crect x='249' y='126' width='1' height='1'%3E%3C/rect%3E%3C/g%3E%3C/g%3E%3C/svg%3E)

### 2.1.2 定时任务

下面是对一笔定时任务的类定义：

- • key：每个定时任务的全局唯一标识键
- • task：包含了定时任务执行逻辑的闭包函数
- • pos：定时任务在环形数组所处的位置，即数组的索引 index
- • cycle：定时任务的延迟轮次. 时间轮的 curSlot 指针每完成一整轮的数组遍历，所有定时任务的 cycle 指数都需要减 1. 当定时任务 cycle 指数为 0 时，代表该任务在当前遍历轮次执行.
```
// 封装了一笔定时任务的明细信息
type taskElement struct {
    // 内聚了定时任务执行逻辑的闭包函数
    task  func()
    // 定时任务挂载在环状数组中的索引位置
    pos   int
    // 定时任务的延迟轮次. 指的是 curSlot 指针还要扫描过环状数组多少轮，才满足执行该任务的条件
    cycle int
    // 定时任务的唯一标识键
    key   string
}
```

## 2.2 构造器

在创建时间轮的构造器函数中，需要传入两个入参：

- • slotNum：由使用方指定 slot 的个数，默认为 10
- • interval：由使用方指定每个 slot 对应的时间范围，默认为 1 秒

初始化时间轮实例的过程中，会完成定时器 ticker 以及各个 channel 的初始化，并针对数组 中的各个 slot 进行初始化，每个 slot 位置都需要填充一个 list.

每个时间轮实例都会异步调用 run 方法，启动一个常驻 goroutine 用于接收和处理定时任务.

```
// 创建单机版时间轮 slotNum——时间轮环状数组长度  interval——扫描时间间隔
func NewTimeWheel(slotNum int, interval time.Duration) *TimeWheel {
    // 环状数组长度默认为 10
    if slotNum <= 0 {
        slotNum = 10
    }
    // 扫描时间间隔默认为 1 秒
    if interval <= 0 {
        interval = time.Second
    }

    // 初始化时间轮实例
    t := TimeWheel{
        interval:     interval,
        ticker:       time.NewTicker(interval),
        stopc:        make(chan struct{}),
        keyToETask:   make(map[string]*list.Element),
        slots:        make([]*list.List, 0, slotNum),
        addTaskCh:    make(chan *taskElement),
        removeTaskCh: make(chan string),
    }
    for i := 0; i < slotNum; i++ {
        t.slots = append(t.slots, list.New())
    }
    
    // 异步启动时间轮常驻 goroutine
    go t.run()
    return &t
}
```

## 2.3 启动与停止

时间轮运行的核心逻辑位于 timeWheel.run 方法中，该方法会通过 for 循环结合 select 多路复用的方式运行，属于 golang 中非常常见的异步编程风格.

goroutine 运行过程中需要从以下四类 channel 中接收不同的信号，并进行逻辑的分发处理：

- • stopc：停止时间轮，使得当前 goroutine 退出
- • ticker：接收到 ticker 的信号说明时间由往前推进了一个 interval，则需要批量检索并执行当前 slot 中的定时任务. 并推进指针 curSlot 往前偏移
- • addTaskCh：接收创建定时任务的指令
- • removeTaskCh：接收删除定时任务的指令

此处值得一提的是，后续不论是创建、删除还是检索定时任务，都是通过这个常驻 goroutine 完成的，因此在访问一些临界资源的时候，不需要加锁，因为不存在并发访问的情况

![图片](data:image/svg+xml,%3C%3Fxml version='1.0' encoding='UTF-8'%3F%3E%3Csvg width='1px' height='1px' viewBox='0 0 1 1' version='1.1' xmlns='http://www.w3.org/2000/svg' xmlns:xlink='http://www.w3.org/1999/xlink'%3E%3Ctitle%3E%3C/title%3E%3Cg stroke='none' stroke-width='1' fill='none' fill-rule='evenodd' fill-opacity='0'%3E%3Cg transform='translate(-249.000000, -126.000000)' fill='%23FFFFFF'%3E%3Crect x='249' y='126' width='1' height='1'%3E%3C/rect%3E%3C/g%3E%3C/g%3E%3C/svg%3E)

```
// 运行时间轮
func (t *TimeWheel) run() {
    defer func() {
        if err := recover(); err != nil {
            // ...
        }
    }()
 
    // 通过 for + select 的代码结构运行一个常驻 goroutine 是常规操作
    for {
        select {
        // 停止时间轮
        case <-t.stopc:
            return
        // 接收到定时信号
        case <-t.ticker.C:
            // 批量执行定时任务
            t.tick()
        // 接收创建定时任务的信号
        case task := <-t.addTaskCh:
            t.addTask(task)
        // 接收到删除定时任务的信号
        case removeKey := <-t.removeTaskCh:
            t.removeTask(removeKey)
        }
    }
}
```

时间轮提供了一个 Stop 方法，用于手动停止时间轮，回收对应的 goroutine 和 ticker 资源.

停止时间轮的操作是通过关闭 stopc channel 完成的，由于 channel 不允许被反复关闭，因此这里通过 sync.Once 保证该逻辑只被调用一次.

```
// 停止时间轮
func (t *TimeWheel) Stop() {
    // 通过单例工具，保证 channel 只能被关闭一次，避免 panic
    t.Do(func() {
        // 定制定时器 ticker
        t.ticker.Stop()
        // 关闭定时器运行的 stopc
        close(t.stopc)
    })
}
```

## 2.4 创建任务

创建一笔定时任务的核心步骤如下：

- • 使用方往 addTaskCh 中投递定时任务，由常驻 goroutine 接收定时任务
- • 根据执行时间，推算出定时任务所处的 slot 位置以及需要延迟的轮次 cycle
- • 将定时任务包装成一个 list node，追加到对应 slot 位置的 list 尾部
- • 以定时任务唯一键为 key，list node 为 value，在 keyToETask map 中建立映射关系，方便后续删除任务时使用

![图片](data:image/svg+xml,%3C%3Fxml version='1.0' encoding='UTF-8'%3F%3E%3Csvg width='1px' height='1px' viewBox='0 0 1 1' version='1.1' xmlns='http://www.w3.org/2000/svg' xmlns:xlink='http://www.w3.org/1999/xlink'%3E%3Ctitle%3E%3C/title%3E%3Cg stroke='none' stroke-width='1' fill='none' fill-rule='evenodd' fill-opacity='0'%3E%3Cg transform='translate(-249.000000, -126.000000)' fill='%23FFFFFF'%3E%3Crect x='249' y='126' width='1' height='1'%3E%3C/rect%3E%3C/g%3E%3C/g%3E%3C/svg%3E)

```
// 添加定时任务到时间轮中
func (t *TimeWheel) AddTask(key string, task func(), executeAt time.Time) {
    // 根据执行时间推算得到定时任务从属的 slot 位置，以及需要延迟的轮次
    pos, cycle := t.getPosAndCircle(executeAt)
    // 将定时任务通过 channel 进行投递
    t.addTaskCh <- &taskElement{
        pos:   pos,
        cycle: cycle,
        task:  task,
        key:   key,
    }
}
```
```
// 根据执行时间推算得到定时任务从属的 slot 位置，以及需要延迟的轮次
func (t *TimeWheel) getPosAndCircle(executeAt time.Time) (int, int) {
    delay := int(time.Until(executeAt))
    // 定时任务的延迟轮次
    cycle := delay / (len(t.slots) * int(t.interval))
    // 定时任务从属的环状数组 index
    pos := (t.curSlot + delay/int(t.interval)) % len(t.slots)
    return pos, cycle
}
```
```
// 常驻 goroutine 接收到创建定时任务后的处理逻辑
func (t *TimeWheel) addTask(task *taskElement) {
    // 获取到定时任务从属的环状数组 index 以及对应的 list
    list := t.slots[task.pos]
    // 倘若定时任务 key 之前已存在，则需要先删除定时任务
    if _, ok := t.keyToETask[task.key]; ok {
        t.removeTask(task.key)
    }
    // 将定时任务追加到 list 尾部
    eTask := list.PushBack(task)
    // 建立定时任务 key 到将定时任务所处的节点
    t.keyToETask[task.key] = eTask
}
```

## 2.5 删除任务

删除一笔定时任务的核心步骤如下：

- • 使用方往 removeTaskCh 中投递删除任务的 key，由常驻 goroutine 接收处理
- • 从 keyToETask map 中，找到该任务对应的 list node
- • 从 keyToETask map 中移除该组 kv 对
- • 从对应 slot 的 list 中移除该 list node

![图片](data:image/svg+xml,%3C%3Fxml version='1.0' encoding='UTF-8'%3F%3E%3Csvg width='1px' height='1px' viewBox='0 0 1 1' version='1.1' xmlns='http://www.w3.org/2000/svg' xmlns:xlink='http://www.w3.org/1999/xlink'%3E%3Ctitle%3E%3C/title%3E%3Cg stroke='none' stroke-width='1' fill='none' fill-rule='evenodd' fill-opacity='0'%3E%3Cg transform='translate(-249.000000, -126.000000)' fill='%23FFFFFF'%3E%3Crect x='249' y='126' width='1' height='1'%3E%3C/rect%3E%3C/g%3E%3C/g%3E%3C/svg%3E)

```
// 删除定时任务，投递信号
func (t *TimeWheel) RemoveTask(key string) {
    t.removeTaskCh <- key
}
```
```
// 时间轮常驻 goroutine 接收到删除任务信号后，执行的删除任务逻辑
func (t *TimeWheel) removeTask(key string) {
    eTask, ok := t.keyToETask[key]
    if !ok {
        return
    }
    // 将定时任务节点从映射 map 中移除
    delete(t.keyToETask, key)
    // 获取到定时任务节点后，将其从 list 中移除
    task, _ := eTask.Value.(*taskElement)
    _ = t.slots[task.pos].Remove(eTask)
}
```

## 2.6 执行定时任务

最后来捋一下最核心的链路——检索并批量执行定时任务的流程.

首先，每当接收到 ticker 信号时，会根据当前的 curSlot 指针，获取到对应 slot 位置挂载的定时任务 list，调用 execute 方法执行其中的定时任务. 最后通过 circularIncr 方法推进 curSlot 指针向前移动.

```
// 常驻 goroutine 每次接收到定时信号后用于执行定时任务的逻辑
func (t *TimeWheel) tick() {
    // 根据 curSlot 获取到当前所处的环状数组索引位置，取出对应的 list
    list := t.slots[t.curSlot]
     // 在方法返回前，推进 curSlot 指针的位置，进行环状遍历
    defer t.circularIncr()
    // 批量处理满足执行条件的定时任务
    t.execute(list)
}
```

在 execute 方法中，会对 list 中的定时任务进行遍历：

- • 对于 cycle > 0 的定时任务，说明当前还未达到执行条件，需要将其 cycle 值减 1，留待后续轮次再处理
- • 对于 cycle = 0 的定时任务，开启一个 goroutine ，执行其中的闭包函数 task，并将其从 list 和 map 中移除
```
// 执行定时任务，每次处理一个 list
func (t *TimeWheel) execute(l *list.List) {
    // 遍历 list
    for e := l.Front(); e != nil; {
        // 获取到每个节点对应的定时任务信息
        taskElement, _ := e.Value.(*taskElement)
        // 倘若任务还存在延迟轮次，则只对 cycle 计数器进行扣减，本轮不作任务的执行
        if taskElement.cycle > 0 {
            taskElement.cycle--
            e = e.Next()
            continue
        }

        // 当前节点对应定时任务已达成执行条件，开启一个 goroutine 负责执行任务
        go func() {
            defer func() {
                if err := recover(); err != nil {
                    // ...
                }
            }()
            taskElement.task()
        }()

        // 任务已执行，需要把对应的任务节点从 list 中删除
        next := e.Next()
        l.Remove(e)
        // 把任务 key 从映射 map 中删除
        delete(t.keyToETask, taskElement.key)
        e = next
    }
}
```

在 circularIncr 方法中，呼应了环状数组的逻辑处理方式：

![图片](data:image/svg+xml,%3C%3Fxml version='1.0' encoding='UTF-8'%3F%3E%3Csvg width='1px' height='1px' viewBox='0 0 1 1' version='1.1' xmlns='http://www.w3.org/2000/svg' xmlns:xlink='http://www.w3.org/1999/xlink'%3E%3Ctitle%3E%3C/title%3E%3Cg stroke='none' stroke-width='1' fill='none' fill-rule='evenodd' fill-opacity='0'%3E%3Cg transform='translate(-249.000000, -126.000000)' fill='%23FFFFFF'%3E%3Crect x='249' y='126' width='1' height='1'%3E%3C/rect%3E%3C/g%3E%3C/g%3E%3C/svg%3E)

```
// 每次 tick 后需要推进 curSlot 指针的位置，slots 在逻辑意义上是环状数组，所以在到达尾部时需要从新回到头部 
func (t *TimeWheel) circularIncr() {
    t.curSlot = (t.curSlot + 1) % len(t.slots)
}
```

## 3 分布式版实现

本章我们讨论一下，如何基于 redis 实现分布式版本的时间轮，以贴合实际生产环境对分布式定时任务调度系统的诉求.

redis 版时间轮的实现思路是使用 redis 中的有序集合 sorted set（简称 zset） 进行定时任务的存储管理，其中以每个定时任务执行时间对应的时间戳作为 zset 中的 score，完成定时任务的有序排列组合.

zset 数据结构的 redis 官方文档链接：https://redis.io/docs/data-types/sorted-sets/

![图片](data:image/svg+xml,%3C%3Fxml version='1.0' encoding='UTF-8'%3F%3E%3Csvg width='1px' height='1px' viewBox='0 0 1 1' version='1.1' xmlns='http://www.w3.org/2000/svg' xmlns:xlink='http://www.w3.org/1999/xlink'%3E%3Ctitle%3E%3C/title%3E%3Cg stroke='none' stroke-width='1' fill='none' fill-rule='evenodd' fill-opacity='0'%3E%3Cg transform='translate(-249.000000, -126.000000)' fill='%23FFFFFF'%3E%3Crect x='249' y='126' width='1' height='1'%3E%3C/rect%3E%3C/g%3E%3C/g%3E%3C/svg%3E)

这里有两个的技术细节需要提前和大家同步：

- • **分钟级时间分片：** 为了避免产生 redis 大 key 问题，此处采用本文 1.2 小节中提到的多级时间轮等级制度，以分钟的维度进行时间片的纵向划分，每个分钟级时间片对应一个独立的 zset 有序表，保证每次执行任务时处理的数据规模仅为分钟的量级
- • **惰性删除机制：** 为了简化删除定时任务的流程. 在使用方指定删除定时任务时，我们不直接从 zset 中删除数据，而是额外记录一个已删除任务的 set. 后续在检索定时任务时，通过使用 set 进行定时任务的过滤，实现定时任务的惰性删除.

![图片](data:image/svg+xml,%3C%3Fxml version='1.0' encoding='UTF-8'%3F%3E%3Csvg width='1px' height='1px' viewBox='0 0 1 1' version='1.1' xmlns='http://www.w3.org/2000/svg' xmlns:xlink='http://www.w3.org/1999/xlink'%3E%3Ctitle%3E%3C/title%3E%3Cg stroke='none' stroke-width='1' fill='none' fill-rule='evenodd' fill-opacity='0'%3E%3Cg transform='translate(-249.000000, -126.000000)' fill='%23FFFFFF'%3E%3Crect x='249' y='126' width='1' height='1'%3E%3C/rect%3E%3C/g%3E%3C/g%3E%3C/svg%3E)

## 3.1 核心类

### 3.1.1 redis 时间轮

在 redis 版时间轮中有两个核心类，第一个是关于时间轮的类定义：

- • redisClient：定时任务的存储是基于 redis zset 实现的，因此需要内置一个 redis 客户端，这部分在 3.2 小节展开；
- • httpClient：定时任务执行时，是通过请求使用方预留回调地址的方式实现的，因此需要内置一个 http 客户端
- • channel × 2：ticker 和 stopc 对应为 golang 标准库定时器以及停止 goroutine 的控制器
```
// 基于 redis 实现的分布式版时间轮
type RTimeWheel struct {
    // 内置的单例工具，用于保证 stopc 只被关闭一次
    sync.Once
    // redis 客户端
    redisClient *redis.Client
    // http 客户端. 在执行定时任务时需要使用到.
    httpClient  *thttp.Client
    // 用于停止时间轮的控制器 channel
    stopc       chan struct{
    // 触发定时扫描任务的定时器 
    ticker      *time.Ticker
}
```

### 3.1.2 定时任务

定时任务的类型定义如下，其中包括定时任务的唯一键 key，以及执行定时任务回调时需要使用到的 http 协议参数.

```
// 使用方提交的每一笔定时任务
type RTaskElement struct {
    // 定时任务全局唯一 key
    Key         string            \`json:"key"\`
    // 定时任务执行时，回调的 http url
    CallbackURL string            \`json:"callback_url"\`
    // 回调时使用的 http 方法
    Method      string            \`json:"method"\`
    // 回调时传递的请求参数
    Req         interface{}       \`json:"req"\`
    // 回调时使用的 http 请求头
    Header      map[string]string \`json:"header"\`
}
```

## 3.2 redis lua 使用事项

本项目使用的 redis 客户端是我个人基于 golang-redis 客户端 sdk——redigo 进一步封装实现的，redigo 的开源地址为： https://github.com/gomodule/redigo

此处主要在 redisClient 中封装了一个 Eval 方法，便于使用 redis lua 脚本执行复合指令.

```
// Eval：执行 lua 脚本.
// src——lua 脚本内容 keyCount——key 的数量 keysAndArgs——由 key 和 args 组成的列表
func (c *Client) Eval(ctx context.Context, src string, keyCount int, keysAndArgs []interface{}) (interface{}, error) {
    args := make([]interface{}, 2+len(keysAndArgs))
    args[0] = src
    args[1] = keyCount
    copy(args[2:], keysAndArgs)

    // 从 redis 链接池中获取一个连接
    conn, err := c.pool.GetContext(ctx)
    if err != nil {
        return -1, err
    }
    // 使用完成后将连接放回池子
    defer conn.Close()
    // 执行 lua 脚本
    return conn.Do("EVAL", args...)
}
```

lua 脚本是 redis 的高级功能，能够保证针在单个 redis 节点内执行的一系列指令具备原子性，中途不会被其他操作者打断.

redis lua 功能介绍：https://redis.io/docs/interact/programmability/eval-intro/

lua 语法教程：https://www.runoob.com/lua/lua-tutorial.html

此处之所以需要使用 lua 脚本，是因为在实现时间轮的过程中，存在一系列本身不具备原子性但在业务流程中不可拆解的复合操作，需要由 lua 脚本赋予其原子性质.

在使用 lua 时，尤其需要注意的点是，只有操作的数据属于单个 redis 节点时，才能保证其原子性. 然而在生产环境中，redis 通常采用纵向分治的集群模式，这使得 key 不同的数据可能被分发在不同的 redis 节点上，此时 lua 脚本的性质就无法保证.

![图片](data:image/svg+xml,%3C%3Fxml version='1.0' encoding='UTF-8'%3F%3E%3Csvg width='1px' height='1px' viewBox='0 0 1 1' version='1.1' xmlns='http://www.w3.org/2000/svg' xmlns:xlink='http://www.w3.org/1999/xlink'%3E%3Ctitle%3E%3C/title%3E%3Cg stroke='none' stroke-width='1' fill='none' fill-rule='evenodd' fill-opacity='0'%3E%3Cg transform='translate(-249.000000, -126.000000)' fill='%23FFFFFF'%3E%3Crect x='249' y='126' width='1' height='1'%3E%3C/rect%3E%3C/g%3E%3C/g%3E%3C/svg%3E)

在使用 lua 脚本时，倘若一系列复合操作都是针对于同一个 key，那么数据必然位于同一个节点，没有任何疑议. 倘若我们在 lua 中涉及到对多个 key 的操作，那么这些 key 对应的数据就可能从属于不同的 redis 节点，此时 lua 脚本存在失效的风险.

针对这个问题，本项目采取的是定制的分区策略，来保证指定的 key 一定被分发到相同的 redis 节点上. 此处使用的方式是通过 "{}" 进行 hash\_tag 的标识，所有拥有相同 hash\_tag 的 key 都一定会被分发到相同的节点上.

该分区策略可参见 redis 官方文档：https://redis.io/commands/cluster-keyslot/

指令示例如下：

```
> CLUSTER KEYSLOT somekey
(integer) 11058
> CLUSTER KEYSLOT foo{hash_tag}
(integer) 2515
> CLUSTER KEYSLOT bar{hash_tag}
(integer) 2515
```

## 3.3 构造器

在构造时间轮实例时，使用方需要注入 redis 客户端以及 http 客户端.

在初始化流程中，ticker 为 golang 标准库实现的定时器，定时器的执行时间间隔固定为 1 s. 此外会异步运行 run 方法，启动一个常驻 goroutine，生命周期会通过 stopc channel 进行控制.

```
// 构造 redis 实现的分布式时间轮
func NewRTimeWheel(redisClient *redis.Client, httpClient *thttp.Client) *RTimeWheel {
    // 创建时间轮实例
    r := RTimeWheel{
        // 创建定时器，每隔 1 s 执行一次
        ticker:      time.NewTicker(time.Second),
        redisClient: redisClient,
        httpClient:  httpClient,
        stopc:       make(chan struct{}),
    }

    // 异步启动时间轮
    go r.run()
    return &r
}
```

## 3.4 启动与停止

时间轮常驻 goroutine 运行流程同样通过 for + select 的形式运行：

- • 接收到 stopc 信号时，goroutine 退出，时间轮停止运行
- • 接收到 ticker 信号时，开启一个异步 goroutine 用于执行当前批次的定时任务

![图片](data:image/svg+xml,%3C%3Fxml version='1.0' encoding='UTF-8'%3F%3E%3Csvg width='1px' height='1px' viewBox='0 0 1 1' version='1.1' xmlns='http://www.w3.org/2000/svg' xmlns:xlink='http://www.w3.org/1999/xlink'%3E%3Ctitle%3E%3C/title%3E%3Cg stroke='none' stroke-width='1' fill='none' fill-rule='evenodd' fill-opacity='0'%3E%3Cg transform='translate(-249.000000, -126.000000)' fill='%23FFFFFF'%3E%3Crect x='249' y='126' width='1' height='1'%3E%3C/rect%3E%3C/g%3E%3C/g%3E%3C/svg%3E)

```
// 运行时间轮
func (r *RTimeWheel) run() {
    // 通过 for + select 的代码结构运行一个常驻 goroutine 是常规操作
    for {
        select {
        // 接收到终止信号，则退出 goroutine
        case <-r.stopc:
            return
        // 每次接收到来自定时器的信号，则批量扫描并执行定时任务
        case <-r.ticker.C:
            // 每次 tick 获取任务
            go r.executeTasks()
        }
    }
}
```

停止时间轮的 Stop 方法通过关闭 stopc 保证常驻 goroutine 能够及时退出.

```
// 停止时间轮
func (r *RTimeWheel) Stop() {
    // 基于单例工具，保证 stopc 只能被关闭一次
    r.Do(func() {
        // 关闭 stopc，使得常驻 goroutine 停止运行
        close(r.stopc)
        // 终止定时器 ticker
        r.ticker.Stop()
    })
}
```

## 3.5 创建任务

在创建定时任务时，每笔定时任务需要根据其执行的时间找到从属的分钟时间片.

定时任务真正的存储逻辑定义在一段 lua 脚本中，通过 redis 客户端的 Eval 方法执行.

```
// 添加定时任务
func (r *RTimeWheel) AddTask(ctx context.Context, key string, task *RTaskElement, executeAt time.Time) error {
    // 前置对定时任务的参数进行校验
    if err := r.addTaskPrecheck(task); err != nil {
        return err
    }

    task.Key = key
    // 将定时任务序列化成字节数组
    taskBody, _ := json.Marshal(task)
    // 通过执行 lua 脚本，实现将定时任务添加 redis zset 中. 本质上底层使用的是 zadd 指令.
    _, err := r.redisClient.Eval(ctx, LuaAddTasks, 2, []interface{}{
        // 分钟级 zset 时间片
        r.getMinuteSlice(executeAt),
        // 标识任务删除的集合
        r.getDeleteSetKey(executeAt),
        // 以执行时刻的秒级时间戳作为 zset 中的 score
        executeAt.Unix(),
        // 任务明细
        string(taskBody),
        // 任务 key，用于存放在删除集合中
        key,
    })
    return err
}
```

下面展示的是获取分钟级定时任务有序表 minuteSlice 以及已删除任务集合 deleteSet 的细节.

此处呼应了 3.3 小节，通过以分钟级表达式作为 {hash\_tag} 的方式，确保 minuteSlice 和 deleteSet 一定会分发到相同的 redis 节点之上，进一步保证 lua 脚本的原子性能够生效.

- • 获取定时任务有序表 key 的方法：
```
func (r *RTimeWheel) getMinuteSlice(executeAt time.Time) string {
    return fmt.Sprintf("xiaoxu_timewheel_task_{%s}", util.GetTimeMinuteStr(executeAt))
}
```
- • 获取删除任务集合 key 的方法：
```
func (r *RTimeWheel) getDeleteSetKey(executeAt time.Time) string {
    return fmt.Sprintf("xiaoxu_timewheel_delset_{%s}", util.GetTimeMinuteStr(executeAt))
}
```

下面展示一下创建定时任务流程中 lua 脚本的执行逻辑：

![图片](data:image/svg+xml,%3C%3Fxml version='1.0' encoding='UTF-8'%3F%3E%3Csvg width='1px' height='1px' viewBox='0 0 1 1' version='1.1' xmlns='http://www.w3.org/2000/svg' xmlns:xlink='http://www.w3.org/1999/xlink'%3E%3Ctitle%3E%3C/title%3E%3Cg stroke='none' stroke-width='1' fill='none' fill-rule='evenodd' fill-opacity='0'%3E%3Cg transform='translate(-249.000000, -126.000000)' fill='%23FFFFFF'%3E%3Crect x='249' y='126' width='1' height='1'%3E%3C/rect%3E%3C/g%3E%3C/g%3E%3C/svg%3E)

```
const (
    // 添加任务时，如果存在删除 key 的标识，则将其删除
    // 添加任务时，根据时间（所属的 min）决定数据从属于哪个分片{}
    LuaAddTasks = \`
       -- 获取的首个 key 为 zset 的 key
       local zsetKey = KEYS[1]
       -- 获取的第二个 key 为标识已删除任务 set 的 key
       local deleteSetKey = KEYS[2]
       -- 获取的第一个 arg 为定时任务在 zset 中的 score
       local score = ARGV[1]
       -- 获取的第二个 arg 为定时任务明细数据
       local task = ARGV[2]
       -- 获取的第三个 arg 为定时任务唯一键，用于将其从已删除任务 set 中移除
       local taskKey = ARGV[3]
       -- 每次添加定时任务时，都直接将其从已删除任务 set 中移除，不管之前是否在 set 中
       redis.call('srem',deleteSetKey,taskKey)
       -- 调用 zadd 指令，将定时任务添加到 zset 中
       return redis.call('zadd',zsetKey,score,task)
    \`
)
```

## 3.6 删除任务

删除定时任务的方式是将定时任务追加到分钟级的已删除任务 set 中. 之后在检索定时任务时，会根据这个 set 对定时任务进行过滤，实现惰性删除机制.

```
// 从 redis 时间轮中删除一个定时任务
func (r *RTimeWheel) RemoveTask(ctx context.Context, key string, executeAt time.Time) error {
    // 执行 lua 脚本，将被删除的任务追加到 set 中.
    _, err := r.redisClient.Eval(ctx, LuaDeleteTask, 1, []interface{}{
        r.getDeleteSetKey(executeAt),
        key,
    })
    return err
}
```

lua 执行逻辑如下：

![图片](data:image/svg+xml,%3C%3Fxml version='1.0' encoding='UTF-8'%3F%3E%3Csvg width='1px' height='1px' viewBox='0 0 1 1' version='1.1' xmlns='http://www.w3.org/2000/svg' xmlns:xlink='http://www.w3.org/1999/xlink'%3E%3Ctitle%3E%3C/title%3E%3Cg stroke='none' stroke-width='1' fill='none' fill-rule='evenodd' fill-opacity='0'%3E%3Cg transform='translate(-249.000000, -126.000000)' fill='%23FFFFFF'%3E%3Crect x='249' y='126' width='1' height='1'%3E%3C/rect%3E%3C/g%3E%3C/g%3E%3C/svg%3E)

```
const(    
    // 删除定时任务 lua 脚本
    LuaDeleteTask = \`
       -- 获取标识删除任务的 set 集合的 key
       local deleteSetKey = KEYS[1]
       -- 获取定时任务的唯一键
       local taskKey = ARGV[1]
       -- 将定时任务唯一键添加到 set 中
       redis.call('sadd',deleteSetKey,taskKey)
       -- 倘若是 set 中的首个元素，则对 set 设置 120 s 的过期时间
       local scnt = redis.call('scard',deleteSetKey)
       if (tonumber(scnt) == 1)
       then
           redis.call('expire',deleteSetKey,120)
       end
       return scnt
)    \`
```

## 3.7 执行定时任务

在执行定时任务时，会通过 getExecutableTasks 方法批量获取到满足执行条件的定时任务 list，然后并发调用 execute 方法完成定时任务的回调执行.

```
// 批量执行定时任务
func (r *RTimeWheel) executeTasks() {
    defer func() {
        if err := recover(); err != nil {
            // log
        }
    }()

    // 并发控制，保证 30 s 之内完成该批次全量任务的执行，及时回收 goroutine，避免发生 goroutine 泄漏
    tctx, cancel := context.WithTimeout(context.Background(), time.Second*30)
    defer cancel()
    // 根据当前时间条件扫描 redis zset，获取所有满足执行条件的定时任务
    tasks, err := r.getExecutableTasks(tctx)
    if err != nil {
        // log
        return
    }

    // 并发执行任务，通过 waitGroup 进行聚合收口
    var wg sync.WaitGroup
    for _, task := range tasks {
        wg.Add(1)
        // shadow
        task := task
        go func() {
            defer func() {
                if err := recover(); err != nil {
                }
                wg.Done()
            }()
            // 执行定时任务
            if err := r.executeTask(tctx, task); err != nil {
                // log
            }
        }()
    }
    wg.Wait()
}
```

## 3.8 检索定时任务

最后介绍一下，如何根据当前时间获取到满足执行条件的定时任务列表：

- • 每次检索时，首先根据当前时刻，推算出所从属的分钟级时间片
- • 然后获得当前的秒级时间戳，作为 zrange 指令检索的 score 范围
- • 调用 lua 脚本，同时获取到已删除任务 set 以及 score 范围内的定时任务 list.
- • 通过 set 过滤掉被删除的任务，然后返回满足执行条件的定时任务
```
func (r *RTimeWheel) getExecutableTasks(ctx context.Context) ([]*RTaskElement, error) {
    now := time.Now()
    // 根据当前时间，推算出其从属的分钟级时间片
    minuteSlice := r.getMinuteSlice(now)
    // 推算出其对应的分钟级已删除任务集合
    deleteSetKey := r.getDeleteSetKey(now)
    nowSecond := util.GetTimeSecond(now)
    // 以秒级时间戳作为 score 进行 zset 检索
    score1 := nowSecond.Unix()
    score2 := nowSecond.Add(time.Second).Unix()
    // 执行 lua 脚本，本质上是通过 zrange 指令结合秒级时间戳对应的 score 进行定时任务检索
    rawReply, err := r.redisClient.Eval(ctx, LuaZrangeTasks, 2, []interface{}{
        minuteSlice, deleteSetKey, score1, score2,
    })
    if err != nil {
        return nil, err
    }

    // 结果中，首个元素对应为已删除任务的 key 集合，后续元素对应为各笔定时任务
    replies := gocast.ToInterfaceSlice(rawReply)
    if len(replies) == 0 {
        return nil, fmt.Errorf("invalid replies: %v", replies)
    }

    deleteds := gocast.ToStringSlice(replies[0])
    deletedSet := make(map[string]struct{}, len(deleteds))
    for _, deleted := range deleteds {
        deletedSet[deleted] = struct{}{}
    }

    // 遍历各笔定时任务，倘若其存在于删除集合中，则跳过，否则追加到 list 中返回，用于后续执行
    tasks := make([]*RTaskElement, 0, len(replies)-1)
    for i := 1; i < len(replies); i++ {
        var task RTaskElement
        if err := json.Unmarshal([]byte(gocast.ToString(replies[i])), &task); err != nil {
            // log
            continue
        }

        if _, ok := deletedSet[task.Key]; ok {
            continue
        }
        tasks = append(tasks, &task)
    }

    return tasks, nil
}
```

lua 脚本的执行逻辑如下：

![图片](data:image/svg+xml,%3C%3Fxml version='1.0' encoding='UTF-8'%3F%3E%3Csvg width='1px' height='1px' viewBox='0 0 1 1' version='1.1' xmlns='http://www.w3.org/2000/svg' xmlns:xlink='http://www.w3.org/1999/xlink'%3E%3Ctitle%3E%3C/title%3E%3Cg stroke='none' stroke-width='1' fill='none' fill-rule='evenodd' fill-opacity='0'%3E%3Cg transform='translate(-249.000000, -126.000000)' fill='%23FFFFFF'%3E%3Crect x='249' y='126' width='1' height='1'%3E%3C/rect%3E%3C/g%3E%3C/g%3E%3C/svg%3E)

```
(    
    // 扫描 redis 时间轮. 获取分钟范围内,已删除任务集合 以及在时间上达到执行条件的定时任务进行返回
    LuaZrangeTasks = \`
       -- 第一个 key 为存储定时任务的 zset key
       local zsetKey = KEYS[1]
       -- 第二个 key 为已删除任务 set 的 key
       local deleteSetKey = KEYS[2]
       -- 第一个 arg 为 zrange 检索的 score 左边界
       local score1 = ARGV[1]
       -- 第二个 arg 为 zrange 检索的 score 右边界
       local score2 = ARGV[2]
       -- 获取到已删除任务的集合
       local deleteSet = redis.call('smembers',deleteSetKey)
       -- 根据秒级时间戳对 zset 进行 zrange 检索，获取到满足时间条件的定时任务
       local targets = redis.call('zrange',zsetKey,score1,score2,'byscore')
       -- 检索到的定时任务直接从时间轮中移除，保证分布式场景下定时任务不被重复获取
       redis.call('zremrangebyscore',zsetKey,score1,score2)
       -- 返回的结果是一个 table
       local reply = {}
       -- table 的首个元素为已删除任务集合
       reply[1] = deleteSet
       -- 依次将检索到的定时任务追加到 table 中
       for i, v in ipairs(targets) do
           reply[#reply+1]=v
       end
       return reply
    \`
)
```

## 4 总结

本期和大家探讨了如何基于 golang 从零到一实现时间轮算法，通过原理结合源码，详细展示了单机版和 redis 分布式版时间轮的实现方式.

最后我们展望一下，时间轮算法在工程实践中具体能得到怎样的应用呢？

此前我基于 redis zset 时间轮的模型实现了一个分布式定时任务调度系统 xtimer，大家感兴趣的话可以阅读一下我之前分享的这篇文章： [基于协程池架构实现的分布式定时器 XTimer](http://mp.weixin.qq.com/s?__biz=MzkxMjQzMjA0OQ==&mid=2247483769&idx=1&sn=38d4a0c15e0e7a9f74a58e43f293e8a2&chksm=c10c4fa7f67bc6b1b08602e328c7ac506e4dc4c942427c730ecbfe3f18f0d1e268b1aa1e0d32&scene=21#wechat_redirect)

该项目源码已于 github 开源：http://github.com/xiaoxuxiansheng/xtimer

![图片](data:image/svg+xml,%3C%3Fxml version='1.0' encoding='UTF-8'%3F%3E%3Csvg width='1px' height='1px' viewBox='0 0 1 1' version='1.1' xmlns='http://www.w3.org/2000/svg' xmlns:xlink='http://www.w3.org/1999/xlink'%3E%3Ctitle%3E%3C/title%3E%3Cg stroke='none' stroke-width='1' fill='none' fill-rule='evenodd' fill-opacity='0'%3E%3Cg transform='translate(-249.000000, -126.000000)' fill='%23FFFFFF'%3E%3Crect x='249' y='126' width='1' height='1'%3E%3C/rect%3E%3C/g%3E%3C/g%3E%3C/svg%3E)

继续滑动看下一个

小徐先生的编程世界

向上滑动看下一个