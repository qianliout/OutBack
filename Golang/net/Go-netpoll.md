---
title: "Go 网络模型 netpoll 全揭秘|Strike Freedom"
source: "https://strikefreedom.top/archives/go-netpoll-io-multiplexing-reactor"
author:
  - "[[潘少]]"
published: 2019-11-09
created: 2026-04-09
description: "深度探究 Go 语言的运行时子系统 netpoll"
tags:
  - "clippings"
---

[![](Attachment/Golang/netpoll-001-header.jpeg)](Attachment/Golang/netpoll-001-header.jpeg)

## 导言

Go 基于 I/O multiplexing 和 goroutine scheduler 构建了一个简洁而高性能的原生网络模型(基于 Go 的 I/O 多路复用 `netpoller` )，提供了 `goroutine-per-connection` 这样简单的网络编程模式。在这种模式下，开发者使用的是同步的模式去编写异步的逻辑，极大地降低了开发者编写网络应用时的心智负担，且借助于 Go runtime scheduler 对 goroutines 的高效调度，这个原生网络模型不论从适用性还是性能上都足以满足绝大部分的应用场景。

然而，在工程性上能做到如此高的普适性和兼容性，最终暴露给开发者提供接口/模式如此简洁，其底层必然是基于非常复杂的封装，做了很多取舍，也有可能放弃了一些追求极致性能的设计和理念。事实上 `Go netpoller` 底层就是基于 epoll/kqueue/iocp 这些 I/O 多路复用技术来做封装的，最终暴露出 `goroutine-per-connection` 这样的极简的开发模式给使用者。

Go netpoller 在不同的操作系统，其底层使用的 I/O 多路复用技术也不一样，可以从 Go 源码目录结构和对应代码文件了解 Go 在不同平台下的网络 I/O 模式的实现。比如，在 Linux 系统下基于 epoll，freeBSD 系统下基于 kqueue，以及 Windows 系统下基于 iocp。

本文将基于 Linux 平台来解析 Go netpoller 之 I/O 多路复用的底层是如何基于 epoll 封装实现的，从源码层层推进，全面而深度地解析 Go netpoller 的设计理念和实现原理，以及 Go 是如何利用 `netpoller` 来构建它的原生网络模型的。主要涉及到的一些概念：I/O 模型、用户/内核空间、epoll、Linux 源码、goroutine scheduler 等等，我会尽量简单地讲解，如果有对相关概念不熟悉的同学，还是希望能提前熟悉一下。

## 用户空间与内核空间

现代操作系统都是采用虚拟存储器，那么对 32 位操作系统而言，它的寻址空间（虚拟存储空间）为 4G（2 的 32 次方）。操作系统的核心是内核，独立于普通的应用程序，可以访问受保护的内存空间，也有访问底层硬件设备的所有权限。为了保证用户进程不能直接操作内核（kernel），保证内核的安全，操心系统将虚拟空间划分为两部分，一部分为内核空间，一部分为用户空间。针对 Linux 操作系统而言，将最高的 1G 字节（从虚拟地址 0xC0000000 到 0xFFFFFFFF），供内核使用，称为内核空间，而将较低的 3G 字节（从虚拟地址 0x00000000 到 0xBFFFFFFF），供各个进程使用，称为用户空间。

[![](Attachment/Golang/netpoll-002-user-kernel-space.jpg)](Attachment/Golang/netpoll-002-user-kernel-space.jpg)

现代的网络服务的主流已经完成从 CPU 密集型到 IO 密集型的转变，所以服务端程序对 I/O 的处理必不可少，而一旦操作 I/O 则必定要在用户态和内核态之间来回切换。

## I/O 模型

在神作《UNIX 网络编程》里，总结归纳了 5 种 I/O 模型，包括同步和异步 I/O：

- 阻塞 I/O (Blocking I/O)
- 非阻塞 I/O (Nonblocking I/O)
- I/O 多路复用 (I/O multiplexing)
- 信号驱动 I/O (Signal driven I/O)
- 异步 I/O (Asynchronous I/O)

操作系统上的 I/O 是用户空间和内核空间的数据交互，因此 I/O 操作通常包含以下两个步骤：

1. 等待网络数据到达网卡(读就绪)/等待网卡可写(写就绪) –> 读取/写入到内核缓冲区
2. 从内核缓冲区复制数据 –> 用户空间(读)/从用户空间复制数据 -> 内核缓冲区(写)

而判定一个 I/O 模型是同步还是异步，主要看第二步：数据在用户和内核空间之间复制的时候是不是会阻塞当前进程，如果会，则是同步 I/O，否则，就是异步 I/O。基于这个原则，这 5 种 I/O 模型中只有一种异步 I/O 模型：Asynchronous I/O，其余都是同步 I/O 模型。

这 5 种 I/O 模型的对比如下：

[![](Attachment/Golang/netpoll-003-io-model.jpg)](Attachment/Golang/netpoll-003-io-model.jpg)

### Non-blocking I/O

什么叫非阻塞 I/O，顾名思义就是：所有 I/O 操作都是立刻返回而不会阻塞当前用户进程。I/O 多路复用通常情况下需要和非阻塞 I/O 搭配使用，否则可能会产生意想不到的问题。比如，epoll 的 ET(边缘触发) 模式下，如果不使用非阻塞 I/O，有极大的概率会导致阻塞 event-loop 线程，从而降低吞吐量，甚至导致 bug。

Linux 下，我们可以通过 `fcntl` 系统调用来设置 `O_NONBLOCK` 标志位，从而把 socket 设置成 Non-blocking。当对一个 Non-blocking socket 执行读操作时，流程是这个样子：  
[![](Attachment/Golang/netpoll-004-non-blocking-io.png)](Attachment/Golang/netpoll-004-non-blocking-io.png)

当用户进程发出 read 操作时，如果 kernel 中的数据还没有准备好，那么它并不会 block 用户进程，而是立刻返回一个 EAGAIN error。从用户进程角度讲 ，它发起一个 read 操作后，并不需要等待，而是马上就得到了一个结果。用户进程判断结果是一个 error 时，它就知道数据还没有准备好，于是它可以再次发送 read 操作。一旦 kernel 中的数据准备好了，并且又再次收到了用户进程的 system call，那么它马上就将数据拷贝到了用户内存，然后返回。

**所以，Non-blocking I/O 的特点是用户进程需要不断的主动询问 kernel 数据好了没有。下一节我们要讲的 I/O 多路复用需要和 Non-blocking I/O 配合才能发挥出最大的威力！**

## I/O 多路复用

**所谓 I/O 多路复用指的就是 select/poll/epoll 这一系列的多路选择器：支持单一线程同时监听多个文件描述符（I/O 事件），阻塞等待，并在其中某个文件描述符可读写时收到通知。 I/O 复用其实复用的不是 I/O 连接，而是复用线程，让一个 thread of control 能够处理多个连接（I/O 事件）。**

### select & poll

```c
#include <sys/select.h>

/* According to earlier standards */

#include <sys/time.h>

#include <sys/types.h>

#include <unistd.h>

int select(int nfds, fd_set *readfds, fd_set *writefds, fd_set *exceptfds, struct timeval *timeout);

// 和 select 紧密结合的四个宏：

void FD_CLR(int fd, fd_set *set);

int FD_ISSET(int fd, fd_set *set);

void FD_SET(int fd, fd_set *set);

void FD_ZERO(fd_set *set);
```

select 是 epoll 之前 Linux 使用的 I/O 事件驱动技术。

理解 select 的关键在于理解 fd_set，为说明方便，取 fd_set 长度为 1 字节，fd_set 中的每一 bit 可以对应一个文件描述符 fd，则 1 字节长的 fd_set 最大可以对应 8 个 fd。select 的调用过程如下：

1. 执行 FD_ZERO(&set), 则 set 用位表示是 `0000,0000`
2. 若 fd＝5, 执行 FD_SET(fd, &set); 后 set 变为 0001,0000(第 5 位置为 1)
3. 再加入 fd＝2, fd=1，则 set 变为 `0001,0011`
4. 执行 select(6, &set, 0, 0, 0) 阻塞等待
5. 若 fd=1, fd=2 上都发生可读事件，则 select 返回，此时 set 变为 `0000,0011` (注意：没有事件发生的 fd=5 被清空)

基于上面的调用过程，可以得出 select 的特点：

- 可监控的文件描述符个数取决于 sizeof(fd_set) 的值。假设服务器上 sizeof(fd_set)＝512，每 bit 表示一个文件描述符，则服务器上支持的最大文件描述符是 512\*8=4096。fd_set 的大小调整可参考 [【原创】技术系列之 网络模型（二）](http://www.cppblog.com/CppExplore/archive/2008/03/21/45061.html) 中的模型 2，可以有效突破 select 可监控的文件描述符上限
- 将 fd 加入 select 监控集的同时，还要再使用一个数据结构 array 保存放到 select 监控集中的 fd，一是用于在 select 返回后，array 作为源数据和 fd_set 进行 FD_ISSET 判断。二是 select 返回后会把以前加入的但并无事件发生的 fd 清空，则每次开始 select 前都要重新从 array 取得 fd 逐一加入（FD_ZERO 最先），扫描 array 的同时取得 fd 最大值 maxfd，用于 select 的第一个参数
- 可见 select 模型必须在 select 前循环 array（加 fd，取 maxfd），select 返回后循环 array（FD_ISSET 判断是否有事件发生）

所以，select 有如下的缺点：

1. 最大并发数限制：使用 32 个整数的 32 位，即 32\*32=1024 来标识 fd，虽然可修改，但是有以下第 2, 3 点的瓶颈
2. 每次调用 select，都需要把 fd 集合从用户态拷贝到内核态，这个开销在 fd 很多时会很大
3. 性能衰减严重：每次 kernel 都需要线性扫描整个 fd_set，所以随着监控的描述符 fd 数量增长，其 I/O 性能会线性下降

poll 的实现和 select 非常相似，只是描述 fd 集合的方式不同，poll 使用 pollfd 结构而不是 select 的 fd_set 结构，poll 解决了最大文件描述符数量限制的问题，但是同样需要从用户态拷贝所有的 fd 到内核态，也需要线性遍历所有的 fd 集合，所以它和 select 只是实现细节上的区分，并没有本质上的区别。

### epoll

epoll 是 Linux kernel 2.6 之后引入的新 I/O 事件驱动技术，I/O 多路复用的核心设计是 1 个线程处理所有连接的 `等待消息准备好` I/O 事件，这一点上 epoll 和 select&poll 是大同小异的。但 select&poll 错误预估了一件事，当数十万并发连接存在时，可能每一毫秒只有数百个活跃的连接，同时其余数十万连接在这一毫秒是非活跃的。select&poll 的使用方法是这样的： `返回的活跃连接 == select(全部待监控的连接)` 。

什么时候会调用 select&poll 呢？在你认为需要找出有报文到达的活跃连接时，就应该调用。所以，select&poll 在高并发时是会被频繁调用的。这样，这个频繁调用的方法就很有必要看看它是否有效率，因为，它的轻微效率损失都会被 `高频` 二字所放大。它有效率损失吗？显而易见，全部待监控连接是数以十万计的，返回的只是数百个活跃连接，这本身就是无效率的表现。被放大后就会发现，处理并发上万个连接时，select&poll 就完全力不从心了。这个时候就该 epoll 上场了，epoll 通过一些新的设计和优化，基本上解决了 select&poll 的问题。

epoll 的 API 非常简洁，涉及到的只有 3 个系统调用：

```c
#include <sys/epoll.h>  

int epoll_create(int size); // int epoll_create1(int flags);

int epoll_ctl(int epfd, int op, int fd, struct epoll_event *event);

int epoll_wait(int epfd, struct epoll_event *events, int maxevents, int timeout);
```

其中，epoll_create 创建一个 epoll 实例并返回 epollfd；epoll_ctl 注册 file descriptor 等待的 I/O 事件(比如 EPOLLIN、EPOLLOUT 等) 到 epoll 实例上；epoll_wait 则是阻塞监听 epoll 实例上所有的 file descriptor 的 I/O 事件，它接收一个用户空间上的一块内存地址 (events 数组)，kernel 会在有 I/O 事件发生的时候把文件描述符列表复制到这块内存地址上，然后 epoll_wait 解除阻塞并返回，最后用户空间上的程序就可以对相应的 fd 进行读写了：

#### 补充：关于 epoll 的 3 个常见疑问

1. 这三个系统调用的参数分别代表什么？

- `int epoll_create(int size);`
  - `size`：历史参数。在老内核里表示“预估要监听多少 fd”，新内核基本忽略这个值，但必须传 `> 0` 才合法。
  - 返回值：成功返回 `epfd`（epoll 实例的文件描述符），失败返回 `-1` 并设置 `errno`。
- `int epoll_create1(int flags);`
  - `flags`：常用 `EPOLL_CLOEXEC`，表示在 `exec` 时自动关闭 `epfd`，防止 fd 泄漏到子进程。

- `int epoll_ctl(int epfd, int op, int fd, struct epoll_event *event);`
  - `epfd`：`epoll_create/epoll_create1` 返回的 epoll 实例。
  - `op`：对监听集合执行什么操作。
    - `EPOLL_CTL_ADD`：添加一个 `fd`
    - `EPOLL_CTL_MOD`：修改 `fd` 关心的事件
    - `EPOLL_CTL_DEL`：删除 `fd`
  - `fd`：要被监听的目标文件描述符（socket、管道等）。
  - `event`：告诉内核“这个 fd 关心什么事件，返回时带什么用户数据”。
    - `event->events`：事件掩码，如 `EPOLLIN`、`EPOLLOUT`、`EPOLLET`（边沿触发）、`EPOLLONESHOT` 等。
    - `event->data`：用户自定义数据（`fd`、指针、u64 等），会在 `epoll_wait` 返回时原样带回，方便定位连接上下文。

- `int epoll_wait(int epfd, struct epoll_event *events, int maxevents, int timeout);`
  - `epfd`：目标 epoll 实例。
  - `events`：用户态缓冲区首地址，用来接收“就绪事件数组”。
  - `maxevents`：本次最多取回多少个就绪事件，必须 `> 0`。
  - `timeout`：
    - `-1`：一直阻塞，直到至少 1 个事件就绪
    - `0`：非阻塞，立即返回
    - `>0`：最多等待指定毫秒数，超时返回 0
  - 返回值：`>0` 表示返回了多少个就绪事件；`0` 表示超时；`-1` 表示错误。

2. 有 I/O 事件时，`epoll_wait` 是怎么解除阻塞并返回的？

核心机制可以理解为三段链路：`注册回调 -> 事件进入就绪队列 -> 唤醒等待线程`。

- 第一步：`epoll_ctl(ADD/MOD)` 时，内核把 `fd` 挂到 epoll 的兴趣集合（内核里常用红黑树管理），并把回调关系建立好。
- 第二步：当网卡/设备中断到来，协议栈处理后发现某 `fd` 可读/可写，会触发该 `fd` 对应的 poll 回调（典型路径里会走到 `ep_poll_callback`），把这个 `fd` 放入 epoll 的就绪链表（`rdllist`）。
- 第三步：如果此时有线程在 `epoll_wait` 里睡眠（挂在 epoll 的等待队列上），内核会 `wake_up` 这个等待队列；线程被调度后继续执行，把 `rdllist` 里的就绪事件拷贝到用户传入的 `events` 数组，然后 `epoll_wait` 返回。

关键点：`epoll_wait` 不是“自己轮询所有 fd 扫描到事件”，而是“睡眠等待被回调唤醒，再批量取就绪队列”，这就是它相对 `select/poll` 高效的根本原因之一。

3. I/O 事件是持续的，是不是需要循环调用 `epoll_wait`？

是的，实际服务端一定是事件循环（event loop）反复调用 `epoll_wait`。因为：

- 新连接会持续到来；
- 已有连接会不断出现新的可读/可写事件；
- 一次 `epoll_wait` 只返回“当前这批”就绪事件，不会覆盖未来事件。

一个典型主循环是：

```c
for (;;) {
    int n = epoll_wait(epfd, events, maxevents, timeout);
    if (n < 0) {
        if (errno == EINTR) continue; // 被信号中断，重试
        // 其他错误：记录并处理
    }
    for (int i = 0; i < n; i++) {
        // 根据 events[i].events 处理可读/可写/异常
        // 可读就 read，直到读完；可写就 write，直到写完或不可写
    }
}
```

补充一个很容易混淆的点：

- **LT（Level Trigger，默认）**：只要缓冲区里还有数据没读完，下次 `epoll_wait` 还会继续返回这个事件。
- **ET（Edge Trigger）**：只在“状态变化”时通知一次，通常要配合非阻塞 fd，并在一次回调里尽量“读/写到 `EAGAIN` 为止”，否则容易丢处理时机。

结合 Go 来看：Golang 的 `net` 包在 Linux 下底层走 runtime 的 netpoll（基于 epoll），采用的是 **ET（`EPOLLET`）** 模式；同时配合 socket 非阻塞和“在一次就绪处理中尽量读/写到 `EAGAIN` 为止”的策略，以保证边沿触发下不会遗漏后续处理时机。

```c
#include <unistd.h>

ssize_t read(int fd, void *buf, size_t count);

ssize_t write(int fd, const void *buf, size_t count);
```

epoll 的工作原理如下：

[![](Attachment/Golang/netpoll-005-epoll-principle.png)](Attachment/Golang/netpoll-005-epoll-principle.png)

与 select&poll 相比，epoll 分清了高频调用和低频调用。例如，epoll_ctl 相对来说就是非频繁调用的，而 epoll_wait 则是会被高频调用的。所以 epoll 利用 epoll_ctl 来插入或者删除一个 fd，实现用户态到内核态的数据拷贝，这确保了每一个 fd 在其生命周期只需要被拷贝一次，而不是每次调用 epoll_wait 的时候都拷贝一次。 epoll_wait 则被设计成几乎没有入参的调用，相比 select&poll 需要把全部监听的 fd 集合从用户态拷贝至内核态的做法，epoll 的效率就高出了一大截。

在实现上 epoll 采用红黑树来存储所有监听的 fd，而红黑树本身插入和删除性能比较稳定，时间复杂度 O(logN)。通过 epoll_ctl 函数添加进来的 fd 都会被放在红黑树的某个节点内，所以，重复添加是没有用的。当把 fd 添加进来的时候时候会完成关键的一步：该 fd 会与相应的设备（网卡）驱动程序建立回调关系，也就是在内核中断处理程序为它注册一个回调函数，在 fd 相应的事件触发（中断）之后（设备就绪了），内核就会调用这个回调函数，该回调函数在内核中被称为： `ep_poll_callback` ， **这个回调函数其实就是把这个 fd 添加到 rdllist 这个双向链表（就绪链表）中** 。epoll_wait 实际上就是去检查 rdllist 双向链表中是否有就绪的 fd，当 rdllist 为空（无就绪 fd）时挂起当前进程，直到 rdllist 非空时进程才被唤醒并返回。

相比于 select&poll 调用时会将全部监听的 fd 从用户态空间拷贝至内核态空间并线性扫描一遍找出就绪的 fd 再返回到用户态，epoll_wait 则是直接返回已就绪 fd，因此 epoll 的 I/O 性能不会像 select&poll 那样随着监听的 fd 数量增加而出现线性衰减，是一个非常高效的 I/O 事件驱动技术。

**由于使用 epoll 的 I/O 多路复用需要用户进程自己负责 I/O 读写，从用户进程的角度看，读写过程是阻塞的，所以 select&poll&epoll 本质上都是同步 I/O 模型，而像 Windows 的 IOCP 这一类的异步 I/O，只需要在调用 WSARecv 或 WSASend 方法读写数据的时候把用户空间的内存 buffer 提交给 kernel，kernel 负责数据在用户空间和内核空间拷贝，完成之后就会通知用户进程，整个过程不需要用户进程参与，所以是真正的异步 I/O。**

#### 延伸

另外，我看到有些文章说 epoll 之所以性能高是因为利用了 Linux 的 mmap 内存映射让内核和用户进程共享了一片物理内存，用来存放就绪 fd 列表和它们的数据 buffer，所以用户进程在 `epoll_wait` 返回之后用户进程就可以直接从共享内存那里读取/写入数据了，这让我很疑惑，因为首先看 `epoll_wait` 的函数声明：

```c
int epoll_wait(int epfd, struct epoll_event *events, int maxevents, int timeout);
```

第二个参数：就绪事件列表，是需要在用户空间分配内存然后再传给 `epoll_wait` 的，如果内核会用 mmap 设置共享内存，直接传递一个指针进去就行了，根本不需要在用户态分配内存，多此一举。其次，内核和用户进程通过 mmap 共享内存是一件极度危险的事情，内核无法确定这块共享内存什么时候会被回收，而且这样也会赋予用户进程直接操作内核数据的权限和入口，非常容易出现大的系统漏洞，因此一般极少会这么做。所以我很怀疑 epoll 是不是真的在 Linux kernel 里用了 mmap，我就去看了下最新版本（5.3.9）的 Linux kernel 源码：

```c
/*
 * Implement the event wait interface for the eventpoll file. It is the kernel
 * part of the user space epoll_wait(2).
 */
static int do_epoll_wait(int epfd, struct epoll_event __user *events,
             int maxevents, int timeout)
{
    ...
    /* Time to fish for events ... */
    error = ep_poll(ep, events, maxevents, timeout);
}
// 如果 epoll_wait 入参时设定 timeout == 0, 那么直接通过 ep_events_available 判断当前是否有用户感兴趣的事件发生，如果有则通过 ep_send_events 进行处理
// 如果设置 timeout > 0，并且当前没有用户关注的事件发生，则进行休眠，并添加到 ep->wq 等待队列的头部；对等待事件描述符设置 WQ_FLAG_EXCLUSIVE 标志
// ep_poll 被事件唤醒后会重新检查是否有关注事件，如果对应的事件已经被抢走，那么 ep_poll 会继续休眠等待

static int ep_poll(struct eventpoll *ep, struct epoll_event __user *events, int maxevents, long timeout)
{
    ...
    send_events:
    /*
     * Try to transfer events to user space. In case we get 0 events and
     * there's still timeout left over, we go trying again in search of
     * more luck.
     */
    // 如果一切正常, 有 event 发生, 就开始准备数据 copy 给用户空间了
    // 如果有就绪的事件发生，那么就调用 ep_send_events 将就绪的事件 copy 到用户态内存中，
    // 然后返回到用户态，否则判断是否超时，如果没有超时就继续等待就绪事件发生，如果超时就返回用户态。
    // 从 ep_poll 函数的实现可以看到，如果有就绪事件发生，则调用 ep_send_events 函数做进一步处理
    if (!res && eavail &&
            !(res = ep_send_events(ep, events, maxevents)) && !timed_out)
        goto fetch_events;
    ...
}

// ep_send_events 函数是用来向用户空间拷贝就绪 fd 列表的，它将用户传入的就绪 fd 列表内存简单封装到
// ep_send_events_data 结构中，然后调用 ep_scan_ready_list 将就绪队列中的事件写入用户空间的内存；
// 用户进程就可以访问到这些数据进行处理
static int ep_send_events(struct eventpoll *ep,
                struct epoll_event __user *events, int maxevents)
{
    struct ep_send_events_data esed;
    esed.maxevents = maxevents;
    esed.events = events;
    // 调用 ep_scan_ready_list 函数检查 epoll 实例 eventpoll 中的 rdllist 就绪链表，
    // 并注册一个回调函数 ep_send_events_proc，如果有就绪 fd，则调用 ep_send_events_proc 进行处理
    ep_scan_ready_list(ep, ep_send_events_proc, &esed, 0, false);
    return esed.res;
}

// 调用 ep_scan_ready_list 的时候会传递指向 ep_send_events_proc 函数的函数指针作为回调函数，
// 一旦有就绪 fd，就会调用 ep_send_events_proc 函数
static __poll_t ep_send_events_proc(struct eventpoll *ep, struct list_head *head, void *priv)
{
    ...
    /*
     * If the event mask intersect the caller-requested one,
     * deliver the event to userspace. Again, ep_scan_ready_list()
     * is holding ep->mtx, so no operations coming from userspace
     * can change the item.
     */
    revents = ep_item_poll(epi, &pt, 1);
    // 如果 revents 为 0，说明没有就绪的事件，跳过，否则就将就绪事件拷贝到用户态内存中
    if (!revents)
        continue;
    // 将当前就绪的事件和用户进程传入的数据都通过 __put_user 拷贝回用户空间,
    // 也就是调用 epoll_wait 之时用户进程传入的 fd 列表的内存
    if (__put_user(revents, &uevent->events) || __put_user(epi->event.data, &uevent->data)) {
        list_add(&epi->rdllink, head);
        ep_pm_stay_awake(epi);
        if (!esed->res)
            esed->res = -EFAULT;
        return 0;
    }
    ...
}
```

从 `do_epoll_wait` 开始层层跳转，我们可以很清楚地看到最后内核是通过 `__put_user` 函数把就绪 fd 列表和事件返回到用户空间，而 `__put_user` 正是内核用来拷贝数据到用户空间的标准函数。此外，我并没有在 Linux kernel 的源码中和 epoll 相关的代码里找到 mmap 系统调用做内存映射的逻辑，所以基本可以得出结论：epoll 在 Linux kernel 里并没有使用 mmap 来做用户空间和内核空间的内存共享，所以那些说 epoll 使用了 mmap 的文章都是误解。

## Go netpoller 核心

**Go netpoller 基本原理**

> Go netpoller 通过在底层对 epoll/kqueue/iocp 的封装，从而实现了使用同步编程模式达到异步执行的效果。总结来说，所有的网络操作都以网络描述符 netFD 为中心实现。netFD 与底层 PollDesc 结构绑定，当在一个 netFD 上读写遇到 EAGAIN 错误时，就将当前 goroutine 存储到这个 netFD 对应的 PollDesc 中，同时调用 gopark 把当前 goroutine 给 park 住，直到这个 netFD 上再次发生读写事件，才将此 goroutine 给 ready 激活重新运行。显然，在底层通知 goroutine 再次发生读写等事件的方式就是 epoll/kqueue/iocp 等事件驱动机制。

总所周知，Go 是一门跨平台的编程语言，而不同平台针对特定的功能有不用的实现，这当然也包括了 I/O 多路复用技术，比如 Linux 里的 I/O 多路复用有 `select` 、 `poll` 和 `epoll` ，而 freeBSD 或者 MacOS 里则是 `kqueue` ，而 Windows 里则是基于异步 I/O 实现的 `iocp` ，等等；因此，Go 为了实现底层 I/O 多路复用的跨平台，分别基于上述的这些不同平台的系统调用实现了多版本的 netpollers，具体的源码路径如下：

- [`src/runtime/netpoll_epoll.go`](https://github.com/golang/go/blob/master/src/runtime/netpoll_epoll.go)
- [`src/runtime/netpoll_kqueue.go`](https://github.com/golang/go/blob/master/src/runtime/netpoll_kqueue.go)
- [`src/runtime/netpoll_solaris.go`](https://github.com/golang/go/blob/master/src/runtime/netpoll_solaris.go)
- [`src/runtime/netpoll_windows.go`](https://github.com/golang/go/blob/master/src/runtime/netpoll_windows.go)
- [`src/runtime/netpoll_aix.go`](https://github.com/golang/go/blob/master/src/runtime/netpoll_aix.go)
- [`src/runtime/netpoll_fake.go`](https://github.com/golang/go/blob/master/src/runtime/netpoll_fake.go)

本文的解析基于 `epoll` 版本，如果读者对其他平台的 netpoller 底层实现感兴趣，可以在阅读完本文后自行翻阅其他 netpoller 源码，所有实现版本的机制和原理基本类似，所以了解了 `epoll` 版本的实现后再去学习其他版本实现应该没什么障碍。

接下来让我们通过分析最新的 Go 源码（**v1.25**），全面剖析一下整个 Go netpoller 的运行机制和流程。

## 数据结构

### netFD

`net.Listen("tcp", ":8888")` 方法返回了一个 \*TCPListener，它是一个实现了 `net.Listener` 接口的 struct，而通过 `listener.Accept()` 接收的新连接 \*TCPConn 则是一个实现了 `net.Conn` 接口的 struct，它内嵌了 `net.conn` struct。仔细阅读上面的源码可以发现，不管是 Listener 的 Accept 还是 Conn 的 Read/Write 方法，都是基于一个 `netFD` 的数据结构的操作， `netFD` 是一个网络描述符，类似于 Linux 的文件描述符的概念，netFD 中包含一个 poll.FD 数据结构，而 poll.FD 中包含两个重要的数据结构 Sysfd 和 pollDesc，前者是真正的系统文件描述符，后者对是底层事件驱动的封装，所有的读写超时等操作都是通过调用后者的对应方法实现的。

`netFD` 和 `poll.FD` 的源码：

```go
// src/net/fd_posix.go

// Network file descriptor.
type netFD struct {
    pfd poll.FD
    // immutable until Close
    family      int
    sotype      int
    isConnected bool // handshake completed or use of association with peer
    net         string
    laddr       Addr
    raddr       Addr
}

// src/internal/poll/fd_unix.go
// FD is a file descriptor. The net and os packages use this type as a
// field of a larger type representing a network connection or OS file.

// FD 是文件描述符，在操作系统和网络编程中，文件描述符是一种用于标识打开的文件、套接字或其他 I/O 资源的整数标识符。
// 在 Go 语言中的应用net 和 os 包将文件描述符类型作为更大类型的一个字段。这些大类型用于表示网络连接（如 TCP/UDP 连接）或操作系统文件（如磁盘文件、设备文件）。
type FD struct {
	// Lock sysfd and serialize access to Read and Write methods.
	fdmu fdMutex
	// System file descriptor. Immutable until Close.
	Sysfd int
	// Platform dependent state of the file descriptor.
	SysFile
	// I/O poller.
	pd pollDesc
	// Semaphore signaled when file is closed.
	csema uint32
	// Non-zero if this file has been set to blocking mode.
	isBlocking uint32
	// Whether this is a streaming descriptor, as opposed to a
	// packet-based descriptor like a UDP socket. Immutable.
	IsStream bool
	// Whether a zero byte read indicates EOF. This is false for a
	// message based socket connection.
	ZeroReadIsEOF bool
	// Whether this is a file rather than a network socket.
	isFile bool
}
```

### pollDesc

前面提到了 pollDesc 是底层事件驱动的封装，netFD 通过它来完成各种 I/O 相关的操作，它的定义如下：

```go
// src/internal/poll/fd_poll_runtime.go
type pollDesc struct {
    runtimeCtx uintptr
}
```

> 源码位置: https://go.dev/src/internal/poll/fd_poll_runtime.go （Go 1.25.6）

这里的 struct 只包含了一个指针，而通过 pollDesc 的 init 方法，我们可以找到它具体的定义是在 `runtime.pollDesc` 这里：

```go
// src/internal/poll/fd_poll_runtime.go
func (pd *pollDesc) init(fd *FD) error {
    serverInit.Do(runtime_pollServerInit)
    ctx, errno := runtime_pollOpen(uintptr(fd.Sysfd))
    if errno != 0 {
        if ctx != 0 {
            runtime_pollUnblock(ctx)
            runtime_pollClose(ctx)
        }
        return syscall.Errno(errno)
    }
    pd.runtimeCtx = ctx
    return nil
}

// src/runtime/netpoll.go
// Network poller descriptor.
//
// No heap pointers.
type pollDesc struct {
	_     sys.NotInHeap
	link  *pollDesc      // in pollcache, protected by pollcache.lock
	fd    uintptr        // constant for pollDesc usage lifetime
	fdseq atomic.Uintptr // protects against stale pollDesc

	// atomicInfo holds bits from closing, rd, and wd,
	// which are only ever written while holding the lock,
	// summarized for use by netpollcheckerr,
	// which cannot acquire the lock.
	// After writing these fields under lock in a way that
	// might change the summary, code must call publishInfo
	// before releasing the lock.
	// Code that changes fields and then calls netpollunblock
	// (while still holding the lock) must call publishInfo
	// before calling netpollunblock, because publishInfo is what
	// stops netpollblock from blocking anew
	// (by changing the result of netpollcheckerr).
	// atomicInfo also holds the eventErr bit,
	// recording whether a poll event on the fd got an error;
	// atomicInfo is the only source of truth for that bit.
	atomicInfo atomic.Uint32 // atomic pollInfo

	// rg, wg are accessed atomically and hold g pointers.
	// (Using atomic.Uintptr here is similar to using guintptr elsewhere.)
	rg atomic.Uintptr // pdReady, pdWait, G waiting for read or pdNil
	wg atomic.Uintptr // pdReady, pdWait, G waiting for write or pdNil

	lock    mutex // protects the following fields
	closing bool
	rrun    bool      // whether rt is running
	wrun    bool      // whether wt is running
	user    uint32    // user settable cookie
	rseq    uintptr   // protects from stale read timers
	rt      timer     // read deadline timer
	rd      int64     // read deadline (a nanotime in the future, -1 when expired)
	wseq    uintptr   // protects from stale write timers
	wt      timer     // write deadline timer
	wd      int64     // write deadline (a nanotime in the future, -1 when expired)
	self    *pollDesc // storage for indirect interface. See (*pollDesc).makeArg.
}
```

这里重点关注里面的 `rg` 和 `wg` ，这里两个 uintptr "万能指针"类型，取值分别可能是 `pdReady` 、 `pdWait` 、等待 file descriptor 就绪的 goroutine 也就是 `g` 数据结构以及 `nil` ，它们是实现唤醒 goroutine 的关键。

`runtime.pollDesc` 包含自身类型的一个指针，用来保存下一个 `runtime.pollDesc` 的地址，以此来实现链表，可以减少数据结构的大小，所有的 `runtime.pollDesc` 保存在 `runtime.pollCache` 结构中，定义如下：

atomicInfo 的解释
```
// atomicInfo 保存了来自 closing、rd 和 wd 的位信息，
// 这些信息只在持有锁的情况下写入，
// 汇总后供 netpollcheckerr 使用，
// netpollcheckerr 无法获取锁。
// 在持有锁的情况下写入这些字段并可能改变汇总信息后，
// 代码必须在释放锁之前调用 publishInfo。
// 更改字段然后调用 netpollunblock 的代码（在仍然持有锁的情况下）
// 必须在调用 netpollunblock 之前调用 publishInfo，
// 因为 publishInfo 的作用是阻止 netpollblock 重新阻塞
// （通过改变 netpollcheckerr 的结果）。
// atomicInfo 还保存了 eventErr 位，
// 记录文件描述符上的轮询事件是否遇到错误；
// atomicInfo 是该位的唯一真实来源。
atomicInfo atomic.Uint32 // 原子的 pollInfo
```

                      
filePath: src/runtime/netpoll.go
          
这是 Go 运行时网络轮询系统中 `pollDesc` 结构体的 `atomicInfo` 字段的注释。让我为您翻译并详细解释：

详细解释

1. **作用**
`atomicInfo` 是一个原子操作的 32 位无符号整数，用于汇总多个状态信息：
- `closing`：文件描述符是否正在关闭
- `rd`：读操作的状态
- `wd`：写操作的状态  
- `eventErr`：轮询事件是否遇到错误

2. **并发安全设计**
- **写操作**：必须在持有锁的情况下进行
- **读操作**：通过原子操作无锁读取，供 `netpollcheckerr` 使用
- **状态发布**：写入后必须调用 `publishInfo` 来发布状态变更

3. **关键流程**
1. **持有锁**修改状态字段
2. 调用 `publishInfo` 将状态汇总到 `atomicInfo`
3. **释放锁**（或继续其他操作）
4. 其他线程通过原子读取 `atomicInfo` 获取最新状态

4. **设计意图**
- **避免锁竞争**：`netpollcheckerr` 可以无锁快速检查状态
- **状态一致性**：确保状态变更对所有线程立即可见
- **错误处理**：`eventErr` 位是错误状态的唯一权威来源

这种设计体现了 Go 运行时在高并发场景下的优化策略，通过原子操作减少锁竞争，同时保证状态的一致性。


```go
// src/runtime/netpoll.go
type pollCache struct {
	lock  mutex
	first *pollDesc
	// PollDesc objects must be type-stable,
	// because we can get ready notification from epoll/kqueue
	// after the descriptor is closed/reused.
	// Stale notifications are detected using seq variable,
	// seq is incremented when deadlines are changed or descriptor is reused.
}
```

因为 `runtime.pollCache` 是一个在 runtime 包里的全局变量，因此需要用一个互斥锁来避免 data race 问题，从它的名字也能看出这是一个用于缓存的数据结构，也就是用来提高性能的，具体如何实现呢？

```go
// src/runtime/netpoll.go

const pollBlockSize = 4 * 1024

func (c *pollCache) alloc() *pollDesc {
	lock(&c.lock)
	if c.first == nil {
		type pollDescPadded struct {
			pollDesc
			pad [tagAlign - unsafe.Sizeof(pollDesc{})]byte
		}
		const pdSize = unsafe.Sizeof(pollDescPadded{})
		n := pollBlockSize / pdSize
		if n == 0 {
			n = 1
		}
		// Must be in non-GC memory because can be referenced
		// only from epoll/kqueue internals.
		mem := persistentalloc(n*pdSize, tagAlign, &memstats.other_sys)
		for i := uintptr(0); i < n; i++ {
			pd := (*pollDesc)(add(mem, i*pdSize))
			lockInit(&pd.lock, lockRankPollDesc)
			pd.rt.init(nil, nil)
			pd.wt.init(nil, nil)
			pd.link = c.first
			c.first = pd
		}
	}
	pd := c.first
	c.first = pd.link
	unlock(&c.lock)
	return pd
}
```

> 源码位置: https://go.dev/src/runtime/netpoll.go （Go 1.25）

Go runtime 会在调用 `poll_runtime_pollOpen` 往 epoll 实例注册 fd 之时首次调用 `runtime.pollCache.alloc` 方法时批量初始化大小 4KB 的 `runtime.pollDesc` 结构体的链表，初始化过程中会调用 `runtime.persistentalloc` 来为这些数据结构分配不会被 GC 回收的内存，确保这些数据结构只能被 `epoll` 和 `kqueue` 在内核空间去引用。
再往后每次调用这个方法则会先判断链表头是否已经分配过值了，若是，则直接返回表头这个 `pollDesc` ，这种批量初始化数据进行缓存而后每次都直接从缓存取数据的方式是一种很常见的性能优化手段，在这里这种方式可以有效地提升 netpoller 的吞吐量。
Go runtime 会在关闭 `pollDesc` 之时调用 `runtime.pollCache.free` 释放内存：

```go
// src/runtime/netpoll.go
func (c *pollCache) free(pd *pollDesc) {
	// pd can't be shared here, but lock anyhow because
	// that's what publishInfo documents.
	lock(&pd.lock)

	// Increment the fdseq field, so that any currently
	// running netpoll calls will not mark pd as ready.
	fdseq := pd.fdseq.Load()
	fdseq = (fdseq + 1) & (1<<tagBits - 1)
	pd.fdseq.Store(fdseq)

	pd.publishInfo()

	unlock(&pd.lock)

	lock(&c.lock)
	pd.link = c.first
	c.first = pd
	unlock(&c.lock)
}
```

## 实现原理

使用 Go 编写一个典型的 TCP echo server:

```go
func main() {
	listen, err := net.Listen("tcp", ":8888")
	if err != nil {
		log.Println("listen error: ", err)
		return
	}
	for {
		conn, err := listen.Accept()
		if err != nil {
			log.Println("accept error: ", err)
			break
		}
		// start a new goroutine to handle the new connection.
		go HandleConn(conn)
	}
}

func HandleConn(conn net.Conn) {
	defer conn.Close()
	packet := make([]byte, 1024)
	for {
		// block here if socket is not available for reading data.
		n, err := conn.Read(packet)
		if err != nil {
			log.Println("read socket error: ", err)
			return
		}
		// same as above, block here if socket is not available for writing.
		_, _ = conn.Write(packet[:n])
	}
}
```

上面是一个基于 Go 原生网络模型（基于 netpoller）编写的一个 TCP server，模式是 `goroutine-per-connection` ，在这种模式下，开发者使用的是同步的模式去编写异步的逻辑而且对于开发者来说 I/O 是否阻塞是无感知的，也就是说开发者无需考虑 goroutines 甚至更底层的线程、进程的调度和上下文切换。而 Go netpoller 最底层的事件驱动技术肯定是基于 epoll/kqueue/iocp 这一类的 I/O 事件驱动技术，只不过是把这些调度和上下文切换的工作转移到了 runtime 的 Go scheduler，让它来负责调度 goroutines，从而极大地降低了程序员的心智负担！

Go 的这种同步模式的网络服务器的基本架构通常如下：

[![](Attachment/Golang/netpoll-006-network-arch.png)](Attachment/Golang/netpoll-006-network-arch.png)

上面的示例代码中相关的在源码里的几个数据结构和方法：

```go
// src/net/tcpsock.go
// TCPListener is a TCP network listener. Clients should typically
// use variables of type Listener instead of assuming TCP.
type TCPListener struct {
    fd *netFD
    lc ListenConfig
}

// Accept implements the Accept method in the Listener interface; it
// waits for the next call and returns a generic Conn.
func (l *TCPListener) Accept() (Conn, error) {
    if !l.ok() {
        return nil, syscall.EINVAL
    }
    c, err := l.accept()
    if err != nil {
        return nil, &OpError{Op: "accept", Net: l.fd.net, Source: nil, Addr: l.fd.laddr, Err: err}
    }
    return c, nil
}

func (ln *TCPListener) accept() (*TCPConn, error) {  
    fd, err := ln.fd.accept()  
    if err != nil {  
       return nil, err  
    }  
    return newTCPConn(fd, ln.lc.KeepAlive, ln.lc.KeepAliveConfig, testPreHookSetKeepAlive, testHookSetKeepAlive), nil  
}

func newTCPConn(fd *netFD, keepAliveIdle time.Duration, keepAliveCfg KeepAliveConfig, preKeepAliveHook func(*netFD), keepAliveHook func(KeepAliveConfig)) *TCPConn {  
    setNoDelay(fd, true)  
    if !keepAliveCfg.Enable && keepAliveIdle >= 0 {  
       keepAliveCfg = KeepAliveConfig{  
          Enable: true,  
          Idle:   keepAliveIdle,  
       }  
    }    c := &TCPConn{conn{fd}}  
    if keepAliveCfg.Enable {  
       if preKeepAliveHook != nil {  
          preKeepAliveHook(fd)  
       }  
       c.SetKeepAliveConfig(keepAliveCfg)  
       if keepAliveHook != nil {  
          keepAliveHook(keepAliveCfg)  
       }  
    }    
    return c  
}

// TCPConn is an implementation of the Conn interface for TCP network
// connections.
type TCPConn struct {
    conn
}

// src/net/net.go
type conn struct {
    fd *netFD
}

func (c *conn) ok() bool { return c != nil && c.fd != nil }

// Read implements the Conn Read method.
func (c *conn) Read(b []byte) (int, error) {
	if !c.ok() {
		return 0, syscall.EINVAL
	}
	n, err := c.fd.Read(b)
	if err != nil && err != io.EOF {
		err = &OpError{Op: "read", Net: c.fd.net, Source: c.fd.laddr, Addr: c.fd.raddr, Err: err}
	}
	return n, err
}

// Write implements the Conn Write method.
func (c *conn) Write(b []byte) (int, error) {
	if !c.ok() {
		return 0, syscall.EINVAL
	}
	n, err := c.fd.Write(b)
	if err != nil {
		err = &OpError{Op: "write", Net: c.fd.net, Source: c.fd.laddr, Addr: c.fd.raddr, Err: err}
	}
	return n, err
}
```

> 源码位置: https://go.dev/src/net/tcpsock.go （Go 1.25）

### net.Listen

基于 Go 1.25.3（Linux）看 `net.Listen`，关键路径是：

1. `socket` 创建非阻塞 + `close-on-exec` 的 listener fd；
2. `listenStream` 完成 `bind + listen`；
3. `fd.init()` 进入 `internal/poll`，把该 fd 注册到 runtime netpoller；
4. runtime 侧使用 **epoll +  **（不是 pipe）实现唤醒与事件分发。

调用链可概括为：

`net.Listen` -> `net.socket` -> `(*netFD).listenStream` -> `(*netFD).init` -> `(*poll.FD).Init` -> `(*pollDesc).init` -> `runtime_pollServerInit` -> `poll_runtime_pollServerInit` -> `netpollGenericInit` -> `netpollinit`（平台实现）

其中有两个容易误解的点：

1. `serverInit sync.Once` 是 **进程级** 初始化，保证 runtime poller 只初始化一次，不是“每个 listener 一个 epoll”；
2. `netpoll`（等待事件）由 runtime 调度器驱动，不是 `netFD` 主动直接调用。

相关源码（Go 1.25.3）：

```go
// src/net/sock_cloexec.go
// socket 返回的 fd 在 Linux 上会带 SOCK_NONBLOCK|SOCK_CLOEXEC

// Wrapper around the socket system call that marks the returned file
// descriptor as nonblocking and close-on-exec.
func sysSocket(family, sotype, proto int) (int, error) {
	// See ../syscall/exec_unix.go for description of ForkLock.
	syscall.ForkLock.RLock()
	s, err := socketFunc(family, sotype, proto)
	if err == nil {
		syscall.CloseOnExec(s)
	}
	syscall.ForkLock.RUnlock()
	if err != nil {
		return -1, os.NewSyscallError("socket", err)
	}
	if err = syscall.SetNonblock(s, true); err != nil {
		poll.CloseFunc(s)
		return -1, os.NewSyscallError("setnonblock", err)
	}
	return s, nil
}


// src/net/sock_posix.go
func (fd *netFD) listenStream(ctx context.Context, laddr sockaddr, backlog int, ctrlCtxFn func(context.Context, string, string, syscall.RawConn) error) error {
	// ...
	if err = syscall.Bind(fd.pfd.Sysfd, lsa); err != nil {
		return os.NewSyscallError("bind", err)
	}
	if err = listenFunc(fd.pfd.Sysfd, backlog); err != nil {
		return os.NewSyscallError("listen", err)
	}
	if err = fd.init(); err != nil {
		return err
	}
	// ...
	return nil
}

// src/internal/poll/fd_poll_runtime.go
var serverInit sync.Once

func (pd *pollDesc) init(fd *FD) error {
	serverInit.Do(runtime_pollServerInit)
	ctx, errno := runtime_pollOpen(uintptr(fd.Sysfd))
	if errno != 0 {
		return errnoErr(syscall.Errno(errno))
	}
	pd.runtimeCtx = ctx
	return nil
}
```

```go
// src/runtime/netpoll_epoll.go (linux, go1.25.3)
var (
	epfd           int32 = -1 // epoll descriptor
	netpollEventFd uintptr    // eventfd for netpollBreak
)

func netpollinit() {
	epfd, errno = linux.EpollCreate1(linux.EPOLL_CLOEXEC)
	if errno != 0 {
		throw("runtime: netpollinit failed")
	}
	efd, errno := linux.Eventfd(0, linux.EFD_CLOEXEC|linux.EFD_NONBLOCK)
	if errno != 0 {
		throw("runtime: eventfd failed")
	}
	ev := linux.EpollEvent{Events: linux.EPOLLIN}
	*(**uintptr)(unsafe.Pointer(&ev.Data)) = &netpollEventFd
	errno = linux.EpollCtl(epfd, linux.EPOLL_CTL_ADD, efd, &ev)
	if errno != 0 {
		throw("runtime: epollctl failed")
	}
	netpollEventFd = uintptr(efd)
}

func netpollopen(fd uintptr, pd *pollDesc) uintptr {
	var ev linux.EpollEvent
	ev.Events = linux.EPOLLIN | linux.EPOLLOUT | linux.EPOLLRDHUP | linux.EPOLLET
	tp := taggedPointerPack(unsafe.Pointer(pd), pd.fdseq.Load())
	*(*taggedPointer)(unsafe.Pointer(&ev.Data)) = tp
	return linux.EpollCtl(epfd, linux.EPOLL_CTL_ADD, int32(fd), &ev)
}
```

总结：在 Go 1.25.3 的 Linux 实现里，`net.Listen` 对应的 netpoll 初始化与唤醒机制应理解为 **epoll + eventfd + runtime 调度器驱动**。

### Listener.Accept()

按 Go 1.25.3（Linux）来看，`Listener.Accept()` 的关键流程是：

1. `(*TCPListener).Accept` 调用 `l.fd.accept()`；
2. `(*netFD).accept` 调用 `fd.pfd.Accept()`，拿到新连接 fd；
3. 新 fd 封装成新的 `netFD`，再执行 `netfd.init()` 注册到 runtime netpoller；
4. 若底层 `accept4` 返回 `EAGAIN`，通过 `pd.waitRead` 挂起当前 goroutine，等待可读事件后重试。

```go
// src/net/tcpsock.go
func (l *TCPListener) Accept() (Conn, error) {
	if !l.ok() {
		return nil, syscall.EINVAL
	}
	c, err := l.accept()
	if err != nil {
		return nil, &OpError{Op: "accept", Net: l.fd.net, Source: nil, Addr: l.fd.laddr, Err: err}
	}
	return c, nil
}

func (ln *TCPListener) accept() (*TCPConn, error) {
	fd, err := ln.fd.accept()
	if err != nil {
		return nil, err
	}
	return newTCPConn(fd, ln.lc.KeepAlive, ln.lc.KeepAliveConfig, testPreHookSetKeepAlive, testHookSetKeepAlive), nil
}
```

```go
// src/net/fd_unix.go
func (fd *netFD) accept() (netfd *netFD, err error) {
	d, rsa, errcall, err := fd.pfd.Accept()
	if err != nil {
		if errcall != "" {
			err = wrapSyscallError(errcall, err)
		}
		return nil, err
	}

	if netfd, err = newFD(d, fd.family, fd.sotype, fd.net); err != nil {
		poll.CloseFunc(d)
		return nil, err
	}
	if err = netfd.init(); err != nil {
		netfd.Close()
		return nil, err
	}

	lsa, _ := syscall.Getsockname(netfd.pfd.Sysfd)
	netfd.setAddr(netfd.addrFunc()(lsa), netfd.addrFunc()(rsa))
	return netfd, nil
}
```

`poll.FD.Accept()` 的核心逻辑（Go 1.25.3）：

1. 先 `prepareRead`；
2. 循环调用 `accept(fd.Sysfd)`（Linux 下优先走 `accept4` 并设置 `SOCK_NONBLOCK|SOCK_CLOEXEC`）；
3. `EINTR` 重试；
4. `EAGAIN` 时调用 `waitRead` 挂起 goroutine，ready 后继续重试；
5. `ECONNABORTED` 直接重试。

```go
// src/internal/poll/fd_unix.go
func (fd *FD) Accept() (int, syscall.Sockaddr, string, error) {
	if err := fd.readLock(); err != nil {
		return -1, nil, "", err
	}
	defer fd.readUnlock()

	if err := fd.pd.prepareRead(fd.isFile); err != nil {
		return -1, nil, "", err
	}
	for {
		s, rsa, errcall, err := accept(fd.Sysfd)
		if err == nil {
			return s, rsa, "", err
		}
		switch err {
		case syscall.EINTR:
			continue
		case syscall.EAGAIN:
			if fd.pd.pollable() {
				if err = fd.pd.waitRead(fd.isFile); err == nil {
					continue
				}
			}
		case syscall.ECONNABORTED:
			continue
		}
		return -1, nil, errcall, err
	}
}
```

```go
// src/internal/poll/sock_cloexec.go (linux)
func accept(s int) (int, syscall.Sockaddr, string, error) {
	ns, sa, err := Accept4Func(s, syscall.SOCK_NONBLOCK|syscall.SOCK_CLOEXEC)
	if err != nil {
		return -1, nil, "accept4", err
	}
	return ns, sa, "", nil
}
```

要点总结：

1. **事件管理是“进程级统一”的，不是“每个 listener 一套 epoll”**  
   新连接 fd 并不是“挂在某个 listener 的私有 epoll 实例”下面，而是和其他网络 fd 一样，统一纳入 runtime 的 netpoll 体系管理。这样做的好处是事件分发与 goroutine 唤醒逻辑一致，不需要按 listener 切分一堆 poller。
2. **`Accept` 看起来阻塞，本质是 goroutine 被 park，不是线程被卡住**  
   非阻塞 `accept` 遇到 `EAGAIN` 后，代码会走 `pd.waitRead(...)`；这一步挂起的是当前 goroutine，底层线程可以继续跑别的 goroutine。等到 netpoll 发现 listener 可读，再把这个 goroutine 唤醒继续 `accept`。
3. **Linux 优先 `accept4 + NONBLOCK + CLOEXEC`，是“一次系统调用原子设置”**  
   `accept4` 在拿到新连接的同时就把 fd 设成非阻塞和 close-on-exec，避免后续再用 `fcntl` 补设。这样既减少系统调用次数，也减少了“先拿到 fd、后设置标志”这段时间窗口里的竞态风险。

### Conn.Read/Conn.Write

按 Go 1.25.3（Linux）来看，`Conn.Read/Write` 的路径与 `Accept` 一致：fd 全程是非阻塞 I/O，遇到 `EAGAIN` 不会阻塞线程，而是通过 `pollDesc.waitRead/waitWrite` 挂起 goroutine，等 netpoll 就绪后继续。

`Conn.Read` 调用链：

`(*conn).Read` -> `(*netFD).Read` -> `(*poll.FD).Read` -> `syscall.Read`（经 `ignoringEINTRIO` 包装）

```go
// src/net/net.go
func (c *conn) Read(b []byte) (int, error) {
	if !c.ok() {
		return 0, syscall.EINVAL
	}
	n, err := c.fd.Read(b)
	if err != nil && err != io.EOF {
		err = &OpError{Op: "read", Net: c.fd.net, Source: c.fd.laddr, Addr: c.fd.raddr, Err: err}
	}
	return n, err
}

func (c *conn) Write(b []byte) (int, error) {
	if !c.ok() {
		return 0, syscall.EINVAL
	}
	n, err := c.fd.Write(b)
	if err != nil {
		err = &OpError{Op: "write", Net: c.fd.net, Source: c.fd.laddr, Addr: c.fd.raddr, Err: err}
	}
	return n, err
}
```

```go
// src/net/fd_posix.go
func (fd *netFD) Read(p []byte) (n int, err error) {
	n, err = fd.pfd.Read(p)
	runtime.KeepAlive(fd)
	return n, wrapSyscallError(readSyscallName, err)
}

func (fd *netFD) Write(p []byte) (nn int, err error) {
	nn, err = fd.pfd.Write(p)
	runtime.KeepAlive(fd)
	return nn, wrapSyscallError(writeSyscallName, err)
}
```

```go
// src/internal/poll/fd_unix.go
func (fd *FD) Read(p []byte) (int, error) {
	if err := fd.readLock(); err != nil {
		return 0, err
	}
	defer fd.readUnlock()

	if len(p) == 0 {
		return 0, nil
	}
	if err := fd.pd.prepareRead(fd.isFile); err != nil {
		return 0, err
	}
	if fd.IsStream && len(p) > maxRW {
		p = p[:maxRW]
	}
	for {
		n, err := ignoringEINTRIO(syscall.Read, fd.Sysfd, p)
		if err != nil {
			n = 0
			if err == syscall.EAGAIN && fd.pd.pollable() {
				if err = fd.pd.waitRead(fd.isFile); err == nil {
					continue
				}
			}
		}
		err = fd.eofError(n, err)
		return n, err
	}
}
```

`Write` 与 `Read` 对称：先 `prepareWrite`，再 `syscall.Write`；若 `EAGAIN` 则 `waitWrite` 挂起 goroutine，ready 后重试。

这一节的核心是：`Conn.Read/Write` 的“阻塞行为”是 goroutine 级别的调度阻塞，而不是内核阻塞式 socket I/O。

### pollDesc.waitRead/pollDesc.waitWrite

按 Go 1.25.3，`waitRead/waitWrite` 的主链路是：

`internal/poll.(*pollDesc).waitRead/waitWrite` -> `runtime_pollWait` -> `runtime.poll_runtime_pollWait` -> `runtime.netpollblock` -> `gopark`

达成无 I/O 事件时 park 住 goroutine 的目的：

先看 `internal/poll` 这一层（接口很薄，主要做 mode 转发）：

```go
// src/internal/poll/fd_poll_runtime.go
func (pd *pollDesc) wait(mode int, isFile bool) error {
	if pd.runtimeCtx == 0 {
		return errors.New("waiting for unsupported file type")
	}
	res := runtime_pollWait(pd.runtimeCtx, mode)
	return convertErr(res, isFile)
}

func (pd *pollDesc) waitRead(isFile bool) error  { return pd.wait('r', isFile) }
func (pd *pollDesc) waitWrite(isFile bool) error { return pd.wait('w', isFile) }
```

runtime 这一层负责真正的阻塞/唤醒语义：

```go
// src/runtime/netpoll.go
//go:linkname poll_runtime_pollWait internal/poll.runtime_pollWait
func poll_runtime_pollWait(pd *pollDesc, mode int) int {
	errcode := netpollcheckerr(pd, int32(mode))
	if errcode != pollNoError {
		return errcode
	}
	// Go 1.25.3: level-triggered 例外平台包含 wasip1
	if GOOS == "solaris" || GOOS == "illumos" || GOOS == "aix" || GOOS == "wasip1" {
		netpollarm(pd, mode)
	}
	for !netpollblock(pd, int32(mode), false) {
		errcode = netpollcheckerr(pd, int32(mode))
		if errcode != pollNoError {
			return errcode
		}
	}
	return pollNoError
}
```

`netpollblock` 在 Go 1.25.3 的关键变化是：通过 `atomic.Uintptr` 的方法（`CompareAndSwap/Load/Swap`）管理 `pd.rg/pd.wg`，而不是旧版直接操作 `uintptr`：

```go
func netpollblock(pd *pollDesc, mode int32, waitio bool) bool {
	gpp := &pd.rg
	if mode == 'w' {
		gpp = &pd.wg
	}

	for {
		if gpp.CompareAndSwap(pdReady, pdNil) {
			return true
		}
		if gpp.CompareAndSwap(pdNil, pdWait) {
			break
		}
		if v := gpp.Load(); v != pdReady && v != pdNil {
			throw("runtime: double wait")
		}
	}

	if waitio || netpollcheckerr(pd, mode) == pollNoError {
		gopark(netpollblockcommit, unsafe.Pointer(gpp), waitReasonIOWait, traceBlockNet, 5)
	}

	old := gpp.Swap(pdNil)
	if old > pdWait {
		throw("runtime: corrupted polldesc")
	}
	return old == pdReady
}
```

`netpollblockcommit` 在 park 提交阶段把当前 goroutine 记录到 `pd.rg/pd.wg`，并通过 `netpollAdjustWaiters(1)` 更新等待计数。

```go
// src/runtime/netpoll.go
func netpollblockcommit(gp *g, gpp unsafe.Pointer) bool {
	r := atomic.Casuintptr((*uintptr)(gpp), pdWait, uintptr(unsafe.Pointer(gp)))
	if r {
		// Bump the count of goroutines waiting for the poller.
		// The scheduler uses this to decide whether to block
		// waiting for the poller if there is nothing else to do.
		
		// 增加等待 poller 的 goroutine 计数。
		// 调度器利用该计数来判断，当没有其他任务可执行时，是否需要阻塞等待 poller
		netpollAdjustWaiters(1)
	}
	return r
}
```

解释如下：

1. `gpp` 是 `pd.rg` 或 `pd.wg` 的地址，在进入该函数前它的预期状态是 `pdWait`；
2. `Casuintptr(..., pdWait, gp)` 成功表示当前 goroutine 已正式挂到该 `pollDesc` 的等待槽位；
3. 只有 CAS 成功才调用 `netpollAdjustWaiters(1)`，避免重复计数；
4. 返回值用于告诉 `gopark/park_m`：提交是否成功，失败则不应继续按“已入队”路径处理。

因此，`waitRead/waitWrite` 的本质是：

1. 当前 goroutine 进入 `gopark`；
2. fd ready 后由 netpoll 路径唤醒对应 goroutine；
3. goroutine 返回到 `Read/Write/Accept` 的重试循环继续执行。

### netpoll

前面已经从源码层面分析了 netpoll 如何通过 park goroutine 来实现 `Accept/Read/Write` 的阻塞语义。这里有一个关键点需要澄清：`gopark` 并不会把 goroutine 放进 epoll 的 `interest list`。epoll 里维护的是 fd 事件（内核对象），而 goroutine 的等待状态保存在 runtime 的 `pollDesc.rg/wg` 槽位中。此时 G 状态从 `_Grunning` 变为 `_Gwaiting`，后续由 `runtime.netpoll` 处理就绪事件并走 `goready` 路径恢复调度。

所以我们现在可以来从整体的层面来概括 Go 的网络业务 goroutine 是如何被规划调度的了：

[![](Attachment/Golang/netpoll-007-goroutine-scheduling.png)](Attachment/Golang/netpoll-007-goroutine-scheduling.png)

首先，client 连接 server 时，listener 通过 `accept` 接收新连接，并由业务层通常为每个连接启动 goroutine 处理。runtime 会把该连接的 fd 注册到 epoll；当 goroutine 调用 `conn.Read`/`conn.Write` 进入等待时，goroutine 被 `gopark` 挂起，等待指针写入 `pollDesc.rg/wg`，而不是写入 epoll。随后 Go scheduler 在 `runtime.schedule()/findrunnable()` 以及 sysmon 路径里调用 `runtime.netpoll`，拿到可运行 goroutine 列表，再通过 `injectglist` 注入到调度队列。
 
那么当 I/O 事件发生之后，netpoller 是通过什么方式唤醒那些在 I/O wait 的 goroutine 的？答案是通过 `runtime.netpoll` 。

`runtime.netpoll`（Linux, go1.26+）核心逻辑如下：

1. 根据 `delay` 计算 `epoll_wait` 的 `waitms`；
2. 调用 `EpollWait` 获取就绪 fd 事件；
3. 遇到 `netpollEventFd`（`eventfd`）时清空唤醒信号；
4. 对普通 fd 事件计算 `mode`（读/写），并通过 `netpollready` 组装可运行 G 链表；
5. 返回 `(toRun, delta)`，其中 `delta` 用于更新 `netpollWaiters` 计数。

```go
// go1.26+ (linux) 关键签名
func netpoll(delay int64) (gList, int32)
func netpollready(toRun *gList, pd *pollDesc, mode int32) int32
func netpollunblock(pd *pollDesc, mode int32, ioready bool, delta *int32) *g
```

netpoll函数

```go 
// src/runtime/netpoll_epoll.go

// netpoll checks for ready network connections.
// Returns a list of goroutines that become runnable,
// and a delta to add to netpollWaiters.
// This must never return an empty list with a non-zero delta.
//
// delay < 0: blocks indefinitely
// delay == 0: does not block, just polls
// delay > 0: block for up to that many nanoseconds
func netpoll(delay int64) (gList, int32) {
	if epfd == -1 {
		return gList{}, 0
	}
	var waitms int32
	if delay < 0 {
		waitms = -1
	} else if delay == 0 {
		waitms = 0
	} else if delay < 1e6 {
		waitms = 1
	} else if delay < 1e15 {
		waitms = int32(delay / 1e6)
	} else {
		// An arbitrary cap on how long to wait for a timer.
		// 1e9 ms == ~11.5 days.
		waitms = 1e9
	}
	var events [128]syscall.EpollEvent
retry:
	n, errno := syscall.EpollWait(epfd, events[:], int32(len(events)), waitms)
	if errno != 0 {
		if errno != _EINTR {
			println("runtime: epollwait on fd", epfd, "failed with", errno)
			throw("runtime: netpoll failed")
		}
		// If a timed sleep was interrupted, just return to
		// recalculate how long we should sleep now.
		if waitms > 0 {
			return gList{}, 0
		}
		goto retry
	}
	var toRun gList
	delta := int32(0)
	for i := int32(0); i < n; i++ {
		ev := events[i]
		if ev.Events == 0 {
			continue
		}

		if *(**uintptr)(unsafe.Pointer(&ev.Data)) == &netpollEventFd {
			if ev.Events != syscall.EPOLLIN {
				println("runtime: netpoll: eventfd ready for", ev.Events)
				throw("runtime: netpoll: eventfd ready for something unexpected")
			}
			if delay != 0 {
				// netpollBreak could be picked up by a
				// nonblocking poll. Only read the 8-byte
				// integer if blocking.
				// Since EFD_SEMAPHORE was not specified,
				// the eventfd counter will be reset to 0.
				var one uint64
				read(int32(netpollEventFd), noescape(unsafe.Pointer(&one)), int32(unsafe.Sizeof(one)))
				netpollWakeSig.Store(0)
			}
			continue
		}

		var mode int32
		if ev.Events&(syscall.EPOLLIN|syscall.EPOLLRDHUP|syscall.EPOLLHUP|syscall.EPOLLERR) != 0 {
			mode += 'r'
		}
		if ev.Events&(syscall.EPOLLOUT|syscall.EPOLLHUP|syscall.EPOLLERR) != 0 {
			mode += 'w'
		}
		if mode != 0 {
			tp := *(*taggedPointer)(unsafe.Pointer(&ev.Data))
			pd := (*pollDesc)(tp.pointer())
			tag := tp.tag()
			if pd.fdseq.Load() == tag {
				pd.setEventErr(ev.Events == syscall.EPOLLERR, tag)
				delta += netpollready(&toRun, pd, mode)
			}
		}
	}
	return toRun, delta
}

```

Go 在多种场景下都可能调用 `netpoll` 检查 fd 状态。`epoll_wait` 返回的是“就绪 fd 事件”，runtime 再据此从 `pollDesc.rg/wg` 取出对应 goroutine，组装为 `gList`。随后通过 `injectglist` 放入全局队列或本地队列，等待 M 绑定 P 执行。

具体调用 `netpoll` 的地方，首先在 Go runtime scheduler 循环调度 goroutines 之时就有可能会调用 `netpoll` 获取到已就绪的 fd 对应的 goroutine 来调度执行。

首先 Go scheduler 的核心方法 `runtime.schedule()` 里会调用 `runtime.findrunnable()` 获取可运行 goroutine，而 `findrunnable()` 内会在合适时机调用 `runtime.netpoll` 拉取 I/O 就绪对应的 goroutine 列表：

```go
// One round of scheduler: find a runnable goroutine and execute it.

// Never returns.

func schedule() {
    ...

  if gp == nil {
        gp, inheritTime = findrunnable() // blocks until work is available
    }
    ...
}

// Finds a runnable goroutine to execute.
// Tries to steal from other P's, get g from global queue, poll network.
func findrunnable() (gp *g, inheritTime bool) {
  ...
  // Poll network.
    if netpollinited() && (netpollWaiters.Load() > 0 || pollUntil != 0) && atomic.Xchg64(&sched.lastpoll, 0) != 0 {
        atomic.Store64(&sched.pollUntil, uint64(pollUntil))
        if _g_.m.p != 0 {
            throw("findrunnable: netpoll with p")
        }
        if _g_.m.spinning {
            throw("findrunnable: netpoll with spinning")
        }
        pollDelay := delta
        if faketime != 0 {
            // When using fake time, just poll.
            pollDelay = 0
        }
        list, deltaNetpoll := netpoll(pollDelay) // go1.26+: netpoll 返回 (gList, delta)
        netpollAdjustWaiters(deltaNetpoll)
        atomic.Store64(&sched.pollUntil, 0)
        atomic.Store64(&sched.lastpoll, uint64(nanotime()))
        if faketime != 0 && list.empty() {
            // Using fake time and nothing is ready; stop M.
            // When all M's stop, checkdead will call timejump.
            stopm()
            goto top
        }
        lock(&sched.lock)
        _p_ = pidleget() // 查找是否有空闲的 P 可以来就绪的 goroutine
        unlock(&sched.lock)
        if _p_ == nil {
            injectglist(&list) // 如果当前没有空闲的 P，则把就绪的 goroutine 放入全局调度队列等待被执行
        } else {
            // 如果当前有空闲的 P，则 pop 出一个 g，返回给调度器去执行，
            // 并通过调用 injectglist 把剩下的 g 放入全局调度队列或者当前 P 本地调度队列
            acquirep(_p_)
            if !list.empty() {
                gp := list.pop()
                injectglist(&list)
                casgstatus(gp, _Gwaiting, _Grunnable)
                if trace.enabled {
                    traceGoUnpark(gp, 0)
                }
                return gp, false
            }
            if wasSpinning {
                _g_.m.spinning = true
                atomic.Xadd(&sched.nmspinning, 1)
            }
            goto top
        }
    } else if pollUntil != 0 && netpollinited() {
        pollerPollUntil := int64(atomic.Load64(&sched.pollUntil))
        if pollerPollUntil == 0 || pollerPollUntil > pollUntil {
            netpollBreak()
        }
    }
    stopm()
    goto top
}
```

另外， `sysmon` 监控线程会在循环过程中检查距离上一次 `runtime.netpoll` 被调用是否超过了 10ms，若是则会去调用它拿到可运行的 goroutine 列表并通过调用 injectglist 把 g 列表放入全局调度队列或者当前 P 本地调度队列等待被执行：

```go
// Always runs without a P, so write barriers are not allowed.
//go:nowritebarrierrec
func sysmon() {
        ...
        // poll network if not polled for more than 10ms
        lastpoll := int64(atomic.Load64(&sched.lastpoll))
        if netpollinited() && lastpoll != 0 && lastpoll+10*1000*1000 < now {
            atomic.Cas64(&sched.lastpoll, uint64(lastpoll), uint64(now))
            list, deltaNetpoll := netpoll(0) // non-blocking
            netpollAdjustWaiters(deltaNetpoll)
            if !list.empty() {
                // Need to decrement number of idle locked M's
                // (pretending that one more is running) before injectglist.
                // Otherwise it can lead to the following situation:
                // injectglist grabs all P's but before it starts M's to run the P's,
                // another M returns from syscall, finishes running its G,
                // observes that there is no work to do and no other running M's
                // and reports deadlock.
                incidlelocked(-1)
                injectglist(&list)
                incidlelocked(1)
            }
        }
  ...
}
```

Go runtime 在程序启动时会创建一个独立的 M 作为监控线程，叫 `sysmon`。该线程无需 P 即可运行，循环间隔会动态调整（不是固定的 `20us~10ms` 常量区间）。`sysmon` 中会轮询执行以下操作（如上面的代码所示）：

1. 以非阻塞的方式调用 `runtime.netpoll` ，从中找出能从网络 I/O 中唤醒的 g 列表，并通过调用 injectglist 把 g 列表放入全局调度队列或者当前 P 本地调度队列等待被执行，调度触发时，有可能从这个全局 runnable 调度队列获取 g。然后再循环调用 `startm` ，直到所有 P 都不处于 `_Pidle` 状态。
2. 调用 `retake` ，抢占长时间处于 `_Psyscall` 状态的 P。

综上，Go 借助于 epoll/kqueue/iocp 和 runtime scheduler 等的帮助，设计出了自己的 I/O 多路复用 netpoller，成功地让 `Listener.Accept` / `conn.Read` / `conn.Write` 等方法从开发者的角度看来是同步模式。

## Go netpoller 的价值

通过前面对源码的分析，我们现在知道 Go netpoller 依托于 runtime scheduler，为开发者提供了一种强大的同步网络编程模式；然而，Go netpoller 存在的意义却远不止于此，Go netpoller I/O 多路复用搭配 Non-blocking I/O 而打造出来的这个原生网络模型，它最大的价值是把网络 I/O 的控制权牢牢掌握在 Go 自己的 runtime 里，关于这一点我们需要从 Go 的 runtime scheduler 说起，Go 的 G-P-M 调度模型如下：

[![](Attachment/Golang/netpoll-008-gpm-scheduler.png)](Attachment/Golang/netpoll-008-gpm-scheduler.png)

G 在运行过程中如果被阻塞在某个 system call 操作上，那么不光 G 会阻塞，执行该 G 的 M 也会解绑 P(实质是被 sysmon 抢走了)，与 G 一起进入 sleep 状态。如果此时有 idle 的 M，则 P 与其绑定继续执行其他 G；如果没有 idle M，但仍然有其他 G 要去执行，那么就会创建一个新的 M。当阻塞在 system call 上的 G 完成 syscall 调用后，G 会去尝试获取一个可用的 P，如果没有可用的 P，那么 G 会被标记为 `_Grunnable` 并把它放入全局的 runqueue 中等待调度，之前的那个 sleep 的 M 将再次进入 sleep。

现在清楚为什么 netpoll 为什么一定要使用非阻塞 I/O 了吧？就是为了避免让操作网络 I/O 的 goroutine 陷入到系统调用从而进入内核态，因为一旦进入内核态，整个程序的控制权就会发生转移(到内核)，不再属于用户进程了，那么也就无法借助于 Go 强大的 runtime scheduler 来调度业务程序的并发了；而有了 netpoll 之后，借助于非阻塞 I/O ，G 就再也不会因为系统调用的读写而 (长时间) 陷入内核态，当 G 被阻塞在某个 network I/O 操作上时，实际上它不是因为陷入内核态被阻塞住了，而是被 Go runtime 调用 gopark 给 park 住了，此时 G 会被放置到某个 wait queue 中，而 M 会尝试运行下一个 `_Grunnable` 的 G，如果此时没有 `_Grunnable` 的 G 供 M 运行，那么 M 将解绑 P，并进入 sleep 状态。当 I/O available，在 epoll 的 `eventpoll.rdr` 中等待的 G 会被放到 `eventpoll.rdllist` 链表里并通过 `netpoll` 中的 `epoll_wait` 系统调用返回放置到全局调度队列或者 P 的本地调度队列，标记为 `_Grunnable` ，等待 P 绑定 M 恢复执行。

## Goroutine 的调度

这一小节主要是讲处理网络 I/O 的 goroutines 阻塞之后，Go scheduler 具体是如何像前面几个章节所说的那样，避免让操作网络 I/O 的 goroutine 陷入到系统调用从而进入内核态的，而是封存 goroutine 然后让出 CPU 的使用权从而令 P 可以去调度本地调度队列里的下一个 goroutine 的。

**温馨提示** ：这一小节属于延伸阅读，涉及到的知识点更偏系统底层，需要有一定的汇编语言基础才能通读，另外，这一节对 Go scheduler 的讲解仅仅涉及核心的一部分，不会把整个调度器都讲一遍（事实上如果真要解析 Go scheduler 的话恐怕重开一篇几万字的文章才能基本讲清楚。。。），所以也要求读者对 Go 的并发调度器有足够的了解，因此这一节可能会稍显深奥。当然这一节也可选择不读，因为通过前面的整个解析，我相信读者应该已经能够基本掌握 Go netpoller 处理网络 I/O 的核心细节了，以及能从宏观层面了解 netpoller 对业务 goroutines 的基本调度了。而这一节主要是通过对 goroutines 调度细节的剖析，能够加深读者对整个 Go netpoller 的彻底理解，接上前面几个章节，形成一个完整的闭环。如果对调度的底层细节没兴趣的话这也可以直接跳过这一节，对理解 Go netpoller 的基本原理影响不大，不过还是建议有条件的读者可以看看。

从源码可知，Go scheduler 的调度 goroutine 过程中所调用的核心函数链如下：

```
runtime.schedule --> runtime.execute --> runtime.gogo --> goroutine code --> runtime.goexit --> runtime.goexit1 --> runtime.mcall --> runtime.goexit0 --> runtime.schedule
```

> Go scheduler 会不断循环调用 `runtime.schedule()` 去调度 goroutines，而每个 goroutine 执行完成并退出之后，会再次调用 `runtime.schedule()` ，使得调度器回到调度循环去执行其他的 goroutine，不断循环，永不停歇。
> 
> 当我们使用 `go` 关键字启动一个新 goroutine 时，最终会调用 `runtime.newproc` --> `runtime.newproc1` ，来得到 g， `runtime.newproc1` 会先从 P 的 `gfree` 缓存链表中查找可用的 g，若缓存未生效，则会新创建 g 给当前的业务函数，最后这个 g 会被传给 `runtime.gogo` 去真正执行。

这里首先需要了解一个 gobuf 的结构体，它用来保存 goroutine 的调度信息，是 `runtime.gogo` 的入参：

```go 
// gobuf 存储 goroutine 调度上下文信息的结构体
type gobuf struct {
    // The offsets of sp, pc, and g are known to (hard-coded in) libmach.
    // ctxt is unusual with respect to GC: it may be a
    // heap-allocated funcval, so GC needs to track it, but it
    // needs to be set and cleared from assembly, where it's
    // difficult to have write barriers. However, ctxt is really a
    // saved, live register, and we only ever exchange it between
    // the real register and the gobuf. Hence, we treat it as a
    // root during stack scanning, which means assembly that saves
    // and restores it doesn't need write barriers. It's still
    // typed as a pointer so that any other writes from Go get
    // write barriers.
    sp   uintptr // Stack Pointer 栈指针
    pc   uintptr // Program Counter 程序计数器
    g    guintptr // 持有当前 gobuf 的 goroutine
    ctxt unsafe.Pointer
    ret  sys.Uintreg
    lr   uintptr
    bp   uintptr // for GOEXPERIMENT=framepointer
}
```

执行 `runtime.execute()` ，进而调用 `runtime.gogo` ：

```go
func execute(gp *g, inheritTime bool) {
    _g_ := getg()
    // Assign gp.m before entering _Grunning so running Gs have an
    // M.
    _g_.m.curg = gp
    gp.m = _g_.m
    casgstatus(gp, _Grunnable, _Grunning)
    gp.waitsince = 0
    gp.preempt = false
    gp.stackguard0 = gp.stack.lo + _StackGuard
    if !inheritTime {
        _g_.m.p.ptr().schedtick++
    }
    // Check whether the profiler needs to be turned on or off.
    hz := sched.profilehz
    if _g_.m.profilehz != hz {
        setThreadCPUProfiler(hz)
    }
    if trace.enabled {
        // GoSysExit has to happen when we have a P, but before GoStart.
        // So we emit it here.
        if gp.syscallsp != 0 && gp.sysblocktraced {
            traceGoSysExit(gp.sysexitticks)
        }
        traceGoStart()
    }
    // gp.sched 就是 gobuf
    gogo(&gp.sched)
}
```

这里还需要了解一个概念：g0，Go G-P-M 调度模型中，g 代表 goroutine，而实际上一共有三种 g：

1. 执行用户代码的 g；
2. 执行调度器代码的 g，也即是 g0；
3. 执行 `runtime.main` 初始化工作的 main goroutine；

第一种 g 就是使用 `go` 关键字启动的 goroutine，也是我们接触最多的一类 g；第三种 g 是调度器启动之后用来执行的一系列初始化工作的，包括但不限于启动 `sysmon` 监控线程、内存初始化和启动 GC 等等工作；第二种 g 叫 g0，用来执行调度器代码，g0 在底层和其他 g 是一样的数据结构，但是性质上有很大的区别，首先 g0 的栈大小是固定的，比如在 Linux 或者其他 Unix-like 的系统上一般是固定 8MB，不能动态伸缩，而普通的 g 初始栈大小是 2KB，可按需扩展，g0 其实就是线程栈，我们知道每个线程被创建出来之时都需要操作系统为之分配一个初始固定的线程栈，就是前面说的 8MB 大小的栈，g0 栈就代表了这个线程栈，因此每一个 m 都需要绑定一个 g0 来执行调度器代码，然后跳转到执行用户代码的地方。

`runtime.gogo` 是真正去执行 goroutine 代码的函数，这个函数由汇编实现，为什么需要用汇编？因为 `gogo` 的工作是完成线程 M 上的堆栈切换：从系统堆栈 g0 切换成 goroutine `gp` ，也就是 CPU 使用权和堆栈的切换，这种切换本质上是对 CPU 的 PC、SP 等寄存器和堆栈指针的更新，而这一类精度的底层操作别说是 Go，就算是最贴近底层的 C 也无法做到，这种程度的操作已超出所有高级语言的范畴，因此只能借助于汇编来实现。

`runtime.gogo` 在不同的 CPU 架构平台上的实现各不相同，但是核心原理殊途同归，我们这里选用 amd64 架构的汇编实现来分析，我会在关键的地方加上解释：

```
// func gogo(buf *gobuf)
// restore state from Gobuf; longjmp
TEXT runtime·gogo(SB), NOSPLIT, $16-8
    // 将第一个 FP 伪寄存器所指向的 gobuf 的第一个参数存入 BX 寄存器, 
    // gobuf 的一个参数即是 SP 指针
    MOVQ    buf+0(FP), BX
    MOVQ    gobuf_g(BX), DX  // 将 gp.sched.g 保存到 DX 寄存器
    MOVQ    0(DX), CX        // make sure g != nil
    // 将 tls (thread local storage) 保存到 CX 寄存器，然后把 gp.sched.g 放到 tls[0]，
    // 这样以后调用 getg() 之时就可以通过 TLS 直接获取到当前 goroutine 的 g 结构体实例，
    // 进而可以得到 g 所在的 m 和 p，TLS 里一开始存储的是系统堆栈 g0 的地址
    get_tls(CX)
    MOVQ    DX, g(CX)
    // 下面的指令则是对函数栈的 BP/SP 寄存器(指针)的存取，
    // 最后进入到指定的代码区域，执行函数栈帧
    MOVQ    gobuf_sp(BX), SP    // restore SP
    MOVQ    gobuf_ret(BX), AX
    MOVQ    gobuf_ctxt(BX), DX
    MOVQ    gobuf_bp(BX), BP
    // 这里是在清空 gp.sched，因为前面已经把 gobuf 里的字段值都存入了寄存器，
    // 所以 gp.sched 就可以提前清空了，不需要等到后面 GC 来回收，减轻 GC 的负担
    MOVQ    $0, gobuf_sp(BX)    // clear to help garbage collector
    MOVQ    $0, gobuf_ret(BX)
    MOVQ    $0, gobuf_ctxt(BX)
    MOVQ    $0, gobuf_bp(BX)
    // 把 gp.sched.pc 值放入 BX 寄存器
    // PC 指针指向 gogo 退出时需要执行的函数地址
    MOVQ    gobuf_pc(BX), BX
    // 用 BX 寄存器里的值去修改 CPU 的 IP 寄存器，
    // 这样就可以根据 CS:IP 寄存器的段地址+偏移量跳转到 BX 寄存器里的地址，也就是 gp.sched.pc
    JMP    BX
```

`runtime.gogo` 函数接收 `gp.sched` 这个 `gobuf` 结构体实例，其中保存了函数栈寄存器 SP/PC/BP，如果熟悉操作系统原理的话可以知道这些寄存器是 CPU 进行函数调用和返回时切换对应的函数栈帧所需的寄存器，而 goroutine 的执行和函数调用的原理是一致的，也是 CPU 寄存器的切换过程，所以这里的几个寄存器当前存的就是 G 的函数执行栈，当 goroutine 在处理网络 I/O 之时，如果恰好处于 I/O 就绪的状态的话，则正常完成 `runtime.gogo` ，并在最后跳转到特定的地址，那么这个地址是哪里呢？

我们知道 CPU 执行函数的时候需要知道函数在内存里的代码段地址和偏移量，然后才能去取来函数栈执行，而典型的提供代码段地址和偏移量的寄存器就是 CS 和 IP 寄存器，而 `JMP BX` 指令则是用 BX 寄存器去更新 IP 寄存器，而 BX 寄存器里的值是 `gp.sched.pc` ，那么这个 PC 指针究竟是指向哪里呢？让我们来看另一处源码。

众所周知，启动一个新的 goroutine 是通过 `go` 关键字来完成的，而 go compiler 会在编译期间利用 [`cmd/compile/internal/gc.state.stmt`](https://github.com/golang/go/blob/1984ee00048b63eacd2155cd6d74a2d13e998272/src/cmd/compile/internal/gc/ssa.go#L1039) 和 [`cmd/compile/internal/gc.state.call`](https://github.com/golang/go/blob/1984ee00048b63eacd2155cd6d74a2d13e998272/src/cmd/compile/internal/gc/ssa.go#L4328) 这两个函数将 `go` 关键字翻译成 [`runtime.newproc`](https://github.com/golang/go/blob/1984ee00048b63eacd2155cd6d74a2d13e998272/src/runtime/proc.go#L3523) 函数调用，而 `runtime.newproc` 接收了函数指针和其大小之后，会获取 goroutine 和调用处的程序计数器，接着再调用 [`runtime.newproc1`](https://github.com/golang/go/blob/1984ee00048b63eacd2155cd6d74a2d13e998272/src/runtime/proc.go#L3548) ：

```
// Create a new g in state _Grunnable, starting at fn, with narg bytes

// of arguments starting at argp. callerpc is the address of the go

// statement that created this. The caller is responsible for adding

// the new g to the scheduler.

//

// This must run on the system stack because it's the continuation of

// newproc, which cannot split the stack.

//

//go:systemstack

func newproc1(fn *funcval, argp unsafe.Pointer, narg int32, callergp *g, callerpc uintptr) *g {

  ...

  memclrNoHeapPointers(unsafe.Pointer(&newg.sched), unsafe.Sizeof(newg.sched))

    newg.sched.sp = sp

    newg.stktopsp = sp

    // 把 goexit 函数地址存入 gobuf 的 PC 指针里

    newg.sched.pc = funcPC(goexit) + sys.PCQuantum // +PCQuantum so that previous instruction is in same function

    newg.sched.g = guintptr(unsafe.Pointer(newg))

    gostartcallfn(&newg.sched, fn)

    newg.gopc = callerpc

    newg.ancestors = saveAncestors(callergp)

    newg.startpc = fn.fn

    if _g_.m.curg != nil {

        newg.labels = _g_.m.curg.labels

    }

    if isSystemGoroutine(newg, false) {

        atomic.Xadd(&sched.ngsys, +1)

    }

    casgstatus(newg, _Gdead, _Grunnable)

  ...

}
```

这里可以看到， `newg.sched.pc` 被设置了 `runtime.goexit` 的函数地址， `newg` 就是后面 `runtime.gogo` 执行的 goroutine，因此 `runtime.gogo` 最后的汇编指令 `JMP BX` 是跳转到了 `runtime.goexit` ，让我们来继续看看这个函数做了什么：

```
// The top-most function running on a goroutine

// returns to goexit+PCQuantum. Defined as ABIInternal

// so as to make it identifiable to traceback (this

// function it used as a sentinel; traceback wants to

// see the func PC, not a wrapper PC).

TEXT runtime·goexit<ABIInternal>(SB),NOSPLIT,$0-0

    BYTE    $0x90    // NOP

    CALL    runtime·goexit1(SB)    // does not return

    // traceback from goexit1 must hit code range of goexit

    BYTE    $0x90    // NOP
```

这个函数也是汇编实现的，但是非常简单，就是直接调用 `runtime·goexit1` ：

```
// Finishes execution of the current goroutine.

func goexit1() {

    if raceenabled {

        racegoend()

    }

    if trace.enabled {

        traceGoEnd()

    }

    mcall(goexit0)

}
```

调用 `runtime.mcall` 函数：

```
// func mcall(fn func(*g))

// Switch to m->g0's stack, call fn(g).

// Fn must never return. It should gogo(&g->sched)

// to keep running g.

// 切换回 g0 的系统堆栈，执行 fn(g)

TEXT runtime·mcall(SB), NOSPLIT, $0-8

    // 取入参 funcval 对象的指针存入 DI 寄存器，此时 fn.fn 是 goexit0 的地址

    MOVQ    fn+0(FP), DI

    get_tls(CX)

    MOVQ    g(CX), AX    // save state in g->sched

    MOVQ    0(SP), BX    // caller's PC

    MOVQ    BX, (g_sched+gobuf_pc)(AX)

    LEAQ    fn+0(FP), BX    // caller's SP

    MOVQ    BX, (g_sched+gobuf_sp)(AX)

    MOVQ    AX, (g_sched+gobuf_g)(AX)

    MOVQ    BP, (g_sched+gobuf_bp)(AX)

    // switch to m->g0 & its stack, call fn

    MOVQ    g(CX), BX

    MOVQ    g_m(BX), BX

    // 把 g0 的栈指针存入 SI 寄存器，后面需要用到

    MOVQ    m_g0(BX), SI

    CMPQ    SI, AX    // if g == m->g0 call badmcall

    JNE    3(PC)

    MOVQ    $runtime·badmcall(SB), AX

    JMP    AX

    // 这两个指令是把 g0 地址存入到 TLS 里，

    // 然后从 SI 寄存器取出 g0 的栈指针，

    // 替换掉 SP 寄存器里存的当前 g 的栈指针

    MOVQ    SI, g(CX)    // g = m->g0

    MOVQ    (g_sched+gobuf_sp)(SI), SP    // sp = m->g0->sched.sp

    PUSHQ    AX

    MOVQ    DI, DX

    // 入口处的第一个指令已经把 funcval 实例对象的指针存入了 DI 寄存器，

    // 0(DI) 表示取出 DI 的第一个成员，即 goexit0 函数地址，再存入 DI

    MOVQ    0(DI), DI

    CALL    DI // 调用 DI 寄存器里的地址，即 goexit0

    POPQ    AX

    MOVQ    $runtime·badmcall2(SB), AX

    JMP    AX

    RET
```

可以看到 `runtime.mcall` 函数的主要逻辑是从当前 goroutine 切换回 g0 的系统堆栈，然后调用 fn(g)，此处的 g 即是当前运行的 goroutine，这个方法会保存当前运行的 G 的 PC/SP 到 g->sched 里，以便该 G 可以在以后被重新恢复执行，因为也涉及到寄存器和堆栈指针的操作，所以也需要使用汇编实现，该函数最后会在 g0 系统堆栈下执行 `runtime.goexit0`:

`runtime.goexit0` 的主要工作是就是

1. 利用 CAS 操作把 g 的状态从 `_Grunning` 更新为 `_Gdead` ；
2. 对 g 做一些清理操作，把一些字段值置空；
3. 调用 `runtime.dropg` 解绑 g 和 m；
4. 把 g 放入 p 存储 g 的 `gfree` 链表作为缓存，后续如果需要启动新的 goroutine 则可以直接从链表里取而不用重新初始化分配内存。
5. 最后，调用 `runtime.schedule()` 再次进入调度循环去调度新的 goroutines，永不停歇。

另一方面，如果 goroutine 处于 I/O 不可用状态，我们前面已经分析过 netpoller 利用非阻塞 I/O + I/O 多路复用避免了陷入系统调用，所以此时会调用 `runtime.gopark` 并把 goroutine 暂时封存在用户态空间，并休眠当前的 goroutine，因此不会阻塞 `runtime.gogo` 的汇编执行，而是通过 `runtime.mcall` 调用 `runtime.park_m` ：

`runtime.mcall` 方法我们在前面已经介绍过，它主要的工作就是是从当前 goroutine 切换回 g0 的系统堆栈，然后调用 fn(g)，而此时 `runtime.mcall` 调用执行的是 `runtime.park_m` ，这个方法里会利用 CAS 把当前运行的 goroutine – gp 的状态 从 `_Grunning` 切换到 `_Gwaiting` ，表明该 goroutine 已进入到等待唤醒状态，此时封存和休眠 G 的操作就完成了，只需等待就绪之后被重新唤醒执行即可。最后调用 `runtime.schedule()` 再次进入调度循环，去执行下一个 goroutine，充分利用 CPU。

至此，我们完成了对 Go netpoller 原理剖析的整个闭环。

## Go netpoller 的问题

Go netpoller 的设计不可谓不精巧、性能也不可谓不高，配合 goroutine 开发网络应用的时候就一个字：爽。因此 Go 的网络编程模式是及其简洁高效的，然而，没有任何一种设计和架构是完美的， `goroutine-per-connection` 这种模式虽然简单高效，但是在某些极端的场景下也会暴露出问题：goroutine 虽然非常轻量，它的自定义栈内存初始值仅为 2KB，后面按需扩容；海量连接的业务场景下， `goroutine-per-connection` ，此时 goroutine 数量以及消耗的资源就会呈线性趋势暴涨，虽然 Go scheduler 内部做了 g 的缓存链表，可以一定程度上缓解高频创建销毁 goroutine 的压力，但是对于瞬时性暴涨的长连接场景就无能为力了，大量的 goroutines 会被不断创建出来，从而对 Go runtime scheduler 造成极大的调度压力和侵占系统资源，然后资源被侵占又反过来影响 Go scheduler 的调度，进而导致性能下降。

## Reactor 网络模型

目前 Linux 平台上主流的高性能网络库/框架中，大都采用 Reactor 模式，比如 netty、libevent、libev、ACE，POE(Perl)、Twisted(Python)等。

Reactor 模式本质上指的是使用 `I/O 多路复用(I/O multiplexing) + 非阻塞 I/O(non-blocking I/O)` 的模式。

通常设置一个主线程负责做 event-loop 事件循环和 I/O 读写，通过 select/poll/epoll_wait 等系统调用监听 I/O 事件，业务逻辑提交给其他工作线程去做。而所谓『非阻塞 I/O』的核心思想是指避免阻塞在 read() 或者 write() 或者其他的 I/O 系统调用上，这样可以最大限度的复用 event-loop 线程，让一个线程能服务于多个 sockets。在 Reactor 模式中，I/O 线程只能阻塞在 I/O multiplexing 函数上（select/poll/epoll_wait）。

Reactor 模式的基本工作流程如下：

- Server 端完成在 `bind&listen` 之后，将 listenfd 注册到 epollfd 中，最后进入 event-loop 事件循环。循环过程中会调用 `select/poll/epoll_wait` 阻塞等待，若有在 listenfd 上的新连接事件则解除阻塞返回，并调用 `socket.accept` 接收新连接 connfd，并将 connfd 加入到 epollfd 的 I/O 复用（监听）队列。
- 当 connfd 上发生可读/可写事件也会解除 `select/poll/epoll_wait` 的阻塞等待，然后进行 I/O 读写操作，这里读写 I/O 都是非阻塞 I/O，这样才不会阻塞 event-loop 的下一个循环。然而，这样容易割裂业务逻辑，不易理解和维护。
- 调用 `read` 读取数据之后进行解码并放入队列中，等待工作线程处理。
- 工作线程处理完数据之后，返回到 event-loop 线程，由这个线程负责调用 `write` 把数据写回 client。

accept 连接以及 conn 上的读写操作若是在主线程完成，则要求是非阻塞 I/O，因为 Reactor 模式一条最重要的原则就是：I/O 操作不能阻塞 event-loop 事件循环。 **实际上 event loop 可能也可以是多线程的，只是一个线程里只有一个 select/poll/epoll_wait** 。

上面提到了 Go netpoller 在某些场景下可能因为创建太多的 goroutine 而过多地消耗系统资源，而在现实世界的网络业务中，服务器持有的海量连接中在极短的时间窗口内只有极少数是 active 而大多数则是 idle，就像这样（非真实数据，仅仅是为了比喻）：

[![](Attachment/Golang/netpoll-009-conn-rate.png)](Attachment/Golang/netpoll-009-conn-rate.png)

那么为每一个连接指派一个 goroutine 就显得太过奢侈了，而 Reactor 模式这种利用 I/O 多路复用进而只需要使用少量线程即可管理海量连接的设计就可以在这样网络业务中大显身手了：

[![MultiReactors.png](Attachment/Golang/netpoll-010-multi-reactors.png)](Attachment/Golang/netpoll-010-multi-reactors.png)

在绝大部分应用场景下，我推荐大家还是遵循 Go 的 best practices，使用原生的 Go 网络库来构建自己的网络应用。然而，在某些极度追求性能、压榨系统资源以及技术栈必须是原生 Go （不考虑 C/C++ 写中间层而 Go 写业务层）的业务场景下，我们可以考虑自己构建 Reactor 网络模型。

## gnet

[gnet](https://gnet.host/) 是一个基于事件驱动的高性能和轻量级网络框架。它直接使用 [epoll](https://en.wikipedia.org/wiki/Epoll) 和 [kqueue](https://en.wikipedia.org/wiki/Kqueue) 系统调用而非标准 Go 网络包： [net](https://golang.org/pkg/net/) 来构建网络应用，它的工作原理类似两个开源的网络库： [netty](https://github.com/netty/netty) 和 [libuv](https://github.com/libuv/libuv) ，这也使得 `gnet` 达到了一个远超 Go [net](https://golang.org/pkg/net/) 的性能表现。

`gnet` 设计开发的初衷不是为了取代 Go 的标准网络库： [net](https://golang.org/pkg/net/) ，而是为了创造出一个类似于 [Redis](http://redis.io/) 、 [Haproxy](http://www.haproxy.org/) 能高效处理网络包的 Go 语言网络服务器框架。

`gnet` 的卖点在于它是一个高性能、轻量级、非阻塞的纯 Go 实现的传输层（TCP/UDP/Unix Domain Socket）网络框架，开发者可以使用 `gnet` 来实现自己的应用层网络协议(HTTP、RPC、Redis、WebSocket 等等)，从而构建出自己的应用层网络应用：比如在 `gnet` 上实现 HTTP 协议就可以创建出一个 HTTP 服务器 或者 Web 开发框架，实现 Redis 协议就可以创建出自己的 Redis 服务器等等。

[gnet](https://github.com/panjf2000/gnet) ，在某些极端的网络业务场景，比如海量连接、高频短连接、网络小包等等场景， [gnet](https://github.com/panjf2000/gnet) 在性能和资源占用上都远超 Go 原生的 [net](https://golang.org/pkg/net/) 包（基于 netpoller）。

`gnet` 已经实现了 `Multi-Reactors` 和 `Multi-Reactors + Goroutine Pool` 两种网络模型，也得益于这些网络模型，使得 `gnet` 成为一个高性能和低损耗的 Go 网络框架：

[![MultiReactors.png](Attachment/Golang/netpoll-010-multi-reactors.png)](Attachment/Golang/netpoll-010-multi-reactors.png)

[![multireactorsthreadpool.png](Attachment/Golang/netpoll-011-multi-reactors-thread-pool.png)](Attachment/Golang/netpoll-011-multi-reactors-thread-pool.png)

### 🚀 功能

- [高性能](#-%E6%80%A7%E8%83%BD%E6%B5%8B%E8%AF%95) 的基于多线程/Go程网络模型的 event-loop 事件驱动
- 内置 goroutine 池，由开源库 [ants](https://github.com/panjf2000/ants) 提供支持
- 内置 bytes 内存池，由开源库 [bytebufferpool](https://github.com/valyala/bytebufferpool) 提供支持
- 整个生命周期是无锁的
- 简单易用的 APIs
- 基于 Ring-Buffer 的高效且可重用的内存 buffer
- 支持多种网络协议/IPC 机制： `TCP` 、 `UDP` 和 `Unix Domain Socket`
- 支持多种负载均衡算法： `Round-Robin(轮询)` 、 `Source-Addr-Hash(源地址哈希)` 和 `Least-Connections(最少连接数)`
- 支持两种事件驱动机制： **Linux** 里的 `epoll` 以及 **FreeBSD/DragonFly/Darwin** 里的 `kqueue`
- 支持异步写操作
- 灵活的事件定时器
- SO_REUSEPORT 端口重用
- 内置多种编解码器，支持对 TCP 数据流分包：LineBasedFrameCodec, DelimiterBasedFrameCodec, FixedLengthFrameCodec 和 LengthFieldBasedFrameCodec，参考自 [netty codec](https://netty.io/4.1/api/io/netty/handler/codec/package-summary.html) ，而且支持自定制编解码器
- 支持 Windows 平台，基于 ~~IOCP 事件驱动机制~~ Go 标准网络库
- 实现 `gnet` 客户端
