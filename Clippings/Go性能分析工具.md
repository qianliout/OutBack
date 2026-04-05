---
title: "Go性能分析工具"
source: "https://farmerchillax.github.io/2023/07/04/Go%E6%80%A7%E8%83%BD%E5%88%86%E6%9E%90%E5%B7%A5%E5%85%B7/"
author:
published:
created: 2026-04-04
description:
tags:
  - "clippings"
---
对于 Golang 程序性能分析来说，pprof 一定是一个大杀器般的存在。主要可以分析 CPU、内存的使用情况、阻塞情况、Goroutine 的堆栈信息以及锁争用情况等性能问题。

pprof 是一个性能分析工具，Go 在语言层面就内置了 profile 采样工具。这会涉及到 `runtime/pprof` 与 `net/http/pprof` 这两个包。但本文着重于使用 pprof 来分析问题，故不讲解采样相关内容。

## 环境搭建

以大家优秀的编码水平一般不会写出性能堪忧的程序，所以在此我们使用一个 GitHub 上开源的 [项目](https://github.com/FarmerChillax/go-pprof-practice) ，这个项目预埋了许多炸弹代码。这个性能堪忧的“炸弹”可以有效的帮助我们观测到程序的性能问题。

> 务必确保你是在个人机器上运行“炸弹”的，能接受机器死机重启的后果（虽然这发生的概率很低）。请你务必不要在危险的边缘试探，比如在线上服务器运行这个程序。

### 图形化依赖安装

pprof 默认提供命令行的方式来查看各项数据，但命令行的方式显然不够直观。因此我们安装一个图形化的依赖（ [graphviz](https://graphviz.gitlab.io/download/) ）来更直观的展示堆栈信息。

你可以在 [官网](https://graphviz.gitlab.io/download/) 上寻找适合自己操作系统的安装方法，此外在下面这些系统上你可以通过包管理工具来安装它：

```shell
brew install graphviz # macos
apt install graphviz # ubuntu
yum install graphviz # centos
```

### 获取炸弹 💣

该炸弹你可以通过 `git` 克隆下来，再按照一般的 Go 项目方式运行。为了演示的方便我这使用 `go get` 的方式展示，注意加上 `-d` 参数来避免自动安装：

```shell
go get -d github.com/FarmerChillax/go-pprof-practice
cd $GOPATH/src/github.com/FarmerChillax/go-pprof-practice
```

## 指标查看

保持程序的运行，然后打开浏览器访问 `http://127.0.0.1:6060/debug/pprof/` ，可以看到如下页面：

[![](https://farmerchillax.github.io/2023/07/04/Go%E6%80%A7%E8%83%BD%E5%88%86%E6%9E%90%E5%B7%A5%E5%85%B7/pprof_web.jpg "pprof web")](https://farmerchillax.github.io/2023/07/04/Go%E6%80%A7%E8%83%BD%E5%88%86%E6%9E%90%E5%B7%A5%E5%85%B7/pprof_web.jpg)

页面上展示了程序运行采样数据，分别有：

| 类型 | 描述 |
| --- | --- |
| allocs | 内存分配情况的采样信息 |
| blocks | 阻塞操作情况的采样信息 |
| cmdline | 显示程序启动命令及参数 |
| goroutine | 当前所有协程的堆栈信息 |
| heap | 堆上内存使用情况的采样信息 |
| mutex | 锁争用情况的采样信息 |
| profile | CPU 占用情况的采样信息 |
| threadcreate | 系统线程创建情况的采样信息 |
| trace | 程序运行跟踪信息 |

由于直接阅读采样信息缺乏直观性，我们需要借助 `go tool pprof` 命令来排查问题，这个命令是 go 原生自带的，所以不用额外安装。

## CPU 占用过高

首先看一下 CPU 的运行情况，打开管理器可以看到此项目在我电脑上占用了 `63.3%` 的 CPU。

[![](https://farmerchillax.github.io/2023/07/04/Go%E6%80%A7%E8%83%BD%E5%88%86%E6%9E%90%E5%B7%A5%E5%85%B7/cpu_pre.jpg "cpu_pre")](https://farmerchillax.github.io/2023/07/04/Go%E6%80%A7%E8%83%BD%E5%88%86%E6%9E%90%E5%B7%A5%E5%85%B7/cpu_pre.jpg)

这显然是有问题的，我们使用 `go tool pprof` 来采集 10 秒 CPU 数据排查下问题：

`go tool pprof "http://localhost:6060/debug/pprof/profile?seconds=10"`

> 因为我们这里采集的是 profile 类型，因此需要等待一定的时间来对 CPU 做采样。你可以通过查询字符串中 seconds 参数来调节采样时间的长短（单位为秒）

等待一会儿后，会进入一个交互式终端：

[![](https://farmerchillax.github.io/2023/07/04/Go%E6%80%A7%E8%83%BD%E5%88%86%E6%9E%90%E5%B7%A5%E5%85%B7/cpu_terminal.jpg "cpu_terminal")](https://farmerchillax.github.io/2023/07/04/Go%E6%80%A7%E8%83%BD%E5%88%86%E6%9E%90%E5%B7%A5%E5%85%B7/cpu_terminal.jpg)

输入 `top` 命令，查看 CPU 占用较高的调用：

[![](https://farmerchillax.github.io/2023/07/04/Go%E6%80%A7%E8%83%BD%E5%88%86%E6%9E%90%E5%B7%A5%E5%85%B7/cpu_top.jpg "cpu top")](https://farmerchillax.github.io/2023/07/04/Go%E6%80%A7%E8%83%BD%E5%88%86%E6%9E%90%E5%B7%A5%E5%85%B7/cpu_top.jpg)

**参数说明：**

| 类型 | 描述 |
| --- | --- |
| flat | 当前函数本身的执行耗时 |
| flat% | flat 占 CPU 总时间的比例 |
| sum% | 上面每一行的 flat% 总和 |
| cum | 当前函数本身加上其周期函数的总耗时 |
| cum% | cum 占 CPU 总时间的比例 |

很明显 `Eat` 方法是造成 CPU 占用过高的原因，输入 `list Eat` ，查看问题具体在代码的哪一个位置：

[![](https://farmerchillax.github.io/2023/07/04/Go%E6%80%A7%E8%83%BD%E5%88%86%E6%9E%90%E5%B7%A5%E5%85%B7/list_eat.jpg "list eat")](https://farmerchillax.github.io/2023/07/04/Go%E6%80%A7%E8%83%BD%E5%88%86%E6%9E%90%E5%B7%A5%E5%85%B7/list_eat.jpg)

从输出结果里可以看到对应的文件为 `animal/felidae/tiger/tiger.go` ，而且具体的代码行为 24 行的一百亿次 for 循环导致的。

还记得我们一开始安装的 graphviz 图形依赖吗？在安装这个工具后，我们可以通过 `web` 命令来生成一个可视化的页面：

[![](https://farmerchillax.github.io/2023/07/04/Go%E6%80%A7%E8%83%BD%E5%88%86%E6%9E%90%E5%B7%A5%E5%85%B7/cpu_web.jpg "cpu web")](https://farmerchillax.github.io/2023/07/04/Go%E6%80%A7%E8%83%BD%E5%88%86%E6%9E%90%E5%B7%A5%E5%85%B7/cpu_web.jpg)

> 注意，虽然这个命令叫 web，但它实际上是生成一个 `.svg` 文件，并调用系统默认打开它。如果你的系统打开 `.svg` 默认不是浏览器，这时候你需要设置下默认使用浏览器打开，或者使用你喜欢的查看方式来查看 svg 文件

上图中， `Eat` 函数的框特别大，箭头特别粗，就是代表这个函数的 CPU 占用很高（pprof 生怕你不知道.jpg）。到这为止我们已经发现了 cpu 占用过高的原因了，我们修复下这个问题：

[![](https://farmerchillax.github.io/2023/07/04/Go%E6%80%A7%E8%83%BD%E5%88%86%E6%9E%90%E5%B7%A5%E5%85%B7/fix_cpu.jpg "fix cpu")](https://farmerchillax.github.io/2023/07/04/Go%E6%80%A7%E8%83%BD%E5%88%86%E6%9E%90%E5%B7%A5%E5%85%B7/fix_cpu.jpg) [![](https://farmerchillax.github.io/2023/07/04/Go%E6%80%A7%E8%83%BD%E5%88%86%E6%9E%90%E5%B7%A5%E5%85%B7/cpu_result.jpg "fix cpu result")](https://farmerchillax.github.io/2023/07/04/Go%E6%80%A7%E8%83%BD%E5%88%86%E6%9E%90%E5%B7%A5%E5%85%B7/cpu_result.jpg)

## 内存占用过高（Heap）

经过改造，可以发现CPU的问题已经解决了，但是内存使用还是很高，我们需要继续排查内存问题。

[![](https://farmerchillax.github.io/2023/07/04/Go%E6%80%A7%E8%83%BD%E5%88%86%E6%9E%90%E5%B7%A5%E5%85%B7/heap_pre.jpg "pre mark heap")](https://farmerchillax.github.io/2023/07/04/Go%E6%80%A7%E8%83%BD%E5%88%86%E6%9E%90%E5%B7%A5%E5%85%B7/heap_pre.jpg)

上面我们介绍了命令行与 web 页面两种方式，因为 web 页面可视化的方式排查比较直观，因此命令行排查的方式就不再展开了，输入以下命令可以看到堆内存的占用情况：

`go tool pprof -http=:8080 "http://localhost:6060/debug/pprof/heap"`

> 这个命令中 http 选项将会启动一个 web 服务器并自动打开网页。其值为 web 服务器的 endpoint

[![](https://farmerchillax.github.io/2023/07/04/Go%E6%80%A7%E8%83%BD%E5%88%86%E6%9E%90%E5%B7%A5%E5%85%B7/heap_web.jpg "heap web")](https://farmerchillax.github.io/2023/07/04/Go%E6%80%A7%E8%83%BD%E5%88%86%E6%9E%90%E5%B7%A5%E5%85%B7/heap_web.jpg)

从上图我们可以看出 `Mouse` 这个对象的 `Steal` 方法占用的内存最多，然后我们点击 `view` -> `source` 还可以查看到具体的代码文件及其行数：

[![](https://farmerchillax.github.io/2023/07/04/Go%E6%80%A7%E8%83%BD%E5%88%86%E6%9E%90%E5%B7%A5%E5%85%B7/heap_source.jpg "heap source")](https://farmerchillax.github.io/2023/07/04/Go%E6%80%A7%E8%83%BD%E5%88%86%E6%9E%90%E5%B7%A5%E5%85%B7/heap_source.jpg)

采样说明， `SAMPLE` 菜单中有几个选项，对应说明如下：

| 类型 | 描述 |
| --- | --- |
| alloc\_objects | 程序累计申请的对象数 |
| alloc\_space | 程序累计申请的内存大小 |
| inuse\_objects | 程序当前持有的对象数 |
| inuse\_space | 程序当前占用的内存大小 |

从代码中可以看到这么高的内存占用是因为会一直向 m.buffer 里追加长度为 1 MiB 的数组，直到总容量到达 1 GiB 为止，因此我们注释掉相关代码来解决这个问题。  
再次编译运行，查看内存占用：

[![](https://farmerchillax.github.io/2023/07/04/Go%E6%80%A7%E8%83%BD%E5%88%86%E6%9E%90%E5%B7%A5%E5%85%B7/heap_result.jpg "heap result")](https://farmerchillax.github.io/2023/07/04/Go%E6%80%A7%E8%83%BD%E5%88%86%E6%9E%90%E5%B7%A5%E5%85%B7/heap_result.jpg)

> 如果你发现程序运行时间长后，内存还是会升高。请不用担心，这是因为后面用于模拟内存泄漏的炸弹

## 协程（Goroutine）

虽然 Go 是带 GC 的，一般不会发生内存泄漏。但凡事都有例外， `goroutine` 泄露也会导致内存泄露。我们在浏览器 debug 页面能看到此时程序的协程数有 106 条，虽然 106 不多，但对这样一个小程序来说显然是不正常的。

[![](https://farmerchillax.github.io/2023/07/04/Go%E6%80%A7%E8%83%BD%E5%88%86%E6%9E%90%E5%B7%A5%E5%85%B7/goroutine_count_pre.jpg "goroutine pre")](https://farmerchillax.github.io/2023/07/04/Go%E6%80%A7%E8%83%BD%E5%88%86%E6%9E%90%E5%B7%A5%E5%85%B7/goroutine_count_pre.jpg)

我们仍然以可视化的方式来排查这个问题，输入以下命令查看堆内存占用情况：

`go tool pprof -http=:8080 "http://localhost:6060/debug/pprof/goroutine"`

`Graph` 类型我们在上面已经介绍过了方块与箭头的关系，相信你也一定能理解这里的关系。那么我们这次来看看 `Flame Graph` 类型，点击 `view` -> `Flame Graph` ：

[![](https://farmerchillax.github.io/2023/07/04/Go%E6%80%A7%E8%83%BD%E5%88%86%E6%9E%90%E5%B7%A5%E5%85%B7/goroutine_web.jpg "goroutine web")](https://farmerchillax.github.io/2023/07/04/Go%E6%80%A7%E8%83%BD%E5%88%86%E6%9E%90%E5%B7%A5%E5%85%B7/goroutine_web.jpg)

我们在火焰图中可以看到 `wolf.(*Wolf).Drink.func1` 这个函数占了总 goroutine 数量的 95%，我们还是像排查内存一样切换的 `source` 页面，查看代码具体位置：

[![](https://farmerchillax.github.io/2023/07/04/Go%E6%80%A7%E8%83%BD%E5%88%86%E6%9E%90%E5%B7%A5%E5%85%B7/goroutine_source.jpg "goroutine source")](https://farmerchillax.github.io/2023/07/04/Go%E6%80%A7%E8%83%BD%E5%88%86%E6%9E%90%E5%B7%A5%E5%85%B7/goroutine_source.jpg)

可以看到，Drink 方法每次会起 10 个协程，每个协程会 sleep 30 秒再推出，而 Drink 函数又被反复的调用，这才导致了大量的协程泄漏。试想一下，如果我们业务中起的协程会永久阻塞，那么泄漏的协程数量便会持续增加，从而导致内存的持续增加，那么迟早会被 OS Kill 掉。我们通过注释掉问题代码，重新运行可以看到协程数量已经降低到个位数的水平了。

## Mutex（锁）

如果你跟着本文一步步走下来，到此为止我们可以说已经完成了拆蛋的工作了，这个程序的所有资源占用问题已经解决了。但日常业务中排查问题除了资源占用问题外，还有性能问题。

接下来我们进一步排查性能问题，首先能想到的便是 **不合理的锁争用** 的问题，比如加锁时间太长等等。我们重新看一下 debug 页面，虽然协程数量已经大幅度降低，但还显示有一个 `mutex` 争用问题：

[![](https://farmerchillax.github.io/2023/07/04/Go%E6%80%A7%E8%83%BD%E5%88%86%E6%9E%90%E5%B7%A5%E5%85%B7/mutex_pre.jpg "goroutine source")](https://farmerchillax.github.io/2023/07/04/Go%E6%80%A7%E8%83%BD%E5%88%86%E6%9E%90%E5%B7%A5%E5%85%B7/mutex_pre.jpg)

相信看到这里你已经触类旁通了，通过 Graph 查看问题出现的函数，然后通过 `source` 定位具体的代码。还是和之前一样，打开可视化页面：

`go tool pprof -http=:8080 http://localhost:6060/debug/pprof/mutex`

[![](https://farmerchillax.github.io/2023/07/04/Go%E6%80%A7%E8%83%BD%E5%88%86%E6%9E%90%E5%B7%A5%E5%85%B7/mutex_pre_web.jpg "mutex web")](https://farmerchillax.github.io/2023/07/04/Go%E6%80%A7%E8%83%BD%E5%88%86%E6%9E%90%E5%B7%A5%E5%85%B7/mutex_pre_web.jpg) [![](https://farmerchillax.github.io/2023/07/04/Go%E6%80%A7%E8%83%BD%E5%88%86%E6%9E%90%E5%B7%A5%E5%85%B7/mutex_source.jpg "mutex source")](https://farmerchillax.github.io/2023/07/04/Go%E6%80%A7%E8%83%BD%E5%88%86%E6%9E%90%E5%B7%A5%E5%85%B7/mutex_source.jpg)

可以看到，这个锁由主协程 Lock，并启动子协程去 Unlock，主协程会阻塞在第二次 Lock 这里等待子协程完成任务，但由于子协程足足睡眠了一秒，导致主协程等待这个锁释放足足等了一秒钟。我们对此处代码进行修改即可修复问题。

## Block（阻塞）

在程序中除了锁的竞争会导致阻塞外，还有很多逻辑会导致阻塞。我们继续看 debug 页面会发现，这里仍有 2 个阻塞的操作：

[![](https://farmerchillax.github.io/2023/07/04/Go%E6%80%A7%E8%83%BD%E5%88%86%E6%9E%90%E5%B7%A5%E5%85%B7/block_pre.jpg "block pre mark")](https://farmerchillax.github.io/2023/07/04/Go%E6%80%A7%E8%83%BD%E5%88%86%E6%9E%90%E5%B7%A5%E5%85%B7/block_pre.jpg)

阻塞不一定是有问题的，但为了保证程序的性能，我们还是要排查一下。还是上面的那三板斧：

`go tool pprof -http=:8080 http://localhost:6060/debug/pprof/block`

[![](https://farmerchillax.github.io/2023/07/04/Go%E6%80%A7%E8%83%BD%E5%88%86%E6%9E%90%E5%B7%A5%E5%85%B7/block_web.jpg "block web graph")](https://farmerchillax.github.io/2023/07/04/Go%E6%80%A7%E8%83%BD%E5%88%86%E6%9E%90%E5%B7%A5%E5%85%B7/block_web.jpg) [![](https://farmerchillax.github.io/2023/07/04/Go%E6%80%A7%E8%83%BD%E5%88%86%E6%9E%90%E5%B7%A5%E5%85%B7/block_source.jpg "block source")](https://farmerchillax.github.io/2023/07/04/Go%E6%80%A7%E8%83%BD%E5%88%86%E6%9E%90%E5%B7%A5%E5%85%B7/block_source.jpg)

可以看到这里不同于直接 sleep 一秒，这里是从一个 channel 里读数据时，发生了阻塞。直到这个 channel 在一秒后才有数据读出，因此这里会导致程序阻塞，而不是睡眠。  
我们对此代码注释掉，重新编译运行后发现程序还有一个 block。通过排查分析后我们发现是因为程序提供了 HTTP 的 pprof 服务，程序阻塞在对 HTTP 端口的监听上，因此这个阻塞是正常的。

## 基准比对的排查方式

pprof 中有一个 `-base` 选项，它用于指定基准采样文件，这样可以通过比较两个采样数据，从而查看到指标的变化，例如函数的 CPU 使用时间或内存分配情况。

举个具体的例子，在业务中有一个低频调用的接口存在内存泄漏（OOM），它每被调用一就会泄漏 1MiB 的内存。这个接口每天被调用10次。假设我们给这个服务分配了 100Mi 空余的内存，也就是说这个接口基本上每十天就会挂一次，但当我们排查问题的时候，会发现内存是缓慢增长的。此时如果你仅通过 pprof 采样单个文件来观察，基本上很难会发现泄漏点。

这时候 `base` 选项就派上用场了，我们可以在服务启动后采集一个基准样本，过几天后再采集一次。通过比对这两个样本增量数据，我们就很容易发现出泄漏点。

同样的，这个炸弹我也已经预埋了这样一个缓慢的泄漏点，但时间我缩短了一下。相信在上面的实操过程中你也发现了端倪，下面我们开始实操一下。

我们运行这个炸弹程序，将启动时的堆内存分配情况保存下来，你可以在 debug 页面点击下载，也可以在终端中执行 `curl -o heap-base http://localhost:6060/debug/pprof/heap` 来下载。

在资源管理器中我们可以看到程序刚启动的时候，内存占用并不高：

[![](https://farmerchillax.github.io/2023/07/04/Go%E6%80%A7%E8%83%BD%E5%88%86%E6%9E%90%E5%B7%A5%E5%85%B7/base_pre.jpg "base pre")](https://farmerchillax.github.io/2023/07/04/Go%E6%80%A7%E8%83%BD%E5%88%86%E6%9E%90%E5%B7%A5%E5%85%B7/base_pre.jpg)

在程序运行了半分钟后，我们可以清楚的发现程序内存开始逐渐增长：

[![](https://farmerchillax.github.io/2023/07/04/Go%E6%80%A7%E8%83%BD%E5%88%86%E6%9E%90%E5%B7%A5%E5%85%B7/base_pre_target.jpg "base pre target")](https://farmerchillax.github.io/2023/07/04/Go%E6%80%A7%E8%83%BD%E5%88%86%E6%9E%90%E5%B7%A5%E5%85%B7/base_pre_target.jpg)

此时我们再执行 `curl -o heap-target http://localhost:6060/debug/pprof/heap` 获取到当前的采样数据。

> 在 debug 页面中直接点击 heap 链接会打开一个新的页面，你可以通过删除该页面 URL 中的 debug=1 这个查询字符串来下载数据文件

再获取到两个样本数据后，我们通过 `base` 选项将 `heap-base` 作为基准，查看运行的这段时间内哪里内存增长了

`go tool pprof -http=:8080 -base heap-base heap-target`

[![](https://farmerchillax.github.io/2023/07/04/Go%E6%80%A7%E8%83%BD%E5%88%86%E6%9E%90%E5%B7%A5%E5%85%B7/base_web.jpg "base web compare")](https://farmerchillax.github.io/2023/07/04/Go%E6%80%A7%E8%83%BD%E5%88%86%E6%9E%90%E5%B7%A5%E5%85%B7/base_web.jpg)

显然在这段时间中 `mouse` 的 `Pee` 方法增长了 1.20GB，显然这就是内存泄漏的地方，接下来还是通过那三板斧来定位出问题的代码，然后注释修复：

[![](https://farmerchillax.github.io/2023/07/04/Go%E6%80%A7%E8%83%BD%E5%88%86%E6%9E%90%E5%B7%A5%E5%85%B7/base_source.jpg "base source")](https://farmerchillax.github.io/2023/07/04/Go%E6%80%A7%E8%83%BD%E5%88%86%E6%9E%90%E5%B7%A5%E5%85%B7/base_source.jpg)

这里主要展示了如果通过 `base` 参数比对两个 pprof 采样文件，从而提高我们排查问题的效率。

> 在 Graph 页面你可能会发现有一些绿色框框存在，机智的你肯定也能猜到绿色框框代表的就是减少的使用量

## 总结

本文主要内容为 pprof 工具的使用，介绍了通过命令行、可视化等方式进行排查。虽然例子比较简单，但是相信通过这些简单的例子可以让你不在畏惧 pprof。