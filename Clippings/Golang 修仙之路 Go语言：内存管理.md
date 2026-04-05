---
title: "[Golang 修仙之路] Go语言：内存管理"
source: "https://juejin.cn/post/7548757818023952434"
author:
  - "[[小猪乔治爱打球]]"
published: 2025-09-12
created: 2026-04-04
description: "本文章仅供个人学习使用。 参考 小徐先生的编程世界：https://mp.weixin.qq.com/s/2TBwpQT5-zU4Gy7-i0LZmQ go语言设计与实现：https://draven本文章仅供个人学习使用。 参考1.  小徐先生的编程世界：2.  go语言设计与实现： 笼统的概念深入"
tags:
  - "clippings"
---
本文章仅供个人学习使用。

## 参考

1. 小徐先生的编程世界： [mp.weixin.qq.com/s/2TBwpQT5-…](https://link.juejin.cn/?target=https%3A%2F%2Fmp.weixin.qq.com%2Fs%2F2TBwpQT5-zU4Gy7-i0LZmQ "https://mp.weixin.qq.com/s/2TBwpQT5-zU4Gy7-i0LZmQ")
2. go语言设计与实现： [draven.co/golang/docs…](https://link.juejin.cn/?target=https%3A%2F%2Fdraven.co%2Fgolang%2Fdocs%2Fpart3-runtime%2Fch07-memory%2Fgolang-memory-allocator%2F "https://draven.co/golang/docs/part3-runtime/ch07-memory/golang-memory-allocator/")

## 笼统的概念

深入一件事情之前，先笼统了解其运用的思想。后续深入时，时刻想着这些思想，可以避免因为过于深入细节而忽略了整体导致的晕头转向。

### 空闲链表分配 + 按大小分类

空闲链表分配方式是区别于线性分配方式的。通过把空闲的内存空间串联成一个链表，实现内存分配。

这就涉及到一个问题：我要申请一块大小为x字节的内存，如何找到一块儿大于等于x的内存空间用来分配呢？

常见的算法有4种：

- 首次适应（First-Fit）— 从链表头开始遍历，选择第一个大小大于申请内存的内存块；
- 循环首次适应（Next-Fit）— 从上次遍历的结束位置开始遍历，选择第一个大小大于申请内存的内存块；
- 最优适应（Best-Fit）— 从链表头遍历整个链表，选择最合适的内存块；
- `隔离适应（Segregated-Fit）` — **将内存分割成多个链表，每个链表中的内存块大小相同，申请内存时先找到满足条件的链表，再从链表中选择合适的内存块；**

这四种，重点了解最后一种，因为go语言采用了与之类似的思想。

![image.png](https://p3-xtjj-sign.byteimg.com/tos-cn-i-73owjymdk6/70f121efcb514449b3f69cf326f41de0~tplv-73owjymdk6-jj-mark-v1:0:0:0:0:5o6Y6YeR5oqA5pyv56S-5Yy6IEAg5bCP54yq5LmU5rK754ix5omT55CD:q75.awebp?rk3s=f64ab15b&x-expires=1775816229&x-signature=qcO8OfpEwZ%2F9Vy7BPh1mLHpziGw%3D)

### 多级缓存

计算机世界很多地方都有多级缓存的体现：从 计算机硬件 到 大型web后端。Go的内存管理也用到了这种思想。

Go的内存管理，管理的是：虚拟内存。

Go的内存分配器，从用户侧到操作系统虚拟内存，依次是：

- 内存管理单元：mspan
- 线程缓存：mcache
- 中心缓存：mcentral
- 页堆：mheap

![image.png](https://p3-xtjj-sign.byteimg.com/tos-cn-i-73owjymdk6/a5716149a8be4d309a063adc2325e7b8~tplv-73owjymdk6-jj-mark-v1:0:0:0:0:5o6Y6YeR5oqA5pyv56S-5Yy6IEAg5bCP54yq5LmU5rK754ix5omT55CD:q75.awebp?rk3s=f64ab15b&x-expires=1775816229&x-signature=B0mUFrnQD0fgrahmAcZSLPuFA9Q%3D)

## 详细介绍内存管理组件

### mspan

- mspan是内存管理的基本单元，管理的页大小为8KB。
- mspan管理的内存是连续的，所以知道了startAddress，又知道每个页大小固定=8KB，那再知道有多少页npages，就能确定mspan管理的内存范围了。
- Go划分了67种跨度，表示不同对象的大小，mspan的spanclass字段，标识了本span管理的是哪一个大小等级的对象。
- 管理相同大小等级的mspan串联在一起，组成双向链表。
- mspan中使用bitmap来标识哪些「对象」是空闲的。（注意不是页）
- 同等级的mspan从属于同一个mcentral，由同一把互斥锁管理。

---

管理相同大小等级的mspan串联在一起，组成双向链表：

![image.png](https://p3-xtjj-sign.byteimg.com/tos-cn-i-73owjymdk6/730933ccd86e4a458051f89660d84aa6~tplv-73owjymdk6-jj-mark-v1:0:0:0:0:5o6Y6YeR5oqA5pyv56S-5Yy6IEAg5bCP54yq5LmU5rK754ix5omT55CD:q75.awebp?rk3s=f64ab15b&x-expires=1775816229&x-signature=HH2KWMFVYs6QYPVF7CLHrG978rA%3D)

---

mspan管理的内存是连续的，所以知道了startAddress，又知道每个页大小固定=8KB，那再知道有多少页npages，就能确定mspan管理的内存范围了：

![image.png](https://p3-xtjj-sign.byteimg.com/tos-cn-i-73owjymdk6/dac51cefe1c941018d66e6968d15234e~tplv-73owjymdk6-jj-mark-v1:0:0:0:0:5o6Y6YeR5oqA5pyv56S-5Yy6IEAg5bCP54yq5LmU5rK754ix5omT55CD:q75.awebp?rk3s=f64ab15b&x-expires=1775816229&x-signature=IFVgkV64hE4ncmtaOflkjwAC6bc%3D)

---

mspan中使用bitmap来标识哪些「对象」是空闲的。（注意不是页）

![image.png](https://p3-xtjj-sign.byteimg.com/tos-cn-i-73owjymdk6/c8251c781d1846df89081f8b31bd2b7a~tplv-73owjymdk6-jj-mark-v1:0:0:0:0:5o6Y6YeR5oqA5pyv56S-5Yy6IEAg5bCP54yq5LmU5rK754ix5omT55CD:q75.awebp?rk3s=f64ab15b&x-expires=1775816229&x-signature=WTFxxvGjgdYbIhq6XJUAwpmg65w%3D)

---

最后我们来完整的看一眼go代码中的mspan：

```
go 体验AI代码助手 代码解读复制代码type mspan struct {  
    // 标识前后节点的指针   
    next *mspan       
    prev *mspan      

    // 起始地址  
    startAddr uintptr   
    // 包含几页，页是连续的  
    npages    uintptr   
  
    // 标识此前的位置都已被占用   
    freeindex uintptr  
    
    // 最多可以存放多少个 object  
    nelems uintptr // number of object in the span.  
  
    // bitmap 每个 bit 对应一个 object 块，标识该块是否已被占用  
    allocCache uint64  
    
    // 标识 mspan 等级，包含 class 和 noscan 两部分信息  
    spanclass             spanClass      
    
    // ...  
}
```

#### spanClass

跨度类。跨度类是一个8位无符号整数。高7位表示（67种跨度id），最低位表示noscan。

- `noscan` 方法返回true --> 不扫描；否则扫描。
- 垃圾回收会对包含指针的 `runtime.mspan` 结构体进行扫描。
- `noscan` 位=1，方法返回true，所以 `noscan` = 1，表示对象不包含指针。

---

源码如下：

```
go 体验AI代码助手 代码解读复制代码type spanClass uint8  
  
// uint8 左 7 位为 mspan 等级，最右一位标识是否为 noscan  
func makeSpanClass(sizeclass uint8, noscan bool) spanClass {  
    return spanClass(sizeclass<<1) | spanClass(bool2int(noscan))  
}  
  
func (sc spanClass) sizeclass() int8 {  
    return int8(sc >> 1)  
}  
  
func (sc spanClass) noscan() bool {  
    return sc&1 != 0  
}
```

---

对象等级一览表：（其实一共68个等级，还有一种id=0表示超过32KB的对象）

- waste 挺好理解的：我想申请一个大小为9B的对象，class=1就不够，只能申请class=2的，那每申请一个对象就浪费16-9=7B。

| class | bytes/obj | bytes/span | objects | tail waste | max waste |
| --- | --- | --- | --- | --- | --- |
| 1 | 8 | 8192 | 1024 | 0 | 87.50% |
| 2 | 16 | 8192 | 512 | 0 | 43.75% |
| 3 | 24 | 8192 | 341 | 0 | 29.24% |
| 4 | 32 | 8192 | 256 | 0 | 46.88% |
| 5 | 48 | 8192 | 170 | 32 | 31.52% |
| 6 | 64 | 8192 | 128 | 0 | 23.44% |
| 7 | 80 | 8192 | 102 | 32 | 19.07% |
| … | … | … | … | … | … |
| 67 | 32768 | 32768 | 1 | 0 | 12.50% |

### mcache

- mcache 是每个 P 独有的缓存，因此交互无锁
- mcache 缓存了 2（noscan维度） \* 68（spanClass维度） = 136 个mspan。
- mcache 中还有 微对象分配器，用于处理小于16B对象的内存分配。

---

源码：

```
go 体验AI代码助手 代码解读复制代码const numSpanClasses = 136  
type mcache struct {  
    // 微对象分配器相关  
    tiny       uintptr  
    tinyoffset uintptr  
    tinyAllocs uintptr  
      
    // mcache 中缓存的 mspan，每种 spanClass 各一个  
    alloc [numSpanClasses]*mspan  
    
    // ...  
}
```

### mcentral

- 每个mcentral一把锁
- 每个mcentral对应一种spanClass，包含2个mspan链表，分别是有空间的和满的。

---

![c4aaf5cd975607ae1a254efe45fd49f3.png](https://p3-xtjj-sign.byteimg.com/tos-cn-i-73owjymdk6/781f896539694aee98cc170a459212f3~tplv-73owjymdk6-jj-mark-v1:0:0:0:0:5o6Y6YeR5oqA5pyv56S-5Yy6IEAg5bCP54yq5LmU5rK754ix5omT55CD:q75.awebp?rk3s=f64ab15b&x-expires=1775816229&x-signature=TOYTet%2BiH1NGTc5HWf6GLKpJpxs%3D)

---

```
go 体验AI代码助手 代码解读复制代码type mcentral struct {  
    // 对应的 spanClass  
    spanclass spanClass  
    // 有空位的 mspan 集合，数组长度为 2 是用于抗一轮 GC  
    partial [2]spanSet   
    // 无空位的 mspan 集合  
    full    [2]spanSet   
}
```

### mheap

- 是OS虚拟内存的抽象，向虚拟内存申请页面。
- mheap中页大小为8KB，负责把连续的页组装成mspan。
- heapArena记录了page到mspan的映射关系，一个heapArena包含8192个页，总大小是8192 \* 8KB = 64M，这也是mheap向虚拟内存申请内存的基本单位。
- mheap通过类似bitmap的方式，寻找空闲的页，组装成mspan，bit=1表示已组装，bit=0表示空闲。但实际上是通过「基数树」快速定位空闲页面的。
- mheap是mcentral的持有者，缓存了每种等级的mcentral。

---

源码：

mheap

```
go 体验AI代码助手 代码解读复制代码type mheap struct {  
    // 堆的全局锁  
    lock mutex  
  
  
    // 空闲页分配器，底层是多棵基数树组成的索引，每棵树对应 16 GB 内存空间  
    pages pageAlloc   
  
  
    // 记录了所有的 mspan. 需要知道，所有 mspan 都是经由 mheap，使用连续空闲页组装生成的  
    allspans []*mspan  
  
  
    // heapAreana 数组，64 位系统下，二维数组容量为 [1][2^22]  
    // 每个 heapArena 大小 64M，因此理论上，Golang 堆上限为 2^22*64M = 256T  
1

  
  
    // ...  
    // 多个 mcentral，总个数为 spanClass 的个数  
    central [numSpanClasses]struct {  
        mcentral mcentral  
        // 用于内存地址对齐  
        pad      [cpu.CacheLinePadSize - unsafe.Sizeof(mcentral{})%cpu.CacheLinePadSize]byte  
    }  
  
  
    // ...  
}
```

heapArena

```
go 体验AI代码助手 代码解读复制代码const pagesPerArena = 8192  
  
type heapArena struct {  
    // ...  
    // 实现 page 到 mspan 的映射  
    spans [pagesPerArena]*mspan  
  
  
    // ...  
}
```

## 内存分配流程

go把对象按照大小分为：

- 微对象：小于16字节
- 小对象：16到32K字节
- 大对象：大于32K字节

对不同对象有不同的分配策略。核心流程类似多级缓存，“如果没有，就尝试从更底层的缓存中取”

对于微对象：

1. 从P专属的mcache的tiny分配器取内存。（无锁）
2. 根据spanClass，从mcache缓存的mspan中取内存。（无锁）
3. 如果mcache中没有满足条件的空闲mspan，则从mcentral中获取mspan，填充到mcache，然后从mspan中取内存。（spanClass粒度锁）
4. 如果还失败了，则从mheap中取得足够的空闲页组装成mspan，填充mcache，然后从mspan中取内存。（全局锁）
5. mheap中都没有，则mheap向堆申请内存，更新页分配器索引信息，然后用空闲页组装成mspan填充mcache。

对于小对象：（只有上述2--5步） 2. 根据spanClass，从mcache缓存的mspan中取内存。

1. 如果mcache中没有满足条件的空闲mspan，则从mcentral中获取mspan，填充到mcache，然后从mspan中取内存。
2. 如果还失败了，则从mheap中取得足够的空闲页组装成mspan，填充mcache，然后从mspan中取内存。
3. mheap中都没有，则mheap向堆申请内存，更新页分配器索引信息，然后用空闲页组装成mspan填充mcache。

对于大对象：（只有上述4 -- 5步） 4. 如果还失败了，则从mheap中取得足够的空闲页组装成mspan，填充mcache，然后从mspan中取内存。（全局锁）

1. mheap中都没有，则mheap向堆申请内存，更新页分配器索引信息，然后用空闲页组装成mspan填充mcache。

### 源码

我个人还是有一种心态：觉得源码在面试场景很难给面试官说的特别清楚，所以打算只背一些核心方法的方法名和用途。

- mspan没有object可用，会调用mcache.nextFree(spc spanClass)
- 如果mcache还是没有足够空间，则会调用mcache.refill向mcentral甚至mheap申请mspan填充自己。
- mcentral.cacheSpan 方法中，会加锁（spanClass 级别的 sweepLocker），分别从 partial 和 full 中尝试获取有空间的 mspan。
- 如果mcentral中仍然没有足够空间，则会调用mcentral.grow来请求mheap
- mheap核心调用mheap.allocSpan来分配内存。加全局锁。