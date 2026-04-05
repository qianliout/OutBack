---
title: "go/c++ 万字解析 sync.Map 技术原理与源码实战"
source: "https://mp.weixin.qq.com/s/YU_nACyZEkQRIHXWO6He5A"
author:
  - "[[小徐先生1212]]"
published:
created: 2026-04-04
description: "本篇是我在自学 c++ 过程中针对 sync.Map 技术的学习实践心得，分为两部分内容：第一部分向大家介绍了 go 中 sync.Map 的实现原理及使用场景; 第二部分向大家介绍了如何基于 c++ 效仿 go 从零实现 sync.Map"
tags:
  - "clippings"
---
原创 小徐先生1212 *2024年10月6日 17:20*

## 0 前言

在近期自学 c++ 过程中，我希望通过使用 c++ 复刻实现 go 中经典模块的实践方法，在增加 c++ 熟练度的同时，也进一步提升对 go 的理解程度，以此来争取做到触类旁通、一石二鸟.

本期的学习目标是 go 中应用于

```
并发场景的 map 工具——sync.Map
```
，本期内容分为如下两部分：

第 1 章——解析 sync.Map 设计原理与适用场景

第 2 章——基于 c++ 从零实现 sync.Map（并发安全无序表），详细展示其中的技术原理与源码细节

> 我在自学 c++ 过程中涉及到的实践项目都会收录于开源项目 https://github.com/xiaoxuxiansheng/cbricks.git 中，大家若觉得有所帮助，希望能留个 star，这是支持鼓励我持续创作的动力，感谢~

## 1 技术原理

### 1.1 哈希表

![图片](https://mmbiz.qpic.cn/sz_mmbiz_png/3ic3aBqT2ibZsjQIvrq5pm8cuxYfcms1VZYqRWDIWqljVV2KhUQ4EvRMkUc5RQANsuPDu352qkPz9AouEeVjEickg/640?wx_fmt=png&from=appmsg&tp=webp&wxfrom=5&wx_lazy=1#imgIndex=0)

```
map（hashmap）
```
又称
```
哈希表
```
，是一种经典的数据结构，其核心特征为：

- •
	```
	kv 形式
	```
	：基于 key-value 映射的存储形式；
- •
	```
	无重复
	```
	：基于 key 维度实现数据的去重；
- •
	```
	高性能
	```
	：读、写、删等单点访问操作，时间复杂度均为 O(1)

map

```
并非并发安全
```
的数据结构，不允许对其出现并发读写行为，否则会发生致命的错误：

- •
	```
	并发读没有问题
	```
	；
- • 一旦涉及写操作，就不允许存在并发行为. 这里的
	```
	“写”是广义的，包括插入、更新、删除操作
	```
	.

map 结构本身并非今天我们所要探讨的重点，但属于必备的前置基础知识，更多详细内容可以参见我此前发表的文章—— [Golang map 实现原理](http://mp.weixin.qq.com/s?__biz=MzkxMjQzMjA0OQ==&mid=2247483868&idx=1&sn=6e954af8e5e98ec0a9d9fc5c8ceb9072&chksm=c10c4f02f67bc614ff40a152a848508aa1631008eb5a600006c7552915d187179c08d4adf8d7&scene=21#wechat_redirect).

### 1.2 sync.Map 设计原理

实际生产环境中，难免存在并发访问 map 的诉求. 一种简单的实现思路是，为 map 加持一把互斥锁 mutex，进而将访问 map 这一临界资源的行为由并发限制为串行.

上述这种简单直接的处理思路也是没问题的，但今天我们要探讨的是一种新的解决思路——基于 c++ 复刻 go 中 的

```
sync.Map
```
，从零构建一款基于
```
读写分离、延迟删除机制实现的并发安全 map 工具
```
.

在并发场景的 map 选型上，很多写 go 的同学会一味地选择使用 sync.Map 工具，亦或是盲目地使用 mutex + map 的实现思路，这样难免都有欠妥当. 没有绝对完美的方案，只有在指定场景下相对契合的方案，我们应该拒绝一味盲从，在充分了解各种方案设计理念、技术优劣之后，做到

```
因地制宜、充分选型思辨
```
.

有关 go 中 sync.Map 底层原理与源码，我在前文 [Golang sync.Map 实现原理](http://mp.weixin.qq.com/s?__biz=MzkxMjQzMjA0OQ==&mid=2247483821&idx=1&sn=f45e9e2b4c4cb7edaa57d904e3bf7bd7&chksm=c10c4f73f67bc6655e7e7c9f808a318b1c74e76782824df18843f4c31e39ce512819d6abe156&scene=21#wechat_redirect) 中，进行过详细介绍. 在本章中，我们主要提纲挈领地回顾一下其中的技术要点，并为下一章 c++ 实战环节做好铺垫.

```
// go 中的 sync.Map
type Map struct{
    // 互斥锁，保护 dirty
    mu Mutex

    // readonly 原子容器，真实类型 readonly，里面包含了一个通过只读操作访问的 map，类型也是 map[any]*entry
    read atomic.Value// readOnly

    // 拥有全量数据的读写 map，被 mu 保护
    dirty map[any]*entry

    // 操作时 readonly 发生 miss 次数，累计达到一定次数时会更新 readonly
    misses int
}

// readOnly is an immutable struct stored atomically in the Map.read field.
type readOnly struct{
    // 只读 map，完全无锁化访问
    m       map[any]*entry
    // readonly 相比 dirty 是否存在数据缺失
    amended bool// true if the dirty map contains some key not in m.
}

// 硬删除状态
var expunged =unsafe.Pointer(new(any))

// 存储一个值的指针容器
type entry struct{
    p unsafe.Pointer// *interface{}
}

// 写操作
func (m *Map)Store(key, value any){
    // ...
}

// 读操作func (m *Map)Load(key any)(value any, ok bool){
    // ...
}

// 删除操作func (m *Map)Delete(key any){
    // ...
}

// 遍历操作func (m *Map)Range(f func(key, value any)bool){
    // ...
}
```

首先，简单一览 go 中 sync.Map 的核心源码，并提炼出有关 sync.Map 在设计实现上的几项关键信息：

![图片](data:image/svg+xml,%3C%3Fxml version='1.0' encoding='UTF-8'%3F%3E%3Csvg width='1px' height='1px' viewBox='0 0 1 1' version='1.1' xmlns='http://www.w3.org/2000/svg' xmlns:xlink='http://www.w3.org/1999/xlink'%3E%3Ctitle%3E%3C/title%3E%3Cg stroke='none' stroke-width='1' fill='none' fill-rule='evenodd' fill-opacity='0'%3E%3Cg transform='translate(-249.000000, -126.000000)' fill='%23FFFFFF'%3E%3Crect x='249' y='126' width='1' height='1'%3E%3C/rect%3E%3C/g%3E%3C/g%3E%3C/svg%3E)

- •
	```
	空间冗余
	```
	：sync.Map 基于
	```
	以空间换时间
	```
	的思路设计实现，在物理意义上存在
	```
	两个独立 map，分别是 readonly.m 和 dirty
	```
	；
- •
	```
	读写分离
	```
	：
	```
	readonly.m 主要面向无锁访问的只读操作
	```
	， 而
	```
	dirty 则面向加锁访问的读写操作
	```
	. 因此读操作优先无锁化访问 readonly，
	```
	当击穿 readonly 时（readonly 数据 miss），才加锁访问 dirty 进行兜底
	```
	. 通过这种读写分离的机制，把更多读操作引导到 readonly 模块减少加锁互斥的频率，提高整体访问性能.

![图片](data:image/svg+xml,%3C%3Fxml version='1.0' encoding='UTF-8'%3F%3E%3Csvg width='1px' height='1px' viewBox='0 0 1 1' version='1.1' xmlns='http://www.w3.org/2000/svg' xmlns:xlink='http://www.w3.org/1999/xlink'%3E%3Ctitle%3E%3C/title%3E%3Cg stroke='none' stroke-width='1' fill='none' fill-rule='evenodd' fill-opacity='0'%3E%3Cg transform='translate(-249.000000, -126.000000)' fill='%23FFFFFF'%3E%3Crect x='249' y='126' width='1' height='1'%3E%3C/rect%3E%3C/g%3E%3C/g%3E%3C/svg%3E)

- •
	```
	广义读操作
	```
	：两个 map 真正的实现类型是基于
	```
	key-entry
	```
	的形式，而
	```
	entry 是存储 value 指针的容器
	```
	. 因此只有在插入一组不存在 key-entry 时，才是严格意义上的写操作；其他针对已存在的 key-entry 的更新操作，都是
	```
	广义上的读操作
	```
	，可以通过
	```
	读取出 key 对应 entry，再基于 CAS（compare-and-swap）操作完成 entry 内的内容更新
	```
	；
- •
	```
	延迟删除
	```
	：删除 key-entry 时，不是立即从 map 中删除 key，而是尝试
	```
	读到 key 对应 entry，然后通过 cas 将 entry 标记为软删除态（nill）
	```
	. 后续在迁移流程统一执行实际删除操作. 因此针对已存在的 key-entry 的删除操作，也是
	```
	广义上的读操作
	```
	.

![图片](data:image/svg+xml,%3C%3Fxml version='1.0' encoding='UTF-8'%3F%3E%3Csvg width='1px' height='1px' viewBox='0 0 1 1' version='1.1' xmlns='http://www.w3.org/2000/svg' xmlns:xlink='http://www.w3.org/1999/xlink'%3E%3Ctitle%3E%3C/title%3E%3Cg stroke='none' stroke-width='1' fill='none' fill-rule='evenodd' fill-opacity='0'%3E%3Cg transform='translate(-249.000000, -126.000000)' fill='%23FFFFFF'%3E%3Crect x='249' y='126' width='1' height='1'%3E%3C/rect%3E%3C/g%3E%3C/g%3E%3C/svg%3E)

- •
	```
	正向迁移
	```
	：由于 readonly 是只读模式，所以
	```
	新增插入数据会作用到 dirty 中，导致 readonly 数据仅是 dirty 的子集
	```
	. sync.Map 中通过
	```
	misses 计数器记录 readonly 被读操作击穿的次数
	```
	，以此反映存在缺失数据的比重，当比重达到阈值时，会
	```
	将 dirty 中的全量数据覆盖到 readonly 中补齐其缺失的部分
	```
	，此为
	```
	正向迁移流程(missLocked)
	```
	.

![图片](data:image/svg+xml,%3C%3Fxml version='1.0' encoding='UTF-8'%3F%3E%3Csvg width='1px' height='1px' viewBox='0 0 1 1' version='1.1' xmlns='http://www.w3.org/2000/svg' xmlns:xlink='http://www.w3.org/1999/xlink'%3E%3Ctitle%3E%3C/title%3E%3Cg stroke='none' stroke-width='1' fill='none' fill-rule='evenodd' fill-opacity='0'%3E%3Cg transform='translate(-249.000000, -126.000000)' fill='%23FFFFFF'%3E%3Crect x='249' y='126' width='1' height='1'%3E%3C/rect%3E%3C/g%3E%3C/g%3E%3C/svg%3E)

- •
	```
	逆向迁移
	```
	：由于 sync.Map 中的延迟删除机制，被置为删除态的 entry 需要有额外的回收时机. 在执行完正向迁移后，会短暂地将 dirty 置空，并在随后到来的下一次击穿 readonly 的插入行为中，执行
	```
	逆向迁移流程（dirtyLocked）
	```
	——
	```
	遍历 readonly
	```
	，过滤所有删除态 entry，并
	```
	将仍存在的 entry 拷贝到 dirty 中
	```
	.
	> 如此一来，在下一次因正向迁移而使用 dirty 覆盖 readonly 时，这部分删除态 entry 就会丢失引用，实现事实意义上的回收

![图片](data:image/svg+xml,%3C%3Fxml version='1.0' encoding='UTF-8'%3F%3E%3Csvg width='1px' height='1px' viewBox='0 0 1 1' version='1.1' xmlns='http://www.w3.org/2000/svg' xmlns:xlink='http://www.w3.org/1999/xlink'%3E%3Ctitle%3E%3C/title%3E%3Cg stroke='none' stroke-width='1' fill='none' fill-rule='evenodd' fill-opacity='0'%3E%3Cg transform='translate(-249.000000, -126.000000)' fill='%23FFFFFF'%3E%3Crect x='249' y='126' width='1' height='1'%3E%3C/rect%3E%3C/g%3E%3C/g%3E%3C/svg%3E)

- •
	```
	软硬删除
	```
	：sync.Map 的删除行为是在
	```
	逻辑意义上将 entry 更新为软删除态（nil）
	```
	，反之，如果想要
	```
	恢复一笔已删除的 key-entry，也只需要将 entry 状态由软删除态（nil）恢复成正常态
	```
	即可，这样都属于广义上的读行为，是无需加锁. 然而，如果一个 entry 处于
	```
	硬删除态（expunged）
	```
	，那么它就是
	```
	不可恢复
	```
	的，必须通过加锁后执行插入操作来补齐对应数据.

> 软硬删除态的区别：

> ```
> 从物理意义上看
> ```
> ：在于 dirty 中是否还存在该 entry，若不存在，则属于硬删除态，必须加锁并在 dirty 中补齐该数据，因为 dirty 必须拥有全局最完整的数据；

> ```
> 从逻辑时序上看
> ```
> ：在于这笔 entry 被软删除后，是否经历过逆向迁移（dirtyLocked），在此流程中，会将所有软删除态（nil）的 entry 更新为硬删除态（expunged），并会将其过滤不添加到 dirty 中，因此在事实意义上，这部分数据已经被剔除且无法恢复了.

> sync.Map 中之所以对软/硬删除态进行单独设计，就是为了能够细分出哪些 key-entry 是属于近期被删除后还未执行过逆向迁移流程的，这样当重新插入相同 key 值时，就能基于 CAS 实现无锁化，进一步提升访问性能.

### 1.3 sync.Map 适用场景

sync.Map 不是银弹，它只是在特定场景中有着比较不俗的表现：

- •
	```
	读多写少
	```
	：相较于不断插入新 key，sync.Map 更支持的访问模式是，
	```
	一个给定 key 被写入后，会存在多次读访问行为加以复用
	```
	. 这里的读访问是广义上的，同时包括更新和删除操作.

与此同时，我们也需要明确 sync.Map 中存在的几个局限性：

- •
	```
	不适用于插入写操作
	```
	：在插入写操作中，sync.Map 退化为基于互斥锁 mutex + dirty map 实现的基础结构，此时不存在性能优势，其复杂的结构设计反而徒增成本；
- •
	```
	存在性能抖动
	```
	：由于延迟删除机制的存在，在 sync.Map 执行到逆向迁移 dirtyLocked 流程时，会对 readonly 进行线性遍历，这次操作是 O(N) 的，相较于其他 O(1) 的访问操作，存在明显的性能短板
- •
	```
	不利于序列化
	```
	：由于底层复杂的结构设计，sync.Map 不像 go 中普通 map 一样，天然契合序列化/反序列化能力
- •
	```
	不利于数量统计
	```
	：由于延迟删除机制的存在，sync.Map 无法直接通过 entry 个数反映出 key-value 数量，因为其中可能存在处于删除态的 entry

## 2 实现源码

纸上得来终觉浅. 从本章开始，我们进入到 c++ 手写实现 sync.Map 的环节.

项目基于 c++ 11 实现，已开源于https://github.com/xiaoxuxiansheng/cbricks.git中，文件为./sync/map.h.

> 此外，由于 c++ 标准库中，将有序表称为 map，无序表称为 unordered\_map，这里特别声明一下，本文实现的 sync.Map 工具是并发安全的无序表，在命名上可能和 c++ 风格略有差异，但和大多数语言保持一致.

### 2.1 底层基础组件

实现过程中，涉及到的知识点包括但不限于：

- • 原子操作 std::atomic；
- • 智能指针 std::shared\_ptr；
- • 哈希表 std::unordered\_map；
- • 互斥锁 lock（基于 c 风格自制）；
- • 自旋锁 spinLock（基于 c 风格自制）；
- • 模板编程 template；
- • 右值移动函数 std::move.

下面是源码文件中依赖到的头文件：

```
// 原子变量 原子操作
#include <atomic>
// 智能指针
#include <memory>
// 哈希表
#include <unordered_map>
// 字符串
#include <string>
// 函数编程
#include <functional>

// 项目内实现的功能——禁用复制&赋值操作
#include "../base/nocopy.h"
// 项目内实现的功能——互斥锁、自旋锁
#include "lock.h"
```

### 2.2 数据结构设计

![图片](data:image/svg+xml,%3C%3Fxml version='1.0' encoding='UTF-8'%3F%3E%3Csvg width='1px' height='1px' viewBox='0 0 1 1' version='1.1' xmlns='http://www.w3.org/2000/svg' xmlns:xlink='http://www.w3.org/1999/xlink'%3E%3Ctitle%3E%3C/title%3E%3Cg stroke='none' stroke-width='1' fill='none' fill-rule='evenodd' fill-opacity='0'%3E%3Cg transform='translate(-249.000000, -126.000000)' fill='%23FFFFFF'%3E%3Crect x='249' y='126' width='1' height='1'%3E%3C/rect%3E%3C/g%3E%3C/g%3E%3C/svg%3E)

在 sync.Map 的类声明，基于 template 实现由使用方自定义 Key 和 Value 的具体类型. 在 sync.Map 中包含如下成员成员变量：

- •
	```
	s_expunged（静态）
	```
	：通过静态实例指针的方式构造全局统一的硬删除态标识. 如果某个 Entry 中存储的内容为该静态实例的指针，则说明该 Entry 处于硬删除态（expunged）；
- •
	```
	m_readonly
	```
	：基于 atomic 容器缓存 readonly 实例的指针. readonly 实例与 sync.Map 是一一对应的，访问时无需加锁；
- •
	```
	m_misses
	```
	：通过 atomic 变量记录 readonly 因数据缺失而被访问击穿的次数. 此次数达到阈值后，会触发正向迁移 missLocked 流程；
- •
	```
	m_dirtyLock
	```
	：保护 m\_dirty 的互斥锁
- •
	```
	m_dirty
	```
	：存储全量 Key-Entry 数据的 unordered\_map 实例. 访问时需要保证持有 m\_dirtyLock.
```
namespace cbricks{namespace sync{

/**
 * @brief: 并发安全的 map 工具——借鉴 golang sync.Map 实现
 * 适用场景：适用于 插入类写操作较少的 场景
 * 功能点：底层基于以空间换时间的思路，建立 read dirty 两份 map
 *        - readonly：应用于读、更新、删除场景，无锁化操作
 *        - dirty：包含全量数据，访问时需要加锁
 */
template<class Key,class Value>
class Map: base::Noncopyable{
    // ... 
public:
    // 智能指针 类型别名
    typedef std::shared_ptr<Map<Key,Value>> ptr;
    // 存储 key-entry 的 map 类型别名
    typedef std::unordered_map<Key,typename Entry::ptr>map;
    // ...
private:
    /**
     * 使用全局静态变量标记特殊的硬删除状态 expunged
     * 当一个 key 于 dirty 中已不存在时，对应 entry 的 container 中会存储该实例，用于标记该项数据处于硬删除态（expunged）
    */
    static typename Map<Key,Value>::Entry::WrappedV* s_expunged;

private:
    /**
     * 只读 map，全程无锁访问
     * 使用 atomic 容器承载，由于内部仅有一个 bool 及共享指针字段，所以发生值拷贝的成本可忽略不计
     */
    std::atomic<typename Map<Key,Value>::ReadOnly*> m_readonly;
    // 记录 readonly miss 次数，与 missLocked 流程紧密相关
    std::atomic<int> m_misses{0};

    // 互斥锁，访问 dirty map 前需要持有此锁
    Lock m_dirtyLock;
    // dirty map，包含全量数据
    map m_dirty;
};

// 全局静态变量，标记硬删除状态.
template<class Key,class Value>
typename Map<Key,Value>::Entry::WrappedV*Map<Key,Value>::s_expunged =new typename Map<Key,Value>::Entry::WrappedV();
```

![图片](data:image/svg+xml,%3C%3Fxml version='1.0' encoding='UTF-8'%3F%3E%3Csvg width='1px' height='1px' viewBox='0 0 1 1' version='1.1' xmlns='http://www.w3.org/2000/svg' xmlns:xlink='http://www.w3.org/1999/xlink'%3E%3Ctitle%3E%3C/title%3E%3Cg stroke='none' stroke-width='1' fill='none' fill-rule='evenodd' fill-opacity='0'%3E%3Cg transform='translate(-249.000000, -126.000000)' fill='%23FFFFFF'%3E%3Crect x='249' y='126' width='1' height='1'%3E%3C/rect%3E%3C/g%3E%3C/g%3E%3C/svg%3E)

效仿于 go 中的实现，c++ 版 sync.Map 中也包含了几个公有的 api：

- •
	```
	load
	```
	：从 sync.Map 中
	```
	读取数据
	```
	. 入参中的 receiver 为接收 Value 的容器；bool 型返回值标识着 Key 是否存在；
- •
	```
	store
	```
	：向 sync.Map 中
	```
	写入数据
	```
	. 可能是插入写也可能是更新写，对使用方而言无感知；
- •
	```
	evict
	```
	：从 sync.Map 中
	```
	删除数据
	```
	. 无论数据此前是否存在，该方法都会执行成功；
- •
	```
	range
	```
	：
	```
	遍历
	```
	sync.Map 中的数据. 通过入参传入的闭包函数 f 依次完成各组 Key-Value 值的接收. f 返回值是 bool 型，若为 false 则遍历行为终止.
```
template <class Key,class Value>
class Map: base::Noncopyable{
public:
    /**
     * 构造、析构函数
     */
    // 构造函数. 初始化 readonly 实例
    Map();
    // 析构函数. 回收 readonly 实例
    ~Map();
    // ...
public:
    /**
     * 公有方法
     */
    /**
     * @brief：读取数据，存储到 receiver 中
     * @param: key——数据键
     * @param: receiver——接收数据的容器
     * @return: true——数据存在，false——数据不存在
     */
    bool load(Key key, Value& receiver);

    /**
     * @brief: 写入数据
     * @param: key——数据键
     * @param: value——拟写入的数据
     */
    void store(Key key, Value value);

    /**
     * @brief: 删除数据
     * @param: key——数据键
     */
    void evict(Key key);

    /**
     * @brief: 遍历 map
     * @param：f——用于接收 key-value 对的闭包函数
     *            key——数据键  value——数据
     */
    void range(std::function<bool(const Key& key, const Value& value)> f);

private:
    /**
     * 私有方法
     */
    /**
     * @brief: 累计 readonly 数据 miss 次数
     * 当 miss 次数 >= dirty 大小时，需要执行 readonlyLocked 流程，使用 dirty 覆盖 readonly，并清空 dirty
     * 1) miss++  2) dirty -> readonly
     */
    void missLocked();

    /**
     * @brief：将 dirty 赋给 readonly，然后清空 dirty
     */
    void readonlyLocked();

    /**
     * @brief: 倘若 dirty 为空，需要遍历 readonly 将未删除的数据转移到 dirty 中
     * 遍历 readonly 时间复杂度 O(N)，同时过滤掉所有处于删除态（expunged or nullptr）的数据
     */
    void dirtyLocked();

    // ...
};
```

此外，还有些 sync.Map 内部闭环的私有方法：

- •
	```
	missLocked
	```
	：每当访问击穿 readonly 直击到 dirty 时，需要执行 missLocked 流程. 该流程会累加 m\_misses 计数器，当 m\_misses >= dirty.size 时，会执行正向迁移的 readonlyLocked 流程；
- •
	```
	readonlyLocked
	```
	：使用 dirty 对 readonly 进行覆盖，然后构造一个空白的 dirty 实例；
- •
	```
	dirtyLocked
	```
	：执行完 readonlyLocked 后，下一次直击 dirty 的插入写行为会触发 dirtyLocked 流程. 此时需要遍历 readonly 中的 Key-Entry，一方面将所有软删除态（nullptr）Entry 更新为硬删除态（expunged），另一方面将非删除态 Entry 拷贝到 dirty 中.
```
namespace cbricks{namespace sync{

/**
 * @brief: 并发安全的 map 工具——借鉴 golang sync.Map 实现
 * 适用场景：适用于 插入类写操作较少的 场景
 * 功能点：底层基于以空间换时间的思路，建立 read dirty 两份 map
 *        - readonly：应用于读、更新、删除场景，实现无锁化
 *        - dirty：包含全量数据，访问时需要加锁
 */
template<class Key,class Value>
class Map: base::Noncopyable{
    // ...
public:
    typedef std::unordered_map<Key,typename Entry::ptr>map;
private:
    /**
     * map 中依赖的内置私有类型
     * 由于内部仅有一个 bool 及共享指针字段，所以发生值拷贝的成本可忽略不计
     */
    // 只读 map 数据类型
    struct ReadOnly{
        // 构造函数
        ReadOnly();
        // 拷贝构造函数
        ReadOnly(constReadOnly& other);
        // 赋值函数
        ReadOnly&operator=(constReadOnly& other);
        /**
         * 标识 readonly 相比 dirty 是否存在数据缺失
         * 1）true——存在缺失，则操作 readonly 时若发现数据 miss，还需要读取 dirty 兜底
         * 2）false——不存在缺少. 则操作 readonly 时若发现数据 miss，无需额外操作 dirty
         */
        volatile bool amended;
        // 存储 readonly 中 kv 数据的 map. value 为 entry 共享指针类型，可供多个 readonly 实例共享
        std::shared_ptr<map> m;
    };
    // ...
};
```

readonly 是 sync.Map 中的私有嵌套类，其包含两个成员变量：

- •
	```
	amended
	```
	：标识 dirty 中是否存在 readonly 中缺失的 Key，即是否存在绕开 readonly 直击到 dirty 的插入写操作. （若此标识为 false，则说明 readonly 拥有全量数据，那么此时读操作可以由 readonly 完全闭环. ）
- •
	```
	m
	```
	：readonly 中存储 Key-Entry 的 unordered\_map 实例（通过智能指针包装），只面向无锁化的读访问行为
```
namespace cbricks{namespace sync{

/**
 * @brief: 并发安全的 map 工具——借鉴 golang sync.Map 实现
 * 适用场景：适用于 插入类写操作较少的 场景
 * 功能点：底层基于以空间换时间的思路，建立 read dirty 两份 map
 *        - readonly：应用于读、更新、删除场景，实现无锁化
 *        - dirty：包含全量数据，访问时需要加锁
 */
template<class Key,class Value>
class Map: base::Noncopyable{
    // ...
private:
    /**
     * map 中依赖的内置私有类型
     */
    // 底层存放 value 数据的容器
    struct Entry{
    public:
        // 智能指针 类型别名
        typedef std::shared_ptr<Entry> ptr;

    public:
        /**
         * 构造 析构函数
         */
        // 构造函数，入参为存储数据
        Entry(Value v);
        // 析构函数
        ~Entry();

    public:
        /**
         * entry 内置类
         */
        // 对存储数据使用该类型的指针进行封装
        struct WrappedV{
        public:
            WrappedV()=default;
            WrappedV(Value v);
        public:
            Value v;
        };

    public:
        /**
         * 读操作相关方法
         */
        /**
         * @brief: 从 entry 中读取数据，存储到 receiver 中
         * @param: 承载读取数据的容器
         * @return: true——读取成功 false——读取失败
         */
        bool load(Value& receiver);

    public:
        /**
         * 写操作相关
         */
        /**
         * @brief: 将数据存储在 entry 的原子容器中 [调用此方法时一定处于持有 dirtyLock 状态]
         * @param: v——拟存储的数据
         */
        void storeLocked(Value v);

        /**
         * @brief: 尝试将数据存储在 entry 的原子容器中
         * @param: v——尝试存储的数据
         * @return: true——存储成功 false——存储失败（硬删除态 expunged 时存储失败）
         */
        bool tryStore(Value v);

        // 删除 entry 中存储的数据，将 container 中内容置为 nullptr（软删除态）
        void evict();

    public:
        /**
         * entry 状态更新操作相关方法
         */
        /**
         * @brief: 将 entry 置为非硬删除态
         * @return: true——之前处于硬删除态（expunged），更新成功；false——之前处于非硬删除态，无需更新
         */
        bool unexpungeLocked();
        /**
         * @brief: 尝试将 entry 置为硬删除态（expunged）
         * @return: true——之前就是硬删除态，或者成功将 nil （软删除态）置为硬删除态；false——更新失败（此前数据未删除）
         */
        bool tryExpungeLocked();

    public:
        /** 
         * 存放数据的 atomic 容器 
         * WrappedV 分为三种可能：
         * 1）s_expunged：硬删除态，表示 dirty 中一定没有该 entry，要恢复数据必须对 dirty 执行 insert 操作
         * 2）nullptr：软删除态. 数据逻辑意义上已被删除，但是 dirty 中还有该数据，还能快速恢复过来
         * 3）其他：数据仍存在，未被删除
        */
        std::atomic<WrappedV*> container;
    };
    // ...
};
```

Entry 也是 sync.Map 中的私有嵌套类，它是用于存储 Value 的容器，无论 readonly 还是 dirty 持有的 unordered\_map 实例，其对应值类型都是 Entry（共享指针包装）.

在 Entry 中，通过原子容器 atomic<WrappedV\*> 实现对 Value 的存储，WrappedV 是 Entry 中的嵌套类，用于将 Value 包装成指针类型.

Entry 中声明了一系列成员方法，只能由 sync.Map 内部访问：

- •
	```
	load
	```
	：从 Entry 中读取 Value 并加载到给定容器 receiver 中. 如果 Entry 已处于删除态（nullptr/expunged），则读取失败
- •
	```
	storeLocked
	```
	：将 Value 存储到 Entry 容器中. store 操作必定成功（调用此方法时，一定已持有 dirty 的互斥锁）
- •
	```
	tryStore
	```
	：尝试将 Value 存储到 Entry 中. 只有在 Entry 已处于硬删除态（expunged）时，可能导致存储失败
- •
	```
	evict
	```
	：将 Entry 置为软删除态（nullptr）
- •
	```
	unexpungeLocked
	```
	：将 Entry 由硬删除态（expunged）更新为软删除态（nullptr）. 若此前 Entry 不处于硬删除态（expunged），则该操作失败（调用此方法时，一定已持有 dirty 的互斥锁）
- •
	```
	tryExpungeLocked
	```
	：将 Entry 升级为硬删除态（expunged）. 若此前 Entry 不处于删除态，则该操作失败（调用此方法时，一定已持有 dirty 的互斥锁）

### 2.3 构造&析构函数

```
// 全局静态变量，标记硬删除状态.
template<class Key,class Value>
typename Map<Key,Value>::Entry::WrappedV*Map<Key,Value>::s_expunged =new typename Map<Key,Value>::Entry::WrappedV();

// 构造函数，初始化 readonly 实例
template<class Key,class Value>
Map<Key,Value>::Map(){
    this->m_readonly.store(new typename Map<Key,Value>::ReadOnly);
}

// 析构函数，回收 readonly 实例
template<class Key,class Value>
Map<Key,Value>::~Map(){
    typename Map<Key,Value>::ReadOnly* readonly =this->m_readonly.load();
    delete readonly;
}
```

在 sync.Map 的构造函数与析构函数中，核心工作是

```
完成对 readonly 实例生命周期的管理
```
，包含构造函数中的的初始化（new）以及析构函数中的回收（delete）操作：

> 此处之所以使用原始指针，是因为 readonly 需要存放在 atomic 容器中，而在 c++ 11 版本中，原子容器 atomic 并不能很好地与智能指针 shared\_ptr 兼容使用，因此采用原始指针实属迫于无奈. 后续关于 Entry 中 WrappedV 原始指针的使用也是出于同样的限制.

```
// readonly 默认构造函数
template<class Key, class Value>
Map<Key,Value>::ReadOnly::ReadOnly():amended(false){
    this->m.reset(new map);
}
```

readonly 实例的生命周期与 sync.Map 是一一对应的. 在 readonly 构造函数中，需要完成对 unordered\_map 实例的初始化，并默认将 amended 标识置为 false.

> 此处使用 shared\_ptr 对 unordered\_map 实例的指针进行托管，是为了防止因 readonly 实例的拷贝或者赋值行为导致引起 unordered\_map 数据的拷贝行为.

由于 readonly 中应用到 shared\_ptr 技术，因此无需在析构函数对 unordered\_map 实例执行回收操作.

```
// entry 构造函数，入参为存储数据对应智能指针
template<class Key,class Value>
Map<Key,Value>::Entry::Entry(Value v){
    typename Map<Key,Value>::Entry::WrappedV* wv =new WrappedV(v);
    this->container.store(wv);
}

// entry 析构函数
template<class Key,class Value>
Map<Key,Value>::Entry::~Entry(){
    typename Map<Key,Value>::Entry::WrappedV* wv =this->container.load();
    if(wv ==nullptr|| wv ==Map::s_expunged){
        return;
    }
    delete wv;
}
```

在 Entry 的构造函数中，需要完成对 Value 包装类 WrappedV 实例的初始化（new），并将其存储到 atomic 容器中；

在 Entry 析构时，需要保证对不处于删除态（nullptr/expunged）的 WrappedV 实例进行内存释放（delete）.

### 2.4 读流程实现

![图片](data:image/svg+xml,%3C%3Fxml version='1.0' encoding='UTF-8'%3F%3E%3Csvg width='1px' height='1px' viewBox='0 0 1 1' version='1.1' xmlns='http://www.w3.org/2000/svg' xmlns:xlink='http://www.w3.org/1999/xlink'%3E%3Ctitle%3E%3C/title%3E%3Cg stroke='none' stroke-width='1' fill='none' fill-rule='evenodd' fill-opacity='0'%3E%3Cg transform='translate(-249.000000, -126.000000)' fill='%23FFFFFF'%3E%3Crect x='249' y='126' width='1' height='1'%3E%3C/rect%3E%3C/g%3E%3C/g%3E%3C/svg%3E)

```
/**
 * @brief：读取数据
 * @param: key——数据键
 * @param: receiver——接收数据的容器
 * @return: true——数据存在，false——数据不存在
 */
template<class Key,class Value>
boolMap<Key,Value>::load(Key key,Value& receiver){
/**
     * 核心处理流程：
     * 1）尝试从 readonly 中读取数据，若存在直接返回
     * 2）若 readonly.amended 为 false，代表 readonly 已有全量数据，也直接返回
     * 3）加锁
     * 4）再次从 readonly 中读取数据，若存在直接返回 (并发场景下的 double check)
     * 5）从 dirty 中读取数据，并且执行 missLocked 流程
     */

    // 尝试从 readonly 中读取数据，若存在直接返回
    typename Map<Key,Value>::Entry::ptr e =nullptr;

    // 从 atomic 容器中读取 readonly 实例指针，并执行拷贝构造函数，生成 readonly 实例
    typename Map<Key,Value>::ReadOnly readonly =*this->m_readonly.load();
    auto it = readonly.m->find(key);
    if(it != readonly.m->end()){
        e = it->second;
    }

    if(e !=nullptr){
        return e->load(receiver);
    }

    // readonly.amended 为 false，表示 readonly 已包含全量数据，无需再读 dirty 了
    if(!readonly.amended){
        return false;
    }

    // readonly 中数据不存在，且 amended 为 true，则需要加锁读 dirty map
    // 加锁
    Lock::lockGuard guard(this->m_dirtyLock);

    // 加锁后对 readonly double check
    // 此处执行的是 readonly 赋值操作函数
    readonly =*this->m_readonly.load();
    it = readonly.m->find(key);
    if(it != readonly.m->end()){
        e = it->second;
    }

    if(e !=nullptr){
        return e->load(receiver);
    }

    if(!readonly.amended){
        return false;
    }

    // readonly 中数据不存在且 amended 为 true，读 dirty
    it =this->m_dirty.find(key);
    if(it !=this->m_dirty.end()){
        e = it->second;
    }

    // 执行 missLocked 流程. 其中可能因为 miss 次数过多，而发生 dirty->readonly 的更新
    this->missLocked();

    // 若 readonly 和 dirty 中都不存在该 key，直接返回 false
    if(e ==nullptr){
        return false;
    }

    // 尝试从 dirty 获取的 entry 中读取结果
    return e->load(receiver);
}
```

用于从 sync.Map 中

```
读取数据
```
的
```
load 方法
```
中，分为如下核心步骤：

- • 尝试
	```
	从 readonly 获取 Key 对应 Entry
	```
- • 若 readonly 中 Key 不存在，但是 readonly 中
	```
	amended 标识为 false，说明 readonly 中已包含全量 Key
	```
	，则直接返回数据不存在
- •
	```
	加 dirtyLock
	```
	，通过
	```
	lock guard（guard 实例 + RAII（resource acquisition is initialization））
	```
	技术保证在方法退出前执行解锁操作
- • 由于是并发场景，加锁后需要
	```
	double check
	```
	，再次尝试从 readonly 中读数据
- • 若 readonly 还是没有数据，则
	```
	从 dirty 中读数据
	```
	，并且因为 readonly 被击穿，需要
	```
	执行 missLocked 流程
	```
- • 在上述任意步骤中，若读取到 Key 对应 Entry，则
	```
	调用 Entry.load
	```
	方法返回读取到的数据.

> 需要注意，哪怕 Key 对应 Entry 存在，也不代表数据一定存在，因为 Entry 还可能处于删除态（nullptr/expunged）

```
/**
 * 读操作相关
 */
/**
 * @brief: 从 entry 中读取数据，存储到 receiver 中
 * @param: 承载读取数据的容器
 * @return: true——读取成功 false——读取失败
 */
template<class Key,class Value>
bool Map<Key,Value>::Entry::load(Value& receiver){
    typename Map<Key,Value>::Entry::WrappedV* wv =this->container.load();
    if(wv ==nullptr|| wv ==Map::s_expunged){
        return false;
    }

    receiver = wv->v;
    return true;
}
```

在 Entry.load 方法中，会从 atomic 容器中读取出 WrappedV 实例的指针，如果对应为删除态（nullptr/expunged），则返回 Value 不存在；否则使用传入的 receiver 接收对应的 Value.

下面是当一次访问操作击穿 readonly 后，需要执行的 missLocked 流程：

![图片](data:image/svg+xml,%3C%3Fxml version='1.0' encoding='UTF-8'%3F%3E%3Csvg width='1px' height='1px' viewBox='0 0 1 1' version='1.1' xmlns='http://www.w3.org/2000/svg' xmlns:xlink='http://www.w3.org/1999/xlink'%3E%3Ctitle%3E%3C/title%3E%3Cg stroke='none' stroke-width='1' fill='none' fill-rule='evenodd' fill-opacity='0'%3E%3Cg transform='translate(-249.000000, -126.000000)' fill='%23FFFFFF'%3E%3Crect x='249' y='126' width='1' height='1'%3E%3C/rect%3E%3C/g%3E%3C/g%3E%3C/svg%3E)

```
/**
 * @brief: 累计 read map 数据 miss 次数. 访问该方法时一定持有 dirtyLock
 * 当 miss 次数 >= dirty 大小时，需要使用 dirty 覆盖 read
 * 1) miss++  2) dirty -> read
 */
template<class Key,class Value>
void Map<Key,Value>::missLocked(){
    this->m_misses++;
    // 由于持有 dirtyLock，所以允许访问 dirty 的 size 方法
    if(this->m_misses.load()<this->m_dirty.size()){
        return;
    }
    // miss 次数过多，需要使用 dirty 覆盖 readonly，并清空原 dirty
    this->readonlyLocked();
}
```

在 missLocked 方法中：

- • 递增 m\_misses 原子计数器，标识新增一次 readonly 击穿行为
- • 若 m\_misses >= dirty.size，则需要
	```
	执行 readonlyLocked
	```
	对应的数据
	```
	正向迁移流程
	```
```
/**
 * @brief：将 dirty 赋给 readonly，然后清空 dirty
 */
template<class Key,class Value>
void Map<Key,Value>::readonlyLocked(){
    // 从原子容器中获取 readonly 实例指针
    typename Map<Key,Value>::ReadOnly* readonly =this->m_readonly.load();
    // 将 dirty 通过移动函数填充到 readonly 中（通过 std::move 操作避免发生拷贝）
    readonly->m = std::make_shared<map>(std::move(this->m_dirty));
    // amended 标识置为 false
    readonly->amended =false;

    // 清空 readonly miss 计数器
    this->m_misses.store(0);
    // 构造新的 dirty 实例
    this->m_dirty =map();
}
```

在 readonlyLocked 方法中：

- • 使用 dirty 覆盖 readonly 中的 m（使用
	```
	移动函数 std::move 实现 dirty 的右值传递
	```
	，避免发生拷贝行为）
- • 将 readonly 的 amended 标识置为 false
- • 将 m\_misses 计数器清 0
- • 构造一个新的空白的 dirty 实例

### 2.5 写流程实现

![图片](data:image/svg+xml,%3C%3Fxml version='1.0' encoding='UTF-8'%3F%3E%3Csvg width='1px' height='1px' viewBox='0 0 1 1' version='1.1' xmlns='http://www.w3.org/2000/svg' xmlns:xlink='http://www.w3.org/1999/xlink'%3E%3Ctitle%3E%3C/title%3E%3Cg stroke='none' stroke-width='1' fill='none' fill-rule='evenodd' fill-opacity='0'%3E%3Cg transform='translate(-249.000000, -126.000000)' fill='%23FFFFFF'%3E%3Crect x='249' y='126' width='1' height='1'%3E%3C/rect%3E%3C/g%3E%3C/g%3E%3C/svg%3E)

```
/**
 * @brief: 写入数据
 * @param: key——数据键
 * @param: value——拟写入的数据
 */
template<class Key,class Value>
void Map<Key,Value>::store(Key key,Value value){
    /**
     * 1）尝试从 readonly 中读取 key 对应 entry
     * 2）若 readonly 中存在该 entry，则尝试通过 cas 操作更新 entry 为新值，前提是 entry 不为硬删除态（expunged）
     * 3）若 2）cas 成功，直接返回
     * 4）加锁
     * 5）再次尝试从 readonly 中读取 key 对应 entry （并发场景下的 double check），若不存在，进入 8）
     * 6）确保将 entry 更新为非硬删除态，若此前为硬删除态（expunged），需要在 dirty 中补充该 entry
     * 7）向 entry 中存储新值，并返回
     * 8）尝试从 dirty 中读取 key 对应 entry，若存在，往 entry 中存储新值并返回
     * 9）此时注定要往 dirty 中插入新数据，若 readonly.amended 为 false，需要将其更新为 true
     * 10）若此时 dirty 为空，需要执行 dirtyLocked 流程，将 readonly 中所有非硬删除态的数据拷贝到 dirty 中
     * 11）往 dirty 中插入新的 entry 并返回
     */

    // 尝试从 readonly 中读取 key 对应 entry
    typename Map<Key,Value>::Entry::ptr e =nullptr;
    // 从 atomic 容器中读取 readonly 实例指针，并执行拷贝构造函数，生成 readonly 实例
    typename Map<Key,Value>::ReadOnly readonly =*this->m_readonly.load();
    auto it = readonly.m->find(key);
    if(it != readonly.m->end()){
        e = it->second;
    }

    // 尝试通过 cas 操作完成 value 的更新，前提是 entry 不为 expunged 状态
    if(e !=nullptr&& e->tryStore(value)){
        return;
    }

    // 加锁
    Lock::lockGuard guard(this->m_dirtyLock);

    // 加锁后对 readonly double check
    // 执行 readonly 赋值函数
    readonly =*this->m_readonly.load();
    it = readonly.m->find(key);
    if(it != readonly.m->end()){
        e = it->second;
    }

    // 若 entry 在 readonly 中存在，直接更新 value
    if(e !=nullptr){
        // 尝试将 e 由硬删除态（expunuged）置为非硬删除态（nullptr）
        if(e->unexpungeLocked()){
            // 若返回 true 表示此前为硬删除态，需要在 dirty 中补充该数据
            this->m_dirty.insert({key,e});
        }
        // 将 entry 中的内容更新为 value 并返回
        e->storeLocked(value);
        return;
    }

    /**
     * 至此，已确定 readonly 中不存在数据
     */
    // 尝试从 dirty 中读取数据
    it =this->m_dirty.find(key);
    if(it !=this->m_dirty.end()){
        e = it->second;
    }

    // 若 dirty 中存在该 entry, 直接更新即可（dirty 中存在 entry 时，不可能为硬删除态（expunged））
    if(e !=nullptr){
        e->storeLocked(value);
        return;
    }

    /**
     * 至此已明确 readonly 和 dirty 中均不存在该 key，需要向 dirty 中插入新数据，此前需要执行一些准备工作
     * 1）若 readonly.amended 为 false，需要将其更新为 true
     * 2）若此前 dirty 为空，需要将 readonly 中所有非硬删除态的数据拷贝到 dirty（dirtyLocked 流程）
     */
    if(!readonly.amended){
        // atomic 容器中，readonly 实例的 amended 更新为 true
        this->m_readonly.load()->amended =true;
        this->dirtyLocked();
    }

    // 向 dirty 中插入新数据对应的 entry
    this->m_dirty.insert({key,typename Map<Key,Value>::Entry::ptr(new Entry(value))});
}
```

在向 sync.Map 中

```
写入数据
```
的
```
store 方法
```
中，包含如下核心步骤：

- • 尝试
	```
	从 readonly 获取 Key 对应 Entry
	```
	. 若 Entry 存在则
	```
	调用 Entry.tryStore
	```
	方法，尝试通过 CAS 完成数据的更新（若 Entry 为硬删除态（expunged），则 tryStore 操作失败）；
- •
	```
	加 dirtyLock
	```
	，并通过 lock guard 技术保证在方法退出前执行解锁操作；
- • 由于是并发场景，所以加锁后需要
	```
	double check
	```
	，再次尝试从 readonly 中获取 Key 对应 Entry；
- • 若加锁后从 readonly 中获取到 Entry，则通过
	```
	Entry.unExpungeLocked
	```
	方法，将 Entry 更新为软删除态（nullptr），并确保
	```
	在 dirty 中补齐该组 Key-Entry 数据
	```
	；
- • 若 readonly 中还是没有数据，则
	```
	从 dirty 中读取 Entry
	```
	；
- • 若 Entry 存在，则
	```
	调用 Entry.storeLocked 方法
	```
	，将 Value 更新到 Entry 中；
- • 若 Entry 不存在，则
	```
	对 dirty 执行 insert 插入写操作
	```
	；此时还需判断
	```
	readonly 中 amended 标识是否为 false
	```
	，若是则更新为 true，并
	```
	执行 dirtyLocked 流程
	```
```
/**
 * @brief: 尝试存储数据
 * @param: v——尝试存储数据的智能指针
 * @return: true——存储成功 false——存储失败（仅当处于硬删除态时失败）
 */
template<class Key,class Value>
bool Map<Key,Value>::Entry::tryStore(Value v){
    typename Map<Key,Value>::Entry::WrappedV* _new =new WrappedV(v);
    while(true){
        typename Map<Key,Value>::Entry::WrappedV* old =this->container.load();
        // 若此前已处于硬删除（expunged），则更新失败
        if(old ==Map::s_expunged){
            delete _new;
            return false;
        }

        if(this->container.compare_exchange_strong(old,_new)){
            // 更新成功，需要删除原本的 wrappedV 实例
            if(old !=nullptr){
                delete old;
            }
            return true;
        }
    }
}
```

在写流程中，无锁访问 readonly 并发现 Entry 存在时，会调用 Entry.tryStore 方法尝试通过 CAS 操作完成 Value 的更新.

> 若此前 Entry 为硬删除态（expunged），则更新失败；若此前 Entry 不为删除态，则需要对旧 Value 对应的 WrappedV 实例进行内存回收（delete）

```
/**
 * entry 状态更新操作相关方法
 */
/**
 * @brief: 将 entry 置为非硬删除态
 * @return: true——之前处于硬删除态（expunged），更新成功；false——之前处于非硬删除态，无需更新
 */
template<class Key, class Value>
bool Map<Key,Value>::Entry::unexpungeLocked(){
    return this->container.compare_exchange_strong(Map::s_expunged,nullptr);
}
```

在加锁后 double check 时，若发现 readonly 中存在 Key-Entry，则需要执行 Entry.unexpungedLocked 方法，确保将原本处于硬删除态（expunged）的 Entry 更新为软删除态（nullptr）

> 此时对应 Entry 可能处于硬删除态（expunged），也可能是在执行 tryStore 到加 dirtyLock 的间隙内被并发写入

```
/**
 * entry 写操作相关
 */
/**
 * @brief: 存储数据 [调用此方法时一定处于持有锁状态]
 * @param: v——存储数据的智能指针
 * 进入该方法时，container 中的 old value 不可能为 expunged 态
 */
template<class Key,class Value>
void Map<Key,Value>::Entry::storeLocked(Value v){
    typename Map<Key,Value>::Entry::WrappedV* _new =new WrappedV(v);
    while(true){
        typename Map<Key,Value>::Entry::WrappedV* old =this->container.load();
        if(this->container.compare_exchange_strong(old,_new)){
            if(old !=nullptr){
                delete old;
            }
            return;
        }
    }
}
```

当加锁，发现 Key 对应 Entry 存在时，会调用 Entry.storeLocked 方法，完成 Value 的更新.

> 在 storeLocked 方法中，会基于 CAS 操作将 Entry 更新为新的 Value，且对原本 Value 封装实例 WrappedV 进行内存回收（delete）.

![图片](data:image/svg+xml,%3C%3Fxml version='1.0' encoding='UTF-8'%3F%3E%3Csvg width='1px' height='1px' viewBox='0 0 1 1' version='1.1' xmlns='http://www.w3.org/2000/svg' xmlns:xlink='http://www.w3.org/1999/xlink'%3E%3Ctitle%3E%3C/title%3E%3Cg stroke='none' stroke-width='1' fill='none' fill-rule='evenodd' fill-opacity='0'%3E%3Cg transform='translate(-249.000000, -126.000000)' fill='%23FFFFFF'%3E%3Crect x='249' y='126' width='1' height='1'%3E%3C/rect%3E%3C/g%3E%3C/g%3E%3C/svg%3E)

```
/**
 * @brief: 倘若 dirty 为空，需要将 read 全量数据转移到 dirty 中
 * 遍历 read 时间复杂度 O(N)，同时清理掉处于硬删除态（expunged）的数据
 */
template<class Key,class Value>
void Map<Key,Value>::dirtyLocked(){
    if(!this->m_dirty.empty()){
        return;
    }

    // 获取 readonly 引用
    // 从 atomic 中获取 readonly 实例，并通过拷贝构造函数生成副本
    typename Map<Key,Value>::ReadOnly readonly =*this->m_readonly.load();
    // 获取 map 引用，不发生拷贝行为
    typename Map<Key,Value>::map& m =*readonly.m;
    this->m_dirty.reserve(m.size());
    /**
     * 遍历 map O(N)
     * 1）将软删除态 nullptr -> 硬删除态 expunged
     * 2）非删除态数据拷贝到 dirty 中
     */
    for(auto it : m){
        // 其中会将 readonly 中，原本为软删除态 nullptr 的 entry 置为硬删除态 expunged
        // 无论此前是软删除态还是硬删除态，都需要过滤该 entry，不添加到 dirty 中
        if(it.second->tryExpungeLocked()){
            continue;
        }
        // 对于 readonly 中未删除的数据，插入到 dirty 中
        this->m_dirty.insert({it.first,it.second});
    }
}

/**
 * @brief: 尝试将 entry 置为硬删除态（expunged）
 * @return: true——之前就是硬删除态，或者成功将 nil （软删除态）置为硬删除态；false——更新失败
 */
template<class Key,class Value>
bool Map<Key,Value>::Entry::tryExpungeLocked(){
    typename Map<Key,Value>::Entry::WrappedV* wv =this->container.load();
    while(wv ==nullptr){
        if(this->container.compare_exchange_strong(wv,Map::s_expunged)){
            return true;
        }
        wv =this->container.load();
    }

    return wv ==Map::s_expunged;
}
```

当需要对 dirty 执行一次插入写操作，且发现此时 readonly 中 amended 标识为 false 时，需要执行 dirtyLocked 流程：

- • 若 dirty 非空，直接返回；
- • 遍历 readonly 中的 Key-Entry 数据；
- • 将所有处于软删除态（nullptr）的 Entry 更新为硬删除态（expunged）；
- • 将所有非删除态的 Key-Entry 拷贝到 dirty 中（延迟删除处理）.

### 2.6 删流程流程

![图片](data:image/svg+xml,%3C%3Fxml version='1.0' encoding='UTF-8'%3F%3E%3Csvg width='1px' height='1px' viewBox='0 0 1 1' version='1.1' xmlns='http://www.w3.org/2000/svg' xmlns:xlink='http://www.w3.org/1999/xlink'%3E%3Ctitle%3E%3C/title%3E%3Cg stroke='none' stroke-width='1' fill='none' fill-rule='evenodd' fill-opacity='0'%3E%3Cg transform='translate(-249.000000, -126.000000)' fill='%23FFFFFF'%3E%3Crect x='249' y='126' width='1' height='1'%3E%3C/rect%3E%3C/g%3E%3C/g%3E%3C/svg%3E)

```
/**
 * @brief: 删除数据
 * @param: key——数据键
 */
template<class Key,class Value>
void Map<Key,Value>::evict(Key key){
    /**
     * 1）尝试从 readonly 中获取 key 对应 entry，若存在，则直接将 entry 中的内容置为 nullptr（软删除态）
     * 2）若 readonly 中 key 不存在，且 readonly.amended 为 false，则直接返回
     * 3）加锁
     * 4）再次执行一次步骤 1） 相同动作（并发场景下的 double check）
     * 5）若确定 readonly 中 key 不存在，从 dirty 中删除 key，并且执行 missLocked 流程
     */

    // 尝试从 readonly 中获取 key 对应 entry
    typename Map<Key,Value>::Entry::ptr e =nullptr;
    // 获取 readonly 引用，执行拷贝构造函数
    typename Map<Key,Value>::ReadOnly readonly =*this->m_readonly.load();
    auto it = readonly.m->find(key);
    if(it != readonly.m->end()){
        e = it->second;
    }

    // 如果 readonly 中 entry 存在，直接将 entry 中内容置为 nullptr 软删除态，并返回
    if(e !=nullptr){
        e->evict();
        return;
    }

    // 若 readonly.amended 为 false，说明 dirty 中也不可能存在数据，直接返回
    if(!readonly.amended){
        return;
    }

    // 加锁
    Lock::lockGuard guard(this->m_dirtyLock);

    // 针对 readonly 进行 double check
    // 获取 readonly 引用，执行赋值函数
    readonly =*this->m_readonly.load();
    it = readonly.m->find(key);
    if(it != readonly.m->end()){
        e = it->second;
    }

    if(e !=nullptr){
        e->evict();
        return;
    }

    if(!readonly.amended){
        return;
    }

    //  readonly 数据 miss，并且 amended 为 false，则需要从 dirty 中删除数据
    this->m_dirty.erase(key);
    // 执行 missLocked 流程
    this->missLocked();
}
```

从 sync.Map 中

```
删除数据
```
的流程，调用的是
```
evict 方法
```
，核心步骤包括：

- • 尝试
	```
	从 readonly 获取 Key 对应 Entry
	```
	. 若 Entry 存在，
	```
	调用 Entry.evict
	```
	方法尝试通过 CAS 将对应 Entry 置为软删除态（nullptr）
- •
	```
	加 dirtyLock
	```
	，并通过 lock guard 技术保证在方法退出前执行解锁操作
- • 由于是并发场景，加锁后需要
	```
	double check
	```
	，再次尝试从 readonly 中获取 Key 对应 Entry 并执行 Entry.evict 方法
- • 若 readonly 中没有数据，且
	```
	readonly amended 标识为 false
	```
	，
	```
	代表 dirty 中也不可能存在数据
	```
	，直接返回
- •
	```
	对 dirty 执行 erase
	```
	擦除对应 Key-Entry
- • 由于 readonly 被击穿了，所以需要
	```
	调用 missLocked 流程
	```
```
// 删除 entry 中存储的数据
template<class Key,class Value>
void Map<Key,Value>::Entry::evict(){
    while(true){
        typename Map<Key,Value>::Entry::WrappedV* old =this->container.load();
        if( old ==nullptr|| old ==Map::s_expunged ){
            return;
        }
        if(this->container.compare_exchange_strong(old,nullptr)){
            delete old;
            return;
        }
    }
}
```

在 Entry.evict 方法中，通过 CAS 操作将一个非删除态的 Entry 置为软删除态（nullptr），并且需要对 WrappedV 实例进行内存回收（delete）.

### 2.7 遍历流程流程

![图片](data:image/svg+xml,%3C%3Fxml version='1.0' encoding='UTF-8'%3F%3E%3Csvg width='1px' height='1px' viewBox='0 0 1 1' version='1.1' xmlns='http://www.w3.org/2000/svg' xmlns:xlink='http://www.w3.org/1999/xlink'%3E%3Ctitle%3E%3C/title%3E%3Cg stroke='none' stroke-width='1' fill='none' fill-rule='evenodd' fill-opacity='0'%3E%3Cg transform='translate(-249.000000, -126.000000)' fill='%23FFFFFF'%3E%3Crect x='249' y='126' width='1' height='1'%3E%3C/rect%3E%3C/g%3E%3C/g%3E%3C/svg%3E)

```
/**
 * @brief: 遍历 map
 * @param：f——用于接收 key-value 对的闭包函数
 *            key——数据键  
 *            value——数据
 */
template<class Key,class Value>
void Map<Key,Value>::range(std::function<bool(const Key& key,const Value& value)> f){
    /**
     * 1）获取 readonly
     * 2）若 readonly.amended 为 true，加锁，然后将 dirty 中所有数据转移到 readonly 中
     * 3）遍历 readonly，依次执行闭包函数
     */
    typename Map<Key,Value>::ReadOnly readonly =*this->m_readonly.load();
    if(readonly.amended){
        Lock::lockGuard guard(this->m_dirtyLock);
        readonly =*this->m_readonly.load();
        if(readonly.amended){
            this->readonlyLocked();
        }
    }

    // 遍历 readonly
    readonly =*this->m_readonly.load();
    typename Map<Key,Value>::map& m =*readonly.m;
    for(auto it : m){
        Value v;
        if(!it.second->load(v)){
            continue;
        }
        if(!f(it.first,v)){
            break;
        }
    }
}
```

通过

```
range 方法
```
实现对 sync.Map 的
```
遍历操作
```
，使用方需要通过传入闭包函数 f，来依次接收每组被遍历访问到的 Key-Value 数据. range 方法的执行步骤包括：

- • 若 readonly 中
	```
	amended 标识为 true
	```
	，则需要
	```
	加 dirtyLock 并 double check
	```
	，然后
	```
	执行 readonlyLocked 流程
	```
	补齐 readonly 中缺失的数据；
- •
	```
	遍历 readonly 中的 m
	```
	，过滤掉删除态的 Key-Entry 后，
	```
	调用闭包函数 f
	```
	，并根据闭包函数执行结果判断是否需要终止遍历行为

## 3 使用示例

最后，我们展示一下对 2 章中实现的 sync.Map 工具的使用示例，其中涉及到在并发场景下对 load、store、evict、range 方法的使用：

```
#include <iostream>

#include "sync/thread.h"
#include "sync/sem.h"
#include "sync/map.h"

// sync.Map 测试用例
void testSyncMap(){
    // sync.Map 类型别名
    typedef cbricks::sync::Map<int,int> smap;
    // 信号量 类型别名
    typedef cbricks::sync::Semaphore semaphore;
    // 线程 类型别名
    typedef cbricks::sync::Thread thread;

    // 构造 sync.Map 实例
    smap sm;
    // 构造信号量实例
    semaphore sem;

    // 并发执行 cnt1 次写操作、cnt1 次读操作
    int cnt1 =1000;
    // 并发执行 cnt2 次删操作
    int cnt2 =500;

    // 使用原子计数器，控制 sync.Map 中的 key value值
    std::atomic<int> flag1{0};
    std::atomic<int> flag2{0};
    std::atomic<int> flag3{0};

    // 并发调用 store  load 方法
    for(int i =0; i < cnt1; i++){
        // 并发写 
        thread thr1([&sm,&sem,&flag1](){
            int index = (flag1++)/10;
            sm.store(index,index);
            sem.notify();
        });

        // 并发读
        thread thr2([&sm,&sem,&flag2](){
            int index = (flag2++)/10;
            int v;
            sm.load(index,v);
            sem.notify();            
        });
    }

    // 并发调用 evict 方法
    for(int i =0; i < cnt2; i++){
    // 并发删
        thread thr3([&sm,&sem,&flag3](){
            int index = (flag3++)/10;
            sm.evict(index);
            sem.notify();            
        });
    }

    // 通过信号量，确保上述并发流程执行完成
    for(int i =0; i <2* cnt1 + cnt2; i++){
        sem.wait();
    }

    // 遍历 sync.Map 中剩余的 key-value 数据
    sm.range(->bool{
        std::cout <<"key: "<< key <<" , "<<"value "<< value << std::endl;
    return true;
    });

    std::cout <<"=========end========="<< std::endl;
}
```

## 4 总结

本篇是我在自学 c++ 过程中针对 sync.Map 技术的学习实践心得，本文分为两部分内容：

- • 第一部分向大家介绍了 go 中 sync.Map 的实现原理及使用场景
- • 第二部分向大家介绍了如何基于 c++ 效仿 go 从零实现 sync.Map

> 我自学 c++ 过程中所有实践项目都会收录于开源项目 https://github.com/xiaoxuxiansheng/cbricks.git 中，本篇介绍的 sync.Map 源码位于./sync/map.h. 大家如有需要，可以自取.

**微信扫一扫赞赏作者**

继续滑动看下一个

小徐先生的编程世界

向上滑动看下一个