# Go 语言 make 与 new 深度剖析

> 掌握 make 和 new 的区别，是 Go 语言面试中最基础也最高频的考点之一。本文将帮助你建立系统化的理解，从原理到实战，扫清所有盲点。

---

## 📋 面试必背：一图流掌握核心区别

| 特性 | `make` | `new` |
|:--|:--|:--|
| **适用类型** | 仅限 `slice`、`map`、`channel` | **任意类型** |
| **返回值** | 类型本身 (`T`) | 指针 (`*T`) |
| **初始化** | 完全初始化，内部结构已构建 | 仅分配内存并清零（零值） |
| **直接使用** | ✅ 可以直接操作 | ❌ 引用类型会导致 panic |
| **典型用途** | 创建可用切片/映射/通道 | 为结构体/基本类型获取指针 |

**一句话总结**：`make` 是"构造函数"，`new` 是"内存分配器"。

---

## 🔬 原理剖析：底层发生了什么？

### `new` 的底层逻辑

new(T) 做了什么：
1. 分配 sizeof(T) 大小的内存块
2. 将这块内存全部清零（所有位设为0）
3. 返回 *T 指针

本质：只做"分配 + 清零"，不负责"初始化"

**零值一览**：
- `int` → `0`
- `string` → `""`
- `struct` → 所有字段为其零值
- `slice` → `nil`（data=nil, len=0, cap=0）
- `map` → `nil`
- `channel` → `nil`

### `make` 的底层逻辑

make(T, args...) 做了什么：
1. 分配并初始化指定类型的内部数据结构
2. 对于 slice：分配底层数组 + 创建 slice header
3. 对于 map：创建哈希桶数据结构
4. 对于 channel：初始化环形队列和同步原语
5. 返回 T（不是指针）

本质：不仅"分配"，更关键是"初始化"，使类型真正可用

---

## 🎯 make 详解：专为引用类型而生

### 适用类型与签名

| 类型 | make 调用方式 | 说明 |
|:--|:--|:--|
| `slice` | `make([]T, len)` / `make([]T, len, cap)` | len ≤ cap |
| `map` | `make(map[K]V)` / `make(map[K]V, hint)` | hint 预分配桶数 |
| `channel` | `make(chan T)` / `make(chan T, buf)` | buf 缓冲区大小 |

### 代码示例

```go
// Slice：创建长度3、容量5的切片
s := make([]int, 3, 5)
// s: [0 0 0], len=3, cap=5

// Map：创建可立即使用的空映射
m := make(map[string]int)
m["a"] = 1  // ✅ 正常工作

// Channel：创建带缓冲的通道
ch := make(chan int, 10)
ch <- 42    // ✅ 正常工作
```

---

## 🎯 new 详解：通用的内存分配器

### 适用类型与特点

- **可用于任何类型**：基本类型、结构体、数组、接口、指针等
- **返回指针**：类型为 `*T`
- **仅做清零**：内部结构未构建

### 代码示例

```go
// 基本类型
i := new(int)      // *int, 值为 0
b := new(bool)     // *bool, 值为 false
s := new(string)   // *string, 值为 ""

// 结构体
p := new(Person)   // *Person, 零值结构体

// 数组
arr := new([5]int) // *[5]int, 值为 [0 0 0 0 0]
```

---

## ⚠️ 面试高频考点：new 用于引用类型的坑

这是面试中**出现频率最高**的陷阱题！

### Slice 对比

```go
// make: 返回可用切片
s1 := make([]int, 0)
s1 = append(s1, 1)  // ✅ 正常

// new: 返回指向 nil 切片的指针
s2 := new([]int)
*s2 = append(*s2, 1)  // ⚠️ 语法可行但写法别扭
// (*s2)[0] 访问会 panic
```

### Map 对比（⚠️ 最易踩坑）

```go
// make: 返回可用空 map
m1 := make(map[string]int)
m1["key"] = 100  // ✅ 正常

// new: 返回指向 nil map 的指针
m2 := new(map[string]int)
(*m2)["key"] = 100  // 💥 panic: assignment to entry in nil map
```

### Channel 对比（⚠️ 导致永久阻塞）

```go
// make: 返回可用 channel
ch1 := make(chan int)
go func() { ch1 <- 100 }()
<-ch1  // ✅ 正常

// new: 返回指向 nil channel 的指针
ch2 := new(chan int)
// 向 nil channel 发送 → 永久阻塞
// 从 nil channel 接收 → 永久阻塞
// 关闭 nil channel → panic
```

---

## 📝 面试真题精选

### Q1: make 和 new 的区别是什么？

**标准答案**：
> `make` 用于创建 slice、map、channel 三种引用类型，返回类型本身；`new` 用于创建任意类型的指针，返回 `*T`。`make` 不仅分配内存，还初始化内部结构使其可用；`new` 仅分配内存并清零。

### Q2: 为什么不推荐用 new 创建 map？

**标准答案**：
> `new(map[K]V)` 返回 `*map[K]V`，指向零值（nil map）。向 nil map 写入会 panic。正确做法是使用 `make(map[K]V)` 创建可用的空 map。

### Q3: make 可以用于基本类型吗？

**标准答案**：
> 不可以。`make` 只能用于 `slice`、`map`、`channel` 三种类型。

### Q4: 下面代码输出什么？

```go
p := new(Person)
fmt.Println(p == nil)
```

**答案**：`false`。`new` 返回的是指向零值结构体的指针，不是 nil。

---

## 🚀 实战速查表

```go
// ✅ 正确示范
s := make([]int, 0, 100)      // 预分配容量
m := make(map[string]int, 50) // 预分配桶
ch := make(chan int, 5)        // 带缓冲
p := new(Person)               // 结构体指针

// ❌ 错误示范
new([]int)       // 返回 *[]int，解引用是 nil slice
new(map[string]int)  // 返回 *map[string]int，解引用是 nil map
new(chan int)    // 返回 *chan int，解引用是 nil channel
```

---

## 🎓 记忆口诀

> **"make 是构造函数，new 是分配器；make 创建就能用，new 返回是指针；引用类型用 make，基本类型用 new。"**

---

## ✅ 自我检测

你能回答以下问题吗？

1. `make([]int, 0)` 和 `new([]int)` 的区别是什么？
2. 下面代码为什么会 panic？
   ```go
   m := new(map[string]int)
   m["a"] = 1
   ```
3. `make` 可以用于创建 `*int` 吗？为什么？
4. `new(Person)` 和 `&Person{}` 等价吗？

---

*答案提示*：
1. `make` 返回 `[]int`（可用），`new` 返回 `*[]int`（解引用是 nil）
2. `new` 返回的是 `*map[string]int`，指向 nil map，向 nil map 写入会 panic
3. 不可以，`make` 仅限 slice/map/channel
4. 等价，两者都返回 `*Person` 指针，指向零值结构体
