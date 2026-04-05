---
title: "Linux Radix Tree详解"
source: "https://zhuanlan.zhihu.com/p/599843140"
author:
  - "[[不理笨蛋​阿里云计算有限公司 员工]]"
published:
created: 2026-04-04
description: "关注微信公众号：Linux内核拾遗 文章来源：Linux Radix Tree详解1 概述众所周知，Linux内核提供了许多不同的库和函数来实现不同的数据结构和算法，其中基数树（Radix Tree）作为一种常见的数据结构，由于其查找速…"
tags:
  - "clippings"
---
> 关注微信公众号：Linux内核拾遗  
> 文章来源： [Linux Radix Tree详解](https://link.zhihu.com/?target=https%3A//mp.weixin.qq.com/s/1cxEi7Q5BoQK5J2s-wVJdg)

![](https://picx.zhimg.com/v2-9e058059ea7f0edb8fdc731683d24737_1440w.jpg)

Linux Radix Tree详解

## 1 概述

众所周知，Linux内核提供了许多不同的库和函数来实现不同的数据结构和算法，其中基数树（Radix Tree）作为一种常见的数据结构，由于其查找速度快、节省存储空间等特性，它在Linux内核中有着广泛的应用，例如在v4.20以前Linux内核使用Radix Tree来存储所有的页缓存（在后续的版本中采用xarray来取代了Radix Tree，本文暂不深入），struct [address\_space](https://zhida.zhihu.com/search?content_id=221431226&content_type=Article&match_order=1&q=address_space&zd_token=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJ6aGlkYV9zZXJ2ZXIiLCJleHAiOjE3NzU0ODc5MDYsInEiOiJhZGRyZXNzX3NwYWNlIiwiemhpZGFfc291cmNlIjoiZW50aXR5IiwiY29udGVudF9pZCI6MjIxNDMxMjI2LCJjb250ZW50X3R5cGUiOiJBcnRpY2xlIiwibWF0Y2hfb3JkZXIiOjEsInpkX3Rva2VuIjpudWxsfQ.LRAoghNkM_6KXF5p4PF1nciwiN1BvfFLquMpA42dFVA&zhida_source=entity) 结构体中包含了一个Radix Tree用于跟踪所有的映射页，以及Linux [内存管理单元](https://zhida.zhihu.com/search?content_id=221431226&content_type=Article&match_order=1&q=%E5%86%85%E5%AD%98%E7%AE%A1%E7%90%86%E5%8D%95%E5%85%83&zd_token=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJ6aGlkYV9zZXJ2ZXIiLCJleHAiOjE3NzU0ODc5MDYsInEiOiLlhoXlrZjnrqHnkIbljZXlhYMiLCJ6aGlkYV9zb3VyY2UiOiJlbnRpdHkiLCJjb250ZW50X2lkIjoyMjE0MzEyMjYsImNvbnRlbnRfdHlwZSI6IkFydGljbGUiLCJtYXRjaF9vcmRlciI6MSwiemRfdG9rZW4iOm51bGx9.HLKjMGanYC2OR4Cl9DDFWuhLxuOoN8lccg16Or2ZMcw&zhida_source=entity) 使用Radix Tree来快速查找脏页或者回写页等等。本文将对基数树这一数据结构作基本介绍，然后探讨Radix Tree在Linux内核中的设计实现以及使用方法。

## 2 基数树

根据维基百科的定义， **基数树** （ **Radix [Trie](https://zhida.zhihu.com/search?content_id=221431226&content_type=Article&match_order=1&q=Trie&zd_token=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJ6aGlkYV9zZXJ2ZXIiLCJleHAiOjE3NzU0ODc5MDYsInEiOiJUcmllIiwiemhpZGFfc291cmNlIjoiZW50aXR5IiwiY29udGVudF9pZCI6MjIxNDMxMjI2LCJjb250ZW50X3R5cGUiOiJBcnRpY2xlIiwibWF0Y2hfb3JkZXIiOjEsInpkX3Rva2VuIjpudWxsfQ.Vf4Lqj0NomQtWfiG0p4oaudt2LjPLfOer722JiwXIdQ&zhida_source=entity)** ，也叫 **基数特里树** 或 **压缩前缀树** ）是一种数据结构，是一种更节省空间的Trie（前缀树），其中作为唯一子节点的每个节点都与其父节点合并，边既可以表示为元素序列又可以表示为单个元素。 因此每个内部节点的子节点数最多为基数树的基数 *r* ，其中 *r* 为正整数以及 *x* 为2的幂（ *x* ≥1），这使得基数树更适用于对于较小的集合（尤其是字符串很长的情况下）和有很长相同前缀的字符串集合。

![](https://pica.zhimg.com/v2-449d0a667053f587c5efa120398b5306_1440w.jpg)

基数树示例

基数树的查找方式与常规树不同（常规的树查找一开始就对整个键进行比较，直到不相同为止），基数树查找节点时，对于节点上的键都按块进行逐块比较，其中该节点中块的长度是基数 *r* 。

- 当 *r* 为2时，基数树是二元基数树（即该节点的键的长度为1bit），增加了树的深度来降低稀疏性（即最大限度地合并键中没有分叉的节点）。
- 当 *r* ≥4且为2的整数次幂时，基数树是r元基数树，牺牲了稀疏性来降低基数树的深度。

基数树支持插入、删除和查找等操作。插入操作将新的元素插入到树结构中，同时尽可能最小化数据的存储空间；删除操作将元素从树结构中移除；查找操作包括但不限于精准查找、查找前继元素、查找后继元素以及查找拥有公共前缀的所有元素。所有的这些操作的时间复杂度均为O(k)，其中k是集合中全部元素的最大长度。

### 2.1 查找

![](https://pic4.zhimg.com/v2-8f60e3ef467f246fcf56a951bf8a5729_1440w.jpg)

基数树查找过程

查找操作用于确定一个元素是否存在于基数树中，基本操作流程如下：

```c
// 输入: x
1. 从根节点开始，递归执行如下的操作直到叶子节点或者中途退出
    1.1 如果当前节点为空，或者当前节点是叶子节点，或者查找的长度大于等于x长度，则跳转到步骤2中
  1.2 遍历当前节点全部的边
      1.2.1 如果边上的标签与x对应位置上的子串匹配，当前节点更新为边指向的子节点
      1.2.2 否则当前节点更新为空
  1.3 回到步骤1.1
2. 判断元素是否存在
  2.1 如果当前节点非空，并且当前节点是叶子节点，同时查找的长度等于x长度，则元素存在
  2.2 否则元素不存在
3. 返回查找结果
```

### 2.2 插入

插入操作是在查找基数树的过程中执行相关节点或者边的调整操作，直到无法执行进一步操作为止。这种调整操作通常可以分为如下两种情况：

1. 对于当前节点，如果有一个输出边与剩余的串共享一个前缀，此时将该输出边拆分成两条边，其中一条边标记为公共前缀。这时候转变成了第二种情况。
2. 对于当前节点，此时所有的输出边都没有和剩余的串共享一个前缀，此时直接添加一个新的输出边，并用输出串的所有剩余元素进行标记。

以下是插入时可以遇到的几种情况示例，其中用r表示基数树根节点，带空标签的边表示一个串的结束。

![](https://pic1.zhimg.com/v2-91561de6ec03d5cdbb89758a80acbdf0_1440w.png)

插入"water"

![](https://picx.zhimg.com/v2-3199d71651559d8e20d956e585a59b07_1440w.png)

插入"slower"，同时保留"slow"

![](https://pic4.zhimg.com/v2-c96c1ea0b0ab0099818db761b67129d5_1440w.jpg)

插入"test"，它是"tester"的前缀

![](https://pic2.zhimg.com/v2-941cc0a4d277b8ae89bf77bb36153b93_1440w.jpg)

插入"team"，同时将"test"进行拆分并且创建了一个带"st"标签的新边

![](https://picx.zhimg.com/v2-11e34f396df9d21a0a341b47e7d8502b_1440w.jpg)

插入"toast"，同时将"te"进行拆分并且将之前的串移到树的下一层

### 2.3 删除

要从基数树中删除串x，首先需要定位到表示x的叶子节点。如果x存在，则将相应的叶子节点移除，同时作如下的结构调整：

1. 如果该叶子节点的父节点只有一个额外的子节点（不包括x自身），那么将子节点的标签追加到父节点的标签之后，并且将该子节点删除。
2. 否则不作调整。

## 3 Linux Radix Tree

Linux radix tree是一种采用基数树数据结构来将一个值（通常是指针）关联到一个整数类型键的机制。

### 3.1 内部结构

Linux radix tree的叶子节点结构如下图所示，该节点包含了多个槽（slot），每个slot包含了一个指针，它指向了radix tree创建者感兴趣的东西（通常是一个结构体或者一段内存），而空的slot包含了NULL指针。

![](https://pic3.zhimg.com/v2-a4f95f0d675a79f97ca98bc7e31520b2_1440w.png)

Linux radix tree 叶子节点结构

Linux radix tree是一种”胖型“树结构，在Linux 2.6版本中每个树节点包含了64个slot，每个slot都是通过整数类型的键值的其中一部分进行索引。如果键值的最大取值小于64，那么整棵树可以表示成单个节点，但是通常情况下键值的范围是相当大的，否则就直接使用一个简单的数组即可而无需使用radix tree。一个稍大点的radix tree结构类似下图：

![](https://pic2.zhimg.com/v2-e18d64859394826c9ba8d5d160574875_1440w.jpg)

Linux radix tree 结构示例

上图的radix tree有三层节点，内核在radix tree中查找一个特定的键值的基本步骤如下：

1. 键值中最高6 bits用于查找根节点中合适的slot；
2. 键值中随后的6 bits用于在中间节点中对slot进行索引；
3. 键值中最后的6 bits则指向了包含了指向实际值的指针的slot。

没有子节点的节点不被包含在radix tree中，因此radix tree能够为稀疏树结构提供高效的存储方式。

### 3.3 内核API

Linux内核提供了相当多的API接口，方便内核用户进行创建、查找、插入、遍历和删除等操作。下面介绍一些最常用的Radix Tree API接口。

### 3.3.1 声明和初始化

```c
#define RADIX_TREE_INIT(mask)    {                    \
    .height = 0,                            \
    .gfp_mask = (mask),                        \
    .rnode = NULL,                            \
}

#define RADIX_TREE(name, mask) \
    struct radix_tree_root name = RADIX_TREE_INIT(mask)

#define INIT_RADIX_TREE(root, mask)                    \
do {                                    \
    (root)->height = 0;                        \
    (root)->gfp_mask = (mask);                    \
    (root)->rnode = NULL;                        \
} while (0)
```

Linux提供了两种声明和初始化radix tree的方法：

1. RADIX\_TREE：这个宏静态声明并初始化了一个名为name的radix tree。
2. INIT\_RADIX\_TREE：这个宏用于在运行时对radix tree进行初始化。

在上面两种方式中，都必须提供gfp\_mask来制定内存分配的策略，例如当radix tree操作（尤其是插入操作）需要在原子上下文中执行时，给定的mask必须是 [GFP\_ATOMIC](https://zhida.zhihu.com/search?content_id=221431226&content_type=Article&match_order=1&q=GFP_ATOMIC&zd_token=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJ6aGlkYV9zZXJ2ZXIiLCJleHAiOjE3NzU0ODc5MDYsInEiOiJHRlBfQVRPTUlDIiwiemhpZGFfc291cmNlIjoiZW50aXR5IiwiY29udGVudF9pZCI6MjIxNDMxMjI2LCJjb250ZW50X3R5cGUiOiJBcnRpY2xlIiwibWF0Y2hfb3JkZXIiOjEsInpkX3Rva2VuIjpudWxsfQ.7AA4FIVg_6yTaH4Z0EGzNoZT8D9EqdeENqJu1Qb6bh8&zhida_source=entity) 。

### 3.3.2 插入

```c
int radix_tree_insert(struct radix_tree_root *root, unsigned long index, void *item);
```

该方法将键值index关联的元素item插入到radix tree树结构root中，成功时返回值为0。

该方法可能触发内存分配操作，如果内存分配失败，那么插入操作失败并且返回值为-ENOMEM。

该方法不会覆写一个已经存在的元素，如果index已经存在于radix tree中，该方法将会返回-EEXIST。

### 3.3.3 删除

```c
void *radix_tree_delete(struct radix_tree_root *root, unsigned long index);
```

该方法将键值index关联的元素从radix tree中移除，如果该元素存在则返回指向该元素的指针。需要注意的是，该方法并不会释放元素的内存空间。

### 3.3.4 内存预分配

Linux内核提供了一对专用的函数来帮助避免radix tree插入操作失败的情况：

```c
int radix_tree_preload(gfp_t gfp_mask);
static inline void radix_tree_preload_end(void) {
    preempt_enable();
}
```
1. radix\_tree\_preload：该方法尝试分配足够的内存来保证下一次radix tree插入操作不会失败。它预分配的数据结构存储在per-CPU变量中，这意味着该函数的调用方必须在其允许调度或者被移动到不同的处理器之前执行插入操作。因此该函数执行成功后会禁用抢占再返回，调用者必须确保最终会重新开启抢占（通过调用radix\_tree\_preload\_end方法）。如果失败，该方法返回-ENOMEM，同时不会禁用抢占。
2. radix\_tree\_preload\_end：重新开启抢占，与radix\_tree\_preload配对使用。

### 3.3.5 查找

```c
void *radix_tree_lookup(struct radix_tree_root *tree, unsigned long index);
void **radix_tree_lookup_slot(struct radix_tree_root *tree, unsigned long index);
```
1. radix\_tree\_lookup：在radix tree中查找键值index并且返回其关联的元素（查找失败返回NULL）。
2. radix\_tree\_lookup\_slot：功能同radix\_tree\_lookup，但返回的是指向slot的指针，而slot中存储了指向元素的指针，可以简单理解为该方法返回的是元素的二级指针，调用者可以通过这个方法修改键值key指向新的元素。如果元素不存在，该方法不会为其创建一个slot，因此它不能用于取代radix\_tree\_insert()方法。

### 3.3.6 遍历

```c
#define radix_tree_for_each_slot(slot, root, iter, start)        \
    for (slot = radix_tree_iter_init(iter, start) ;            \
         slot || (slot = radix_tree_next_chunk(root, iter, 0)) ;    \
         slot = radix_tree_next_slot(slot, iter, 0))
```

该宏用于遍历radix tree中所有的非空slot：

- slot：指向slot的指针，类型为void\*\*。
- root：指向radix tree根节点的指针，类型为struct radix\_tree\_root \*。
- iter：类型为struct radix\_tree\_iter \*的指针，iter->index包含了当前元素的索引。
- start：遍历的起始索引。

### 3.3.7 销毁

Linux内核并没有提供用于销毁和回收radix tree内存空间的API，这意味着radix tree会永久存在于内存中直到通过其他方式手动释放。在实践中，通常的方法是遍历radix tree中的所有元素并将其从radix tree中移除，然后对全部的元素调用对应的内存释放方法（kfree等）。

## 4 示例

下面演示了Linux radix tree API的基本使用：

```c
#include <linux/init.h>
#include <linux/module.h>
#include <linux/radix-tree.h>
#include <linux/slab.h>

struct element {
    unsigned long key;
    unsigned char data[16];
};

RADIX_TREE(test_tree, GFP_KERNEL);

static int __init test_init(void) {
    struct element *e, *e1, *e2, *e3;
    struct radix_tree_iter iter;
    void** slot;

    e1 = kmalloc(sizeof(struct element) * 3, GFP_KERNEL);
    if (!e1) {
        pr_info("kmalloc failed\n");
        return -ENOMEM;
    }
    e2 = e1 + 1;
    e3 = e2 + 1;

    e1->key = 100;
    sprintf(e1->data, "element 1");
    e2->key = 2000;
    sprintf(e2->data, "element 2");
    e3->key = 33333;
    sprintf(e3->data, "element 3");

    radix_tree_insert(&test_tree, e1->key, e1);

    e = radix_tree_lookup(&test_tree, 500L);
    if (!e) {
        pr_info("radix_tree_lookup %ld failed\n", 500L);
    }

    radix_tree_insert(&test_tree, e2->key, e2);
    e = radix_tree_lookup(&test_tree, 2000L);
    if (!e) {
        pr_info("radix_tree_lookup failed\n");
    }
    pr_info("radix_tree_lookup: %ld, %s\n", e->key, e->data);

    radix_tree_preload(GFP_KERNEL);
    radix_tree_insert(&test_tree, e3->key, e3);
    radix_tree_preload_end();

    e = radix_tree_delete(&test_tree, 2000L);
    if (!e) {
        pr_info("radix_tree_delete failed\n");
    }
    pr_info("radix_tree_delete: %ld, %s\n", e->key, e->data);

    radix_tree_for_each_slot(slot, &test_tree, &iter, 0) {
        e = *slot;
        pr_info("radix_tree_for_each_slot: %ld, %s\n", e->key, e->data);
    }

    radix_tree_for_each_slot(slot, &test_tree, &iter, 0) {
        e = *slot;
        radix_tree_delete(&test_tree, e->key);
    }
    kfree(e1);

    return 0;
}

static void __exit test_exit(void) {}

module_init(test_init);
module_exit(test_exit);

MODULE_LICENSE("GPL");
MODULE_DESCRIPTION("test routine for Linux radix tree");
```

编译运行输出如下：

```c
[root@host radix-tree]# make all
make -C /lib/modules/3.10.0-957.el7.x86_64/build  M=/root/radix-tree modules
make[1]: Entering directory \`/usr/src/kernels/3.10.0-957.el7.x86_64'
  CC [M]  /root/radix-tree/radix-tree-test.o
  Building modules, stage 2.
  MODPOST 1 modules
  CC      /root/radix-tree/radix-tree-test.mod.o
  LD [M]  /root/radix-tree/radix-tree-test.ko
make[1]: Leaving directory \`/usr/src/kernels/3.10.0-957.el7.x86_64'
[root@host radix-tree]# dmesg | tail
[3907932.381368] IPVS: rr: TCP 10.10.238.40:8380 - no destination available
[3907933.383983] IPVS: rr: TCP 10.10.238.40:8380 - no destination available
[3907963.386460] IPVS: rr: TCP 10.10.238.40:8380 - no destination available
[3907982.298189] radix_tree_lookup 500 failed
[3907982.298192] radix_tree_lookup: 2000, element 2
[3907982.298194] radix_tree_delete: 2000, element 2
[3907982.298196] radix_tree_for_each_slot: 100, element 1
[3907982.298197] radix_tree_for_each_slot: 33333, element 3
[3907993.390862] IPVS: rr: TCP 10.10.238.40:8380 - no destination available
[3907994.391450] IPVS: rr: TCP 10.10.238.40:8380 - no destination available
```

## 5 参考资料

1. Trees I: Radix trees： [lwn.net/Articles/175432](https://link.zhihu.com/?target=https%3A//lwn.net/Articles/175432/)
2. Radix tree： [en.wikipedia.org/wiki/R](https://link.zhihu.com/?target=https%3A//en.wikipedia.org/wiki/Radix_tree)
3. Why we use radix-tree(or xarray) for storing page caches?： [stackoverflow.com/quest](https://link.zhihu.com/?target=https%3A//stackoverflow.com/questions/62447084/why-we-use-radix-treeor-xarray-for-storing-page-caches)

> 关注微信公众号：Linux内核拾遗  
> 文章来源： [Linux Radix Tree详解](https://link.zhihu.com/?target=https%3A//mp.weixin.qq.com/s/1cxEi7Q5BoQK5J2s-wVJdg)

编辑于 2023-01-18 12:48・广东[Linux 开发](https://www.zhihu.com/topic/19610306)[数据结构](https://www.zhihu.com/topic/19591797)[Linux 内核](https://www.zhihu.com/topic/19614193)