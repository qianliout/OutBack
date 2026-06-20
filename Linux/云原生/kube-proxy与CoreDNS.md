# kube-proxy 与 CoreDNS 原理详解

> **一句话**：kube-proxy 负责 **Service 流量的负载均衡转发**（L3/L4），CoreDNS 负责 **Service 名称到 IP 的解析**（服务发现）。两者协作构成了 K8s 内部的服务发现与路由体系。

---

## 目录

- [第一部分：kube-proxy](#第一部分kube-proxy)
  - [一、kube-proxy 是什么](#一kube-proxy-是什么)
  - [二、kube-proxy 的三种代理模式](#二kube-proxy-的三种代理模式)
  - [三、iptables 模式深入](#三iptables-模式深入)
  - [四、IPVS 模式深入](#四ipvs-模式深入)
  - [五、iptables vs IPVS 对比](#五iptables-vs-ipvs-对比)
  - [六、kube-proxy 的内部工作机制](#六kube-proxy-的内部工作机制)
  - [七、conntrack 与 kube-proxy](#七conntrack-与-kube-proxy)
  - [八、为什么 ClusterIP ping 不通？](#八为什么-clusterip-ping-不通)
  - [九、kube-proxy 排错](#九kube-proxy-排错)
- [第二部分：CoreDNS](#第二部分coredns)
  - [十、CoreDNS 是什么](#十coredns-是什么)
  - [十一、CoreDNS 的插件架构](#十一coredns-的插件架构)
  - [十二、K8s 中的 CoreDNS 部署](#十二k8s-中的-coredns-部署)
  - [十三、DNS 解析全链路](#十三dns-解析全链路)
  - [十四、Corefile 配置详解](#十四corefile-配置详解)
  - [十五、Service DNS 记录格式](#十五service-dns-记录格式)
  - [十六、Pod DNS 策略](#十六pod-dns-策略)
  - [十七、CoreDNS 排错](#十七coredns-排错)
- [第三部分：kube-proxy + CoreDNS 的协作全景](#第三部分kube-proxy--coredns-的协作全景)
  - [十八、一次完整的服务调用链路](#十八一次完整的服务调用链路)
  - [十九、两者职责边界表](#十九两者职责边界表)

---

# 第一部分：kube-proxy

## 一、kube-proxy 是什么

### 1.1 一句话定义

> **kube-proxy 是运行在每个 K8s 节点上的网络代理组件，它负责将发往 Service ClusterIP 的流量负载均衡地转发到后端的 Pod IP。**

### 1.2 为什么需要 kube-proxy？

K8s 中 Pod 的 IP 是**短暂且可变的**（Pod 重建后 IP 会变）。如果客户端直接使用 Pod IP 通信：

- Pod 重启后客户端需要更新 IP
- 无法做负载均衡

Service 解决了这两个问题：提供一个**稳定的虚拟 IP（ClusterIP）**，自动发现后端 Pod 并负载均衡。但 ClusterIP 是一个虚拟 IP——**没有任何网络接口持有它**。kube-proxy 的工作就是把这个虚拟 IP 翻译成真实的 Pod IP。

### 1.3 kube-proxy 在节点上的位置

```
Pod A (客户端)                        Pod B (服务端)
    │                                      ▲
    │  curl 10.96.0.10:80                  │
    ▼                                      │
┌───▼──────────────────────────────────────┴───┐
│              节点网络栈                        │
│                                               │
│   ① DNS 解析: my-svc → 10.96.0.10             │
│   ② 数据包目标: 10.96.0.10:80                 │
│   ③ kube-proxy 规则拦截                       │
│   ④ DNAT: 10.96.0.10:80 → 10.244.1.5:8080    │
│   ⑤ 数据包发往 Pod B                          │
│                                               │
│   kube-proxy (iptables/IPVS 规则)             │
└───────────────────────────────────────────────┘
```

---

## 二、kube-proxy 的三种代理模式

### 2.1 模式演进历史

```
userspace 模式          iptables 模式           IPVS 模式
（K8s 1.0，已废弃）  →  （1.2，曾长期默认）  →  （1.11 GA，性能最优）
```

### 2.2 三种模式对比

| 特性 | userspace | iptables | IPVS |
|------|-----------|----------|------|
| **转发方式** | kube-proxy 进程在用户态转发 | iptables 内核规则 DNAT | IPVS 内核模块直接转发 |
| **性能** | 差（用户态 ↔ 内核态切换） | 中等（规则数量多时线性匹配变慢） | 好（哈希表 O(1) 查找） |
| **负载均衡算法** | 轮询 | 随机（`-m statistic --mode random`） | rr / lc / dh / sh / sed / nq 共 8 种 |
| **连接追踪** | 不需要 | 依赖 conntrack | 依赖 conntrack |
| **大规模 Service** | N/A | 5,000+ 规则时性能显著下降 | 10,000+ Service 仍高效 |
| **当前状态** | 已废弃 | 默认/广泛使用 | 推荐用于大规模集群 |

### 2.3 为什么 userspace 模式被淘汰？

userspace 模式下，kube-proxy 自身作为一个**代理进程**监听在随机端口。数据包从内核态搬运到用户态（kube-proxy 进程），重新封装后再从用户态搬回内核态——**每次请求两次上下文切换**。这在吞吐量稍高时就成为瓶颈。

iptables 和 IPVS 模式都在**内核态**直接完成转发，没有这种开销。

---

## 三、iptables 模式深入

### 3.1 规则生成原理

kube-proxy 监听 API Server 的 Service 和 EndpointSlice 变更事件，动态生成 iptables 规则。

假设有 Service：`my-svc`，ClusterIP `10.96.0.10:80`，后端有 3 个 Pod（`10.244.1.2:8080`，`10.244.1.3:8080`，`10.244.2.1:8080`）。

kube-proxy 会生成如下规则链逻辑：

```
PREROUTING → KUBE-SERVICES
                │
                │ 匹配: dst=10.96.0.10, dport=80
                ▼
           KUBE-SVC-XXXX (Service 链)
                │
                │ 随机选择 (--mode random --probability)
                ├──→ KUBE-SEP-AAAA (Pod 1 endpoint 链)
                │       └── DNAT: 10.96.0.10:80 → 10.244.1.2:8080
                │
                ├──→ KUBE-SEP-BBBB (Pod 2 endpoint 链)
                │       └── DNAT: 10.96.0.10:80 → 10.244.1.3:8080
                │
                └──→ KUBE-SEP-CCCC (Pod 3 endpoint 链)
                        └── DNAT: 10.96.0.10:80 → 10.244.2.1:8080
```

### 3.2 实际查看规则

```bash
# 查看所有 K8s 相关的 iptables 规则（NAT 表）
iptables-save -t nat | grep -E "KUBE-SERVICES|KUBE-SVC|KUBE-SEP" | head -50

# 查看具体 Service 的规则链
iptables -t nat -L KUBE-SERVICES -n | grep 10.96.0.10

# 查看某个 Endpoint 的规则
iptables -t nat -L KUBE-SEP-XXXX -n
```

### 3.3 iptables 模式的随机负载均衡

iptables 不支持真正的"轮询"，而是使用**随机匹配**：

```bash
# 伪代码：概率随机选择
iptables -A KUBE-SVC-XXXX \
  -m statistic --mode random --probability 0.33333 \
  -j KUBE-SEP-AAAA

iptables -A KUBE-SVC-XXXX \
  -m statistic --mode random --probability 0.50000 \
  -j KUBE-SEP-BBBB

iptables -A KUBE-SVC-XXXX \
  -j KUBE-SEP-CCCC   # 剩下的流量自然走最后一个
```

第 1 条规则匹配 33% 流量 → Pod1，第 2 条匹配剩余流量的 50%（即总量的 33%）→ Pod2，第 3 条兜底 34% → Pod3。三条规则叠加实现 1:1:1 的均匀分布。

### 3.4 iptables 模式的性能问题

iptables 规则匹配是**线性遍历**的。当 Service 数量达到数千个时：

- 每个 Service 有 1 条 `KUBE-SERVICES` 规则 + 1 条 `KUBE-SVC` 链 + N 条 `KUBE-SEP` 链
- 5,000 个 Service × 10 个 Pod 平均 = ~55,000 条规则
- 每一条规则都需要按顺序匹配，O(n) 复杂度
- iptables 规则更新需要**全量替换**，不能增量（`iptables-restore`）

**这就是 5,000+ Service 集群中 iptables 模式不可用的原因。**

---

## 四、IPVS 模式深入

### 4.1 IPVS 是什么？

IPVS（IP Virtual Server）是 Linux 内核的 L4 负载均衡框架，作为 `ip_vs` 内核模块运行。它维护一个**哈希表**来查找虚拟服务和后端真实服务器。

### 4.2 IPVS 的数据结构

kube-proxy 在 IPVS 模式下，为每个 Service 创建以下结构：

```
虚拟服务器 (VirtualServer):  10.96.0.10:80  (Service ClusterIP)
    │
    ├── 真实服务器 (RealServer):  10.244.1.2:8080  (Pod 1)
    ├── 真实服务器 (RealServer):  10.244.1.3:8080  (Pod 2)
    └── 真实服务器 (RealServer):  10.244.2.1:8080  (Pod 3)
```

内核 IPVS 模块收到发往 `10.96.0.10:80` 的数据包后，在哈希表中 O(1) 查找虚拟服务器，然后根据调度算法选择真实服务器，直接改写目标 IP/端口。

### 4.3 启用 IPVS 模式

```bash
# 1. 确保节点加载了 IPVS 内核模块
lsmod | grep ip_vs
# 如果没有，加载：
modprobe ip_vs
modprobe ip_vs_rr
modprobe ip_vs_wrr
modprobe ip_vs_sh

# 2. kube-proxy 配置（ConfigMap 或命令行）
# kube-proxy --proxy-mode=ipvs
# 或在 kube-proxy ConfigMap 中设置 mode: "ipvs"

# 3. 重启 kube-proxy
kubectl -n kube-system rollout restart daemonset kube-proxy
```

### 4.4 查看 IPVS 规则

```bash
# 查看所有虚拟服务和真实服务器
ipvsadm -Ln

# 输出示例：
# IP Virtual Server version 1.2.1 (size=4096)
# Prot LocalAddress:Port Scheduler Flags
#   -> RemoteAddress:Port           Forward Weight ActiveConn InActConn
# TCP  10.96.0.10:80 rr
#   -> 10.244.1.2:8080              Masq    1      0          0
#   -> 10.244.1.3:8080              Masq    1      0          0
#   -> 10.244.2.1:8080              Masq    1      0          0

# 查看调度算法和连接统计
ipvsadm -Ln --stats
```

### 4.5 IPVS 支持的调度算法

| 算法 | 缩写 | 说明 |
|------|------|------|
| Round Robin | `rr` | 轮询（默认） |
| Weighted Round Robin | `wrr` | 加权轮询 |
| Least Connection | `lc` | 最少连接数 |
| Weighted Least Connection | `wlc` | 加权最少连接 |
| Destination Hashing | `dh` | 基于目标 IP 的哈希（会话保持） |
| Source Hashing | `sh` | 基于源 IP 的哈希（来自同一源 IP 的请求发给同一后端） |
| Shortest Expected Delay | `sed` | 最短期望延迟 |
| Never Queue | `nq` | 从不排队 |

```bash
# 指定调度算法（kube-proxy 参数）
kube-proxy --proxy-mode=ipvs --ipvs-scheduler=lc
```

---

## 五、iptables vs IPVS 对比

### 5.1 核心区别

| 维度 | iptables | IPVS |
|------|----------|------|
| **数据结构** | 链表（规则按序匹配） | 哈希表（O(1) 查找） |
| **规则更新** | 全量替换（`iptables-restore`） | 增量更新（`ipvsadm -a/-d`） |
| **10,000 Service 性能** | 规则匹配延迟显著增加 | 几乎无影响 |
| **负载均衡算法** | 随机（1 种） | 8 种可选 |
| **连接追踪** | conntrack 记录每个连接 | conntrack 记录每个连接 |
| **健康检查** | 依赖 EndpointSlice 状态 | 支持 TCP/HTTP 主动健康检查 |
| **会话保持** | 通过 conntrack 实现的 session affinity | 原生 sh/dh 算法 |
| **内核模块依赖** | 无（iptables 是基础能力） | 需要 `ip_vs` 等内核模块 |

### 5.2 如何选择？

| 场景 | 推荐 |
|------|------|
| 小集群（< 500 Service） | iptables（简单可靠） |
| 中大型集群（500-5000 Service） | iptables 仍可用，但建议迁移 IPVS |
| 大型集群（> 5000 Service） | **必须用 IPVS** |
| 需要特定调度算法（最少连接等） | IPVS |
| 边缘/嵌入式 K8s（k3s 等） | iptables（无额外内核模块依赖） |

---

## 六、kube-proxy 的内部工作机制

### 6.1 kube-proxy 的 DaemonSet 结构

```
每个 K8s Worker 节点上都运行一个 kube-proxy Pod：

$ kubectl -n kube-system get daemonset kube-proxy
NAME         DESIRED   CURRENT   READY   UP-TO-DATE   AVAILABLE
kube-proxy   3         3         3       3            3
```

**为什么是 DaemonSet？** 因为 kube-proxy 修改的是**节点的 iptables/IPVS 规则**，必须在每个节点本地运行。

### 6.2 核心流程

```
kube-proxy 启动
    │
    ├── ① 连接 API Server，获取当前所有 Service + EndpointSlice
    │
    ├── ② 根据配置选择代理模式（iptables / IPVS）
    │
    ├── ③ 在本节点生成网络规则（iptables 规则链 / IPVS 虚拟服务器）
    │
    ├── ④ Watch API Server（Informer 模式）
    │      │
    │      ├── Service 新增 → 添加规则/虚拟服务器
    │      ├── Service 删除 → 移除规则/虚拟服务器
    │      ├── EndpointSlice 新增 → 添加后端真实服务器
    │      └── EndpointSlice 删除 → 移除后端真实服务器
    │
    └── ⑤ 定期全量同步（sync-loop，默认 30s）
           确保节点规则与 API Server 状态最终一致
```

### 6.3 关键参数

```bash
kube-proxy \
  --proxy-mode=ipvs \              # 代理模式
  --ipvs-scheduler=rr \            # IPVS 调度算法
  --ipvs-min-sync-period=2s \      # IPVS 规则最小同步间隔
  --iptables-sync-period=30s \     # iptables 同步周期
  --conntrack-max-per-core=32768 \ # 每个 CPU 核心的 conntrack 表上限
  --masquerade-all=false           # 是否对所有流量做 SNAT
```

### 6.4 kube-proxy 写入的 iptables 链一览

```bash
iptables-save | grep -E '^:KUBE'

# 主要链：
# KUBE-SERVICES      - 入口链，匹配 Service ClusterIP
# KUBE-SVC-XXXX      - 每个 Service 一条链，负载均衡分发
# KUBE-SEP-XXXX      - 每个 Endpoint 一条链，做 DNAT
# KUBE-NODEPORTS     - NodePort 流量入口
# KUBE-POSTROUTING   - 出站 SNAT/MASQUERADE
# KUBE-MARK-MASQ     - 标记需要 MASQUERADE 的数据包
# KUBE-MARK-DROP     - 标记需要丢弃的数据包
# KUBE-FIREWALL      - 防火墙相关规则
```

---

## 七、conntrack 与 kube-proxy

### 7.1 conntrack 的作用

conntrack（Connection Tracking）是 Linux 内核的连接跟踪表。当你做 DNAT 时，conntrack 记录：

```
原始连接: client_ip:random_port → 10.96.0.10:80
DNAT 后:  client_ip:random_port → 10.244.1.2:8080
```

这样返回包到达时，内核能通过 conntrack 表**反向 DNAT**：把 `10.244.1.2:8080 → client_ip` 改回 `10.96.0.10:80 → client_ip`。

### 7.2 conntrack 表满的灾难

当节点上连接数很高时，conntrack 表可能被填满：

```bash
# 查看 conntrack 表当前使用量
cat /proc/sys/net/netfilter/nf_conntrack_count
# 查看上限
cat /proc/sys/net/netfilter/nf_conntrack_max

# 如果 count 接近 max → 新连接会被丢弃！
```

**表现**：Pod 之间间歇性连接超时、DNS 查询失败（UDP 尤其容易触发）。

**缓解措施**：

```bash
# 调大 conntrack 表
echo 524288 > /proc/sys/net/netfilter/nf_conntrack_max

# kube-proxy 参数
kube-proxy --conntrack-max-per-core=32768

# 缩短 conntrack 超时时间
echo 1800 > /proc/sys/net/netfilter/nf_conntrack_tcp_timeout_established
```

### 7.3 查看 conntrack 表

```bash
# 查看所有连接（可能非常多，建议加过滤）
conntrack -L -d 10.96.0.10 | head -20

# 统计 Service 的连接数
conntrack -L -d 10.96.0.10 | wc -l
```

---

## 八、为什么 ClusterIP ping 不通？

这是高频问题。**原因**：ClusterIP 是一个纯虚拟 IP，没有网络接口持有它。

```
ping 10.96.0.10  →  内核发出 ICMP Echo Request
                      ↓
                     没有网卡持有 10.96.0.10
                     ↓
                     无法 ARP 解析（没有哪个 MAC 地址响应）
                     ↓
                     丢包
```

kube-proxy 的 iptables/IPVS 规则只能拦截 **TCP/UDP** 数据包（匹配了目标端口），ICMP **不走** kube-proxy 的规则。

> `curl 10.96.0.10:80` 能通是因为 iptables 击中了 `-p tcp --dport 80` 规则，做了 DNAT。

---

## 九、kube-proxy 排错

### 9.1 检查 kube-proxy Pod 状态

```bash
kubectl -n kube-system get pods -l k8s-app=kube-proxy
kubectl -n kube-system logs -l k8s-app=kube-proxy --tail=50
```

### 9.2 检查节点上的规则

```bash
# iptables 模式
iptables-save -t nat | grep -c KUBE    # 规则数量
iptables -t nat -L KUBE-SERVICES -n     # 入口规则

# IPVS 模式
ipvsadm -Ln                             # 虚拟服务列表
ipvsadm -Ln --stats                     # 连接统计
```

### 9.3 常见问题

| 问题 | 可能原因 | 排查命令 |
|------|----------|----------|
| 新 Service 不可达 | kube-proxy 规则未同步 | `kubectl -n kube-system logs kube-proxy-xxx` 查同步日志 |
| 间歇性连接超时 | conntrack 表满 | `cat /proc/sys/net/netfilter/nf_conntrack_count` |
| IPVS 模式不生效 | 内核模块未加载 | `lsmod \| grep ip_vs` |
| NodePort 不可达 | 本地防火墙拦截 | `iptables -t nat -L KUBE-NODEPORTS -n` |
| Service 负载不均衡 | `externalTrafficPolicy: Local` 导致 | 检查 Service spec |

---

# 第二部分：CoreDNS

## 十、CoreDNS 是什么

### 10.1 一句话定义

> **CoreDNS 是 K8s 集群的 DNS 服务器，负责将 Service 名称解析为 ClusterIP，是集群内服务发现的核心组件。**

### 10.2 为什么需要 CoreDNS？

K8s 中更新换代的历程：

```
Kube-dns (早期 K8s)        CoreDNS (1.11 GA)         CoreDNS (1.13 默认)
┌─────────────────┐       ┌──────────────┐          ┌──────────────┐
│ kube-dns (dnsmasq)│  →  │ 单个二进制文件  │   →    │ 插件化架构     │
│ dnsmasq          │      │ 插件化设计     │         │ 性能更好       │
│ sidecar          │      │ 内存占用更小    │         │ 配置更灵活     │
│ 三个容器！        │      │ 一个容器搞定    │         │ 默认 2 副本    │
└─────────────────┘       └──────────────┘          └──────────────┘
```

CoreDNS 相比 kube-dns 的优势：
- **单进程**：不再是三个容器侧车模式
- **插件化**：按需加载插件，灵活扩展
- **缓存**：内置缓存插件减少 API Server 压力
- **内存**：比 kube-dns 少约 50% 内存

### 10.3 CoreDNS 在集群中的位置

```
┌──────────────────────────────────────────────────┐
│                   K8s 集群                         │
│                                                   │
│  Pod A (10.244.1.5)                              │
│    │                                              │
│    │ DNS 查询: my-svc.default.svc.cluster.local   │
│    ▼                                              │
│  /etc/resolv.conf                                │
│    nameserver 10.96.0.10   ← CoreDNS Service IP  │
│    search default.svc.cluster.local               │
│    search svc.cluster.local                       │
│    search cluster.local                           │
│    │                                              │
│    ▼                                              │
│  ┌───────────────────────────────────┐            │
│  │     CoreDNS Service (kube-system) │            │
│  │     ClusterIP: 10.96.0.10:53      │            │
│  │                                    │            │
│  │  ┌──────────┐  ┌──────────┐       │            │
│  │  │ CoreDNS  │  │ CoreDNS  │       │            │
│  │  │ Pod 1    │  │ Pod 2    │       │            │
│  │  └──────────┘  └──────────┘       │            │
│  └────────────┬──────────────────────┘            │
│               │                                    │
│               │ Watch Services + EndpointSlices    │
│               ▼                                    │
│         ┌──────────┐                               │
│         │ API Server│                              │
│         └──────────┘                               │
└──────────────────────────────────────────────────┘
```

---

## 十一、CoreDNS 的插件架构

### 11.1 插件模型

CoreDNS 的核心本身只做一件事：接收 DNS 请求，按照 **Corefile** 配置的插件链依次处理，最后返回响应。所有功能都由插件提供。

```
DNS 请求进入
    │
    ▼
┌─────────┐    ┌─────────┐    ┌─────────┐    ┌────────┐
│ errors  │ →  │ health  │ →  │  kubernetes │ →  │ cache │ → DNS 响应
│ (日志)  │    │ (就绪探测)│   │ (K8s 集成)  │    │ (缓存) │
└─────────┘    └─────────┘    └─────────┘    └────────┘
```

### 11.2 K8s 场景下的关键插件

| 插件 | 作用 | 为什么重要 |
|------|------|-----------|
| **kubernetes** | 从 K8s API 读取 Service 和 Pod 信息，生成 DNS 记录 | CoreDNS 的核心价值——自动服务发现 |
| **cache** | 缓存 DNS 查询结果，默认 30s | 减少 API Server 负载，日请求量打对折 |
| **forward** | 将非集群域名（如 `google.com`）转发到上游 DNS | 保证 Pod 能解析外部域名 |
| **errors** | 记录错误日志 | 排查 DNS 问题 |
| **health** | 提供健康检查端点（`:8080/health`） | 配合 K8s liveness probe |
| **ready** | 提供就绪检查端点（`:8181/ready`） | 配合 K8s readiness probe |
| **loop** | 检测 DNS 转发环路 | 防止 CoreDNS 自己转发给自己导致死循环 |
| **reload** | 检测 Corefile 变更并自动重载 | 配置热更新不中断服务 |
| **prometheus** | 暴露 Prometheus metrics | DNS 延迟、请求量监控 |
| **log** | 记录每次 DNS 查询 | 调试时开启 |

---

## 十二、K8s 中的 CoreDNS 部署

### 12.1 部署形态

```bash
# CoreDNS 是一个 Deployment（不是 DaemonSet）
kubectl -n kube-system get deployment coredns

# 默认 2 副本（可扩缩）
kubectl -n kube-system scale deployment coredns --replicas=4

# 前面有一个 Service 做负载均衡
kubectl -n kube-system get svc kube-dns
# NAME       TYPE        CLUSTER-IP    PORT(S)
# kube-dns   ClusterIP   10.96.0.10    53/UDP,53/TCP,9153/TCP
```

> 注意：Service 名仍叫 `kube-dns` 是历史兼容，底层已经是 CoreDNS。

### 12.2 为什么 CoreDNS 是 Deployment 而不是 DaemonSet？

- CoreDNS 是无状态服务，不像 kube-proxy 需要修改本地节点网络规则
- 部署在任意节点上即可，通过 Service 暴露，全集群可达
- 副本数可弹性扩缩

### 12.3 查看 CoreDNS 状态

```bash
# Pod 状态
kubectl -n kube-system get pods -l k8s-app=kube-dns

# Service 信息
kubectl -n kube-system get svc kube-dns

# ConfigMap（Corefile 配置）
kubectl -n kube-system get cm coredns -o yaml
```

---

## 十三、DNS 解析全链路

### 13.1 Pod 内 DNS 配置

```bash
# 在任意 Pod 内查看
$ cat /etc/resolv.conf

nameserver 10.96.0.10
search default.svc.cluster.local svc.cluster.local cluster.local
options ndots:5
```

各字段的含义：

| 字段 | 含义 | 示例 |
|------|------|------|
| `nameserver` | 上游 DNS 服务器 IP，即 CoreDNS Service 的 ClusterIP | `10.96.0.10` |
| `search` | DNS 搜索域，按顺序追加到短名称后面尝试解析 | `my-svc` → 先试 `my-svc.default.svc.cluster.local`，再试 `my-svc.svc.cluster.local` |
| `options ndots:5` | 名称中少于 5 个点 → 先按 search 域尝试；≥ 5 个点 → 直接查询 | `foo`（少于 5 个点）→ 加上 search 域；`www.google.com`（3 个点）→ 又会触发 search 域尝试！ |

### 13.2 `ndots:5` 的坑

这是最容易被忽视的 DNS 性能问题：

```
解析 www.google.com（3 个点 < 5）：
  第 1 次 → www.google.com.default.svc.cluster.local (NXDOMAIN)
  第 2 次 → www.google.com.svc.cluster.local (NXDOMAIN)
  第 3 次 → www.google.com.cluster.local (NXDOMAIN)
  第 4 次 → www.google.com (终于对了)
```

**4 次 DNS 查询才能解析一个外部域名！** 对于频繁访问外部 API 的 Pod，这会导致：

- DNS 查询延迟翻倍
- CoreDNS 负载增倍
- 可能触发 `conntrack` 竞争（UDP DNS）

**解决方案**：

```yaml
# Pod spec 中覆盖 ndots
spec:
  dnsConfig:
    options:
    - name: ndots
      value: "2"          # 外部域名不会有那么多点
```

### 13.3 完整解析示例

Pod A 访问 `my-svc`（位于 `default` namespace）的全链路：

```
① Pod A 内的进程调用: connect("my-svc", 80)
    │
② glibc getaddrinfo() 读取 /etc/resolv.conf
    │  nameserver=10.96.0.10, search=default.svc.cluster.local
    │  "my-svc" 只有 0 个点 < ndots:5 → 追加 search 域
    ▼
③ 发起 DNS A 查询: my-svc.default.svc.cluster.local → 10.96.0.10:53
    │
④ kube-proxy iptables/IPVS 规则拦截: dst=10.96.0.10:53
    │  DNAT 到某个 CoreDNS Pod: 10.244.2.3:53
    ▼
⑤ CoreDNS 收到查询
    │  kubernetes 插件查询本地缓存 → 缓存未命中
    │  kubernetes 插件查 API Server → Service "my-svc" in "default"
    │  返回 ClusterIP: 10.96.0.20
    ▼
⑥ Pod A 收到 DNS 响应: my-svc → 10.96.0.20
    │
⑦ Pod A 发起 TCP 连接: 10.96.0.20:80
    │
⑧ kube-proxy 规则拦截，DNAT 到真实的 Pod IP
    │
⑨ 连接建立！
```

---

## 十四、Corefile 配置详解

### 14.1 默认 Corefile

```bash
# 查看当前配置
kubectl -n kube-system get cm coredns -o jsonpath='{.data.Corefile}'
```

```dns
.:53 {
    errors
    health {
        lameduck 5s
    }
    ready
    kubernetes cluster.local in-addr.arpa ip6.arpa {
        pods insecure
        fallthrough in-addr.arpa ip6.arpa
        ttl 30
    }
    prometheus :9153
    forward . /etc/resolv.conf {
        max_concurrent 1000
    }
    cache 30
    loop
    reload
    loadbalance
}
```

### 14.2 逐行解释

| 配置 | 含义 |
|------|------|
| `.:53` | 监听所有网络接口的 53 端口，处理所有域名的查询 |
| `errors` | 记录错误日志到 stdout |
| `health { lameduck 5s }` | 健康检查端点，关闭前等待 5s 排空请求 |
| `ready` | 就绪检查端点 |
| `kubernetes cluster.local …` | K8s 插件：管理 `cluster.local` 域的 Service/Pod DNS |
| `pods insecure` | 为 Pod 生成 DNS 记录（格式：`pod-ip.service.namespace.pod.cluster.local`） |
| `fallthrough` | 如果 K8s 中没有匹配的记录，交给下一个插件（forward）处理 |
| `ttl 30` | DNS 记录 TTL 为 30 秒 |
| `forward . /etc/resolv.conf` | 非集群域名转发到上游 DNS（节点 `/etc/resolv.conf`） |
| `cache 30` | 缓存正向响应 30 秒 |
| `loop` | 检测转发环路 |
| `reload` | Corefile 变更后自动重载 |
| `loadbalance` | 对 DNS 响应中多个 A 记录进行轮询排序 |

### 14.3 常见自定义

```dns
# 为特定域名指定上游 DNS
forward internal.company.com 10.0.0.53 {
    policy sequential       # 按顺序尝试 DNS 服务器
}

# 自定义 TTL
kubernetes cluster.local in-addr.arpa ip6.arpa {
    ttl 10                  # 缩短 TTL，更快响应 Service 变更
}

# 增加日志（调试时临时开启，生产环境会爆炸）
log . {
    class denial error      # 只记录被拒绝和错误的查询
}

# 为外部域名配置缓存时间
cache 60 {
    denial 5                # NXDOMAIN 等否定缓存 5 秒
}
```

---

## 十五、Service DNS 记录格式

### 15.1 A/AAAA 记录

| 查询 | 返回 |
|------|------|
| `my-svc.default.svc.cluster.local` | Service 的 ClusterIP（1 个 A 记录） |
| `my-svc.default.svc.cluster.local`（Headless） | 所有 Ready Pod 的 IP（多个 A 记录） |

### 15.2 SRV 记录

```
_<port>._<proto>.<svc>.<ns>.svc.cluster.local

# 示例：
_http._tcp.my-svc.default.svc.cluster.local
→ 返回: 0 100 80 my-svc.default.svc.cluster.local
         (优先级 权重 端口 目标主机名)
```

### 15.3 Pod DNS 记录

启用 `pods insecure` 后：

```
# 格式: <pod-ip-with-dashes>.<namespace>.pod.cluster.local
10-244-1-5.default.pod.cluster.local → 10.244.1.5
```

### 15.4 短名称解析规则表

Pod 在 `default` namespace 中查询 `my-svc`：

| 尝试顺序 | 完整域名 | 结果 |
|----------|----------|------|
| 1 | `my-svc.default.svc.cluster.local` | ✅ 命中 |
| — | 如果第 1 次命中，则不再继续 | |

Pod 在 `production` namespace 中查询 `my-svc.default`：

| 尝试顺序 | 完整域名 | 结果 |
|----------|----------|------|
| 1 | `my-svc.default.production.svc.cluster.local` | ❌ |
| 2 | `my-svc.default.svc.cluster.local` | ✅ 命中 |

---

## 十六、Pod DNS 策略

### 16.1 四种 DNS 策略

| 策略 | `/etc/resolv.conf` 来源 | 适用场景 |
|------|------------------------|----------|
| `ClusterFirst`（默认） | K8s 自动生成，nameserver 指向 CoreDNS | 99% 的 Pod |
| `Default` | 继承节点 `/etc/resolv.conf` | Pod 需要绕过 CoreDNS，直接使用宿主机 DNS |
| `ClusterFirstWithHostNet` | 同 ClusterFirst，但用于 hostNetwork Pod | hostNetwork Pod 使用集群 DNS |
| `None` | 完全自定义 | 需要精确控制 DNS 配置 |

```yaml
# 示例：使用 Default 策略
spec:
  dnsPolicy: "Default"

# 示例：完全自定义 DNS
spec:
  dnsPolicy: "None"
  dnsConfig:
    nameservers:
    - 8.8.8.8
    - 1.1.1.1
    searches:
    - mycompany.com
    options:
    - name: ndots
      value: "2"
```

### 16.2 hostNetwork 的 DNS 问题

`hostNetwork: true` 的 Pod 默认继承节点 `/etc/resolv.conf`，里面可能指向了 CoreDNS 的 ClusterIP——但这个 IP 可能在 hostNetwork 命名空间不可达。

```yaml
# hostNetwork Pod 应显式设置 DNS 策略
spec:
  hostNetwork: true
  dnsPolicy: ClusterFirstWithHostNet   # 仍然使用 CoreDNS
```

---

## 十七、CoreDNS 排错

### 17.1 检查 CoreDNS 健康状态

```bash
# Pod 状态
kubectl -n kube-system get pods -l k8s-app=kube-dns -o wide

# 日志
kubectl -n kube-system logs -l k8s-app=kube-dns --tail=100

# 查看 metrics（需要安装 prometheus 插件）
kubectl -n kube-system exec -it deployment/coredns -- curl localhost:9153/metrics
```

### 17.2 在 Pod 内调试 DNS

```bash
# 测试 Pod 内 DNS 解析
kubectl run dns-debug --rm -it --image=busybox:1.28 -- sh

# 在 Pod 内：
nslookup kubernetes.default.svc.cluster.local
nslookup www.google.com
cat /etc/resolv.conf

# 指定 CoreDNS Pod IP 查询（绕过 Service）
nslookup my-svc.default.svc.cluster.local <coredns-pod-ip>
```

### 17.3 常见问题速查

| 现象 | 可能原因 | 排查方法 |
|------|----------|----------|
| 所有 DNS 解析超时 | CoreDNS Pod 全部挂掉 | `kubectl -n kube-system get pods -l k8s-app=kube-dns` |
| 集群内域名解析失败，外部 OK | kubernetes 插件异常 | 查 CoreDNS 日志关键词 `kubernetes` |
| 部分 Pod DNS 解析失败 | conntrack 表满（UDP DNS 丢包） | `conntrack -S` 查看 insert_failed |
| DNS 解析很慢（>1s） | `ndots:5` 导致外部域名触发 search 域 | 检查 `/etc/resolv.conf`；降 `ndots` 到 2 |
| 偶尔 5 秒超时 | Linux 内核 conntrack UDP 竞争 | 降 CoreDNS 副本数、用 TCP DNS |
| NXDOMAIN 但 Service 存在 | namespace 不匹配 | 确认用了正确的 FQDN |
| CoreDNS 频繁重启 | OOM → 扩大 memory limit | `kubectl -n kube-system describe pod -l k8s-app=kube-dns` |

### 17.4 CoreDNS 性能优化

```yaml
# 关键优化参数
spec:
  template:
    spec:
      containers:
      - name: coredns
        resources:
          limits:
            memory: "256Mi"    # 默认 170Mi，大集群需调高
          requests:
            cpu: "100m"
            memory: "70Mi"
        # 调整副本数
      # 通过 HPA 自动扩缩
---
# CoreDNS HPA
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
spec:
  scaleTargetRef:
    name: coredns
  minReplicas: 2
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
```

---

# 第三部分：kube-proxy + CoreDNS 的协作全景

## 十八、一次完整的服务调用链路

```
Pod A (default namespace) 调用 Pod B (my-svc, production namespace)
──────────────────────────────────────────────────────────────────────

Step 1: 域名 → IP（CoreDNS）
  Pod A 的代码: http.Get("http://my-svc.production:8080/health")
    │
    ├── glibc 读 /etc/resolv.conf
    │     nameserver=10.96.0.10 (CoreDNS ClusterIP)
    │     search=default.svc.cluster.local
    │     ndots=5
    │     "my-svc.production" 有 1 个点 < 5 → 追加 search 域
    │     → 先查 "my-svc.production.default.svc.cluster.local" (NXDOMAIN)
    │     → 再查 "my-svc.production.svc.cluster.local" → 命中！
    │
    ├── CoreDNS kubernetes 插件查 API Server → my-svc.production → 10.96.0.20
    │
    └── DNS 响应: 10.96.0.20

Step 2: IP → Pod（kube-proxy）
  Pod A 发出 TCP SYN: dst=10.96.0.20:8080
    │
    ├── 数据包到达节点网络栈
    │
    ├── kube-proxy iptables/IPVS 规则击中
    │     目标 10.96.0.20:8080 → DNAT 到 10.244.2.8:8080 (Pod B 的 IP)
    │     conntrack 记录此连接的 NAT 映射
    │
    ├── 数据包经过 CNI 网络到达 Pod B 所在节点
    │
    └── Pod B 收到请求，处理，返回响应
         返回包经 conntrack 反向 NAT: 10.244.2.8 → 10.96.0.20
         Pod A 看到的源地址仍是 Service ClusterIP

整个过程中 Pod A 只知道 "my-svc.production:8080"，
完全不感知 Pod B 的真实 IP。
```

## 十九、两者职责边界表

| 维度 | kube-proxy | CoreDNS |
|------|-----------|---------|
| **解决的问题** | Service ClusterIP → Pod IP（流量转发） | Service 名称 → ClusterIP（名称解析） |
| **工作层** | L3/L4（IP + 端口） | L7（DNS 协议） |
| **运行方式** | DaemonSet（每节点一个） | Deployment + Service（多副本） |
| **依赖** | API Server（Watch Service + EndpointSlice） | API Server（Watch Service + EndpointSlice + Pod） |
| **生效位置** | 节点的 iptables/IPVS 规则 | 全集群可达的 DNS Service |
| **数据面协议** | TCP / UDP（所有） | UDP 53 / TCP 53 |
| **规则管理** | iptables 全量替换 / IPVS 增量更新 | Corefile 热重载 |
| **性能瓶颈** | conntrack 表、iptables 规则数量 | DNS 请求量、缓存命中率 |
| **如果挂了** | Service ClusterIP 不可达（Pod 之间直连 IP 不受影响） | 集群内 DNS 解析失败（Pod 之间的已有 TCP 连接不受影响） |
| **关键配置** | `proxy-mode` / `ipvs-scheduler` / `conntrack-max` | Corefile 中的 `kubernetes` / `cache` / `forward` 插件 |

### 总结

```
"Pod A 想访问 my-svc"

    CoreDNS 回答: "my-svc 的 IP 是 10.96.0.20"
                               │
    kube-proxy 回答: "10.96.0.20:80 的数据应该发给 10.244.1.3:8080"
                               │
    CNI 网络层回答: "10.244.1.3 在 Node 2 上，走 vxlan 隧道过去"
```

> **CoreDNS 告诉你去哪（服务发现），kube-proxy 帮你走过去（负载均衡转发），CNI 负责把路铺通。**
