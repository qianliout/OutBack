# K8s 集群默认组件全览

> 一个全新的、可用的 K8s 集群启动后，到底跑了哪些组件？每个组件干什么？互相怎么协作？这篇笔记做一个完整的梳理。

---

## 目录

- [一、总览](#一总览)
- [二、控制平面组件](#二控制平面组件)
  - [2.1 kube-apiserver](#21-kube-apiserver)
  - [2.2 etcd](#22-etcd)
  - [2.3 kube-scheduler](#23-kube-scheduler)
  - [2.4 kube-controller-manager](#24-kube-controller-manager)
  - [2.5 cloud-controller-manager（可选）](#25-cloud-controller-manager可选)
- [三、节点组件](#三节点组件)
  - [3.1 kubelet](#31-kubelet)
  - [3.2 kube-proxy](#32-kube-proxy)
  - [3.3 容器运行时](#33-容器运行时)
- [四、插件组件](#四插件组件)
  - [4.1 CoreDNS](#41-coredns)
  - [4.2 CNI 网络插件](#42-cni-网络插件)
- [五、组件间协作：一次 Pod 创建的完整链路](#五组件间协作一次-pod-创建的完整链路)
- [六、如何查看集群中运行了哪些组件](#六如何查看集群中运行了哪些组件)
- [七、组件分布速查表](#七组件分布速查表)
- [八、小结](#八小结)

---

## 一、总览

```
                     ┌──────────────────────────────┐
                     │       Control Plane 节点       │
                     │                              │
                     │  ┌────────────┐              │
                     │  │kube-apiserver│◀── 一切入口  │
                     │  └──────┬─────┘              │
                     │         │                    │
                     │    ┌────▼────┐               │
                     │    │  etcd    │  ← 状态存储   │
                     │    └─────────┘               │
                     │                              │
                     │  ┌──────────────────┐        │
                     │  │kube-controller-   │        │
                     │  │    manager        │        │
                     │  └──────────────────┘        │
                     │                              │
                     │  ┌──────────────────┐        │
                     │  │ kube-scheduler    │        │
                     │  └──────────────────┘        │
                     └──────────────────────────────┘

  ┌──────────── Worker Node 1 ────────────┐   ┌──────────── Worker Node 2 ────────────┐
  │                                        │   │                                        │
  │  ┌──────────┐  ┌──────────────┐       │   │  ┌──────────┐  ┌──────────────┐       │
  │  │ kubelet  │  │ kube-proxy   │       │   │  │ kubelet  │  │ kube-proxy   │       │
  │  └──────────┘  └──────────────┘       │   │  └──────────┘  └──────────────┘       │
  │  ┌──────────────────────────┐        │   │  ┌──────────────────────────┐        │
  │  │  容器运行时 (containerd)  │        │   │  │  容器运行时 (containerd)  │        │
  │  └──────────────────────────┘        │   │  └──────────────────────────┘        │
  └──────────────────────────────────────┘   └──────────────────────────────────────┘
```

---

## 二、控制平面组件

控制平面组件负责集群的全局决策和状态管理。它们通常运行在 Master 节点上，以 **Static Pod** 的形式由 kubelet 直接管理。

> **Static Pod 是什么？** 不经过 API Server 调度，而是由 kubelet 直接读取 `/etc/kubernetes/manifests/` 下的 YAML 文件来启动的 Pod。这样即使 API Server 挂了，控制平面组件也能被 kubelet 独立重启。

### 2.1 kube-apiserver

**一句话**：集群的总入口。所有对集群的操作（kubectl 命令、控制器查询、调度器绑定、kubelet 上报）都必须经过它。

| 维度 | 说明 |
|------|------|
| **核心职责** | 提供 REST API，对所有请求做认证（AuthN）、授权（AuthZ）、准入控制（Admission） |
| **数据流向** | 它是唯一直接读写 etcd 的组件——其他所有组件都通过它间接操作状态 |
| **运行方式** | Static Pod，`kube-system` namespace 中可见 |
| **典型端口** | 6443（HTTPS） |
| **水平扩展** | 支持多副本（配合负载均衡器），无状态 |

**关键机制：**

```
kubectl apply -f pod.yaml
        │
        ▼
  kube-apiserver
    ├── 1. 认证 (AuthN)：你是谁？（证书/Token/ServiceAccount）
    ├── 2. 授权 (AuthZ)：你有权限吗？（RBAC）
    ├── 3. 准入控制 (Admission)：请求合法吗？需要修改吗？（Mutating + Validating Webhooks）
    ├── 4. 写入 etcd
    └── 5. 返回结果，同时通知所有 Watcher（controller-manager、scheduler、kubelet）
```

> 详细机制见同一目录下的 [认证与授权](认证与授权.md) 笔记。

**命令验证：**

```bash
kubectl get pod -n kube-system | grep apiserver
kubectl get --raw /healthz          # 检查 API Server 健康
kubectl get --raw /livez            # 检查存活
```

---

### 2.2 etcd

**一句话**：集群的"大脑数据库"。存储所有 K8s 对象的期望状态和实际状态。

| 维度 | 说明 |
|------|------|
| **核心职责** | 分布式 KV 存储，存储集群所有配置和状态数据（Pod/Service/Node/ConfigMap/Secret……） |
| **数据一致性** | Raft 协议，通常 3 或 5 个节点，奇数个避免脑裂 |
| **运行方式** | Static Pod 或外部独立集群 |
| **典型端口** | 2379（客户端）、2380（peer 通信） |
| **特点** | 唯一有状态的控制平面组件，数据持久化在磁盘 |

**关键机制：**

- 任何 `kubectl apply` 或控制器修改，最终都变成 etcd 的一次写入。
- Watch 机制：组件不轮询 etcd，而是建立长连接，etcd 在数据变化时**主动推送**。这是 K8s 声明式 API 高效运转的基础。
- 备份 = 备份 etcd。只要 etcd 数据在，集群可以重建。

```
kubectl apply → apiserver → etcd 写入
                                │
                                ▼
                   Watch 事件推送给所有监听者:
                   controller-manager、scheduler、kubelet
```

**命令验证：**

```bash
# 查看 etcd Pod
kubectl get pod -n kube-system | grep etcd

# 检查 etcd 健康（需要在 control plane 节点上）
ETCDCTL_API=3 etcdctl \
  --endpoints=https://127.0.0.1:2379 \
  --cacert=/etc/kubernetes/pki/etcd/ca.crt \
  --cert=/etc/kubernetes/pki/etcd/server.crt \
  --key=/etc/kubernetes/pki/etcd/server.key \
  endpoint health
```

---

### 2.3 kube-scheduler

**一句话**：Pod 的"婚介所"。为新创建的 Pod 选择最合适的 Node。

| 维度 | 说明 |
|------|------|
| **核心职责** | 监听 `nodeName` 为空的新 Pod，通过预选（Filtering）+ 优选（Scoring）算法选出最佳 Node |
| **工作方式** | 它不直接启动 Pod，只把 Node 名字写回 Pod 的 `spec.nodeName`，然后 kubelet 接管 |
| **运行方式** | Static Pod |
| **典型端口** | 10259（HTTPS） |
| **水平扩展** | 支持多副本（通过 Leader Election，同时只有一个在工作） |

**调度两步走：**

```
1. 预选 (Filtering / Predicate)
   ├── Node 资源够吗？（CPU、内存）
   ├── Node 的 selector 匹配吗？
   ├── Node 有污点吗？（Taints & Tolerations）
   └── 端口冲突吗？
         │
         ▼ 通过预选的 Node 列表
2. 优选 (Scoring / Priority)
   ├── 哪个 Node 剩余资源最多？（LeastRequestedPriority）
   ├── 哪个 Node 上已有该 Pod 的同类镜像？（ImageLocality）
   ├── 是否满足 Pod 亲和性/反亲和性？
   └── 得分最高者 → 更新 Pod.spec.nodeName
```

> 详细机制见同一目录下的 [Pod启动流程](Pod启动流程.md) 和 [亲和性](亲和性.md) 笔记。

**命令验证：**

```bash
kubectl get pod -n kube-system | grep scheduler
kubectl describe pod -n kube-system kube-scheduler-<node>  # 查看调度器配置
```

---

### 2.4 kube-controller-manager

**一句话**：集群的"自动化总管"。运行一系列**控制器**，每个控制器负责将集群的"实际状态"向"期望状态"调谐。

| 维度 | 说明 |
|------|------|
| **核心职责** | 运行一组控制器，持续执行 Reconcile Loop（观察 → 对比 → 修正） |
| **运行方式** | Static Pod，单个二进制包含所有控制器 |
| **典型端口** | 10257（HTTPS） |
| **水平扩展** | 支持多副本（Leader Election） |

**内置控制器清单（部分重要者）：**

| 控制器 | 职责 | 典型行为 |
|--------|------|---------|
| **Node Controller** | 监控节点健康状态 | Node 失联 → 标记 NotReady → 驱逐 Pod |
| **Deployment Controller** | 管理 Deployment → ReplicaSet 的创建和更新 | 创建新 RS，滚动更新 |
| **ReplicaSet Controller** | 保证指定数量的 Pod 副本始终运行 | Pod 少了 → 创建；Pod 多了 → 删除 |
| **Service Controller** | 为 Service 创建 EndpointSlice | 匹配 Label Selector，收集 Pod IP |
| **Job Controller** | 管理 Job/CronJob | Job 完成 → 标记状态；CronJob → 按 cron 表达式创建 Job |
| **PV Controller** | 将 PersistentVolumeClaim 绑定到 PersistentVolume | PVC 创建 → 找匹配的 PV → 绑定 |
| **Namespace Controller** | 管理 Namespace 生命周期 | Namespace 删除 → 级联删除旗下所有资源 |
| **ServiceAccount Controller** | 为每个 Namespace 创建默认 ServiceAccount | 新 Namespace → 自动创建 `default` SA |
| **TTL Controller** | 清理过期的已完成 Job/Pod | 过了 TTL → 删除 |

**Reconcile Loop 工作原理：**

```
期望状态（etcd 中记录的 spec）         实际状态（kubelet 上报的 status）
        │                                        │
        └──────────── Controller 对比 ────────────┘
                          │
                   ┌──────┴──────┐
                   │  不一致！   │
                   ├────有动作───►│  创建/删除/更新资源
                   │  一致       │
                   └──什么也不做─┘
```

**命令验证：**

```bash
kubectl get pod -n kube-system | grep controller-manager

# 查看控制器列表
kubectl get pod -n kube-system kube-controller-manager-<node> -o yaml \
  | grep -A 50 'spec:' | grep '\-controllers='
```

---

### 2.5 cloud-controller-manager（可选）

**一句话**：连接 K8s 和云厂商 API 的适配层。只有部署在公有云上时才需要。

| 控制器 | 职责 |
|--------|------|
| **Node Controller** | 通过云 API 检查被删除的虚拟机，清理对应 Node 对象 |
| **Route Controller** | 在云 VPC 中配置路由 |
| **Service Controller** | 为 LoadBalancer 类型的 Service 创建云负载均衡器 |

> 自建集群（bare-metal / 虚拟机）通常**没有**这个组件，其部分功能由 `kube-controller-manager` 中的 Node Controller 和 Service Controller 承担。

---

## 三、节点组件

节点组件运行在**每个 Node**（包括 Master 节点）上，负责 Pod 的实际运行。

### 3.1 kubelet

**一句话**：每个 Node 上的"工头"，负责执行 Master 发来的 Pod 指令。

| 维度 | 说明 |
|------|------|
| **核心职责** | ① 接收 API Server 分配的 Pod 清单；② 调用 CRI 创建容器；③ 调用 CNI 配置网络；④ 调用 CSI 挂载存储；⑤ 执行探针（Liveness/Readiness/Startup）；⑥ 上报 Pod 状态 |
| **运行方式** | 系统服务（`systemctl` 管理），不是 Pod |
| **典型端口** | 10250（API）、10255（只读，已废弃）、10248（healthz） |
| **特点** | 唯一不跑在容器里的核心组件——它负责启动容器，但不能自己跑在容器里（鸡生蛋问题） |

**kubelet 的工作循环：**

```
kubelet 启动
  │
  ├── 1. 注册 Node 到 API Server
  │
  └── 2. 进入主循环（周期性）：
        ├── 从 API Server 拉取绑定到本节点的 Pod 列表
        ├── 对比本地运行的容器
        │   ├── 多的 → 删除
        │   ├── 少的 → 创建（CRI → CNI → CSI → start）
        │   └── 变的 → 更新
        ├── 执行探针（Liveness / Readiness）
        ├── 上报 Pod 和 Node 状态到 API Server
        └── 上报资源使用情况（cAdvisor 内置）
```

> 详细机制见同一目录下的 [Pod启动流程](Pod启动流程.md)、[Pod状态](Pod状态.md)、[Cgroups](Cgroups.md) 笔记。

**命令验证：**

```bash
systemctl status kubelet
journalctl -u kubelet -f           # 实时日志

# 查看该 Node 上的所有 Pod（包括 Static Pod）
ls /etc/kubernetes/manifests/
```

---

### 3.2 kube-proxy

**一句话**：每个 Node 上的"网络调度员"，负责将 Service ClusterIP 的流量转发到后端 Pod。

| 维度 | 说明 |
|------|------|
| **核心职责** | Watch Service + EndpointSlice 变化，在节点上写入 iptables 或 IPVS 规则，实现 Service → Pod 的负载均衡 |
| **运行方式** | DaemonSet（每节点一个 Pod） |
| **工作模式** | iptables（默认）、IPVS（推荐大规模集群） |
| **特点** | 只转发，不响应；ClusterIP 只是一个虚拟 IP，只存在于 iptables/IPVS 规则中 |

```
Service: my-svc (ClusterIP 10.96.0.20:80)
  后端 Pod:
    Pod A: 10.244.1.3:8080
    Pod B: 10.244.2.8:8080

请求 10.96.0.20:80 到达 Node →
  iptables/IPVS 规则拦截 →
    随机选 Pod A 或 Pod B →
      DNAT: 目标 IP 改为 Pod IP
```

> 详细机制见同一目录下的 [kube-proxy与CoreDNS](kube-proxy与CoreDNS.md) 笔记。

**命令验证：**

```bash
kubectl get pod -n kube-system -l k8s-app=kube-proxy
kubectl logs -n kube-system kube-proxy-<pod> --tail=20
```

---

### 3.3 容器运行时

**一句话**：真正创建和运行容器进程的底层组件。

| 维度 | 说明 |
|------|------|
| **核心职责** | 接收 CRI 调用，拉取镜像、创建容器、启动/停止/删除容器 |
| **运行方式** | 系统服务 + 每个容器一个 shim 进程 |
| **典型实现** | containerd（最常用）、CRI-O |
| **已被移除** | Docker Engine（K8s 1.24 起） |

```
kubelet ── CRI gRPC ── containerd ── containerd-shim ── runc ── 容器进程
```

> 详细机制见同一目录下的 [containerd](containerd.md) 笔记。

**命令验证：**

```bash
systemctl status containerd

# 直接调用 CRI 查看容器列表
crictl ps
crictl pods
```

---

## 四、插件组件

插件组件不是 K8s 内核的一部分，但**一个"可用"的集群几乎必然包含它们**。它们以 Pod 形式运行在集群中。

### 4.1 CoreDNS

**一句话**：集群内部的 DNS 服务器，提供 **服务发现**（Service 名称 → ClusterIP 的解析）。

| 维度 | 说明 |
|------|------|
| **核心职责** | 为每个 Service 创建 DNS 记录，让 Pod 可以通过 `<service>.<namespace>.svc.cluster.local` 访问服务 |
| **运行方式** | Deployment（通常 2 副本），通过 Service 暴露（`kube-dns`） |
| **典型端口** | 53（UDP/TCP） |
| **插件架构** | `errors` / `health` / `kubernetes` / `cache` / `forward` / `reload` |

**如果没有它**：Pod 之间只能用 IP 地址通信，所有服务发现逻辑要自己实现。

**Pod 中 `/etc/resolv.conf` 示例：**

```
nameserver 10.96.0.10          # ← CoreDNS Service 的 ClusterIP
search default.svc.cluster.local svc.cluster.local cluster.local
```

> 详细机制见同一目录下的 [kube-proxy与CoreDNS](kube-proxy与CoreDNS.md) 笔记。

**命令验证：**

```bash
kubectl get deploy -n kube-system coredns
kubectl get svc -n kube-system kube-dns
kubectl logs -n kube-system -l k8s-app=kube-dns --tail=30
```

---

### 4.2 CNI 网络插件

**一句话**：给 Pod 分配 IP，打通跨节点网络。

| 维度 | 说明 |
|------|------|
| **核心职责** | ① 为每个 Pod 分配唯一 IP；② 实现跨 Node 的 Pod 互通；③ 实现 NetworkPolicy（部分插件） |
| **运行方式** | DaemonSet + 各节点上的 CNI 二进制 |
| **典型实现** | Flannel（简单）、Calico（功能全、NetworkPolicy）、Cilium（eBPF、高性能） |

**如果不装 CNI 插件**：Pod 一直卡在 `ContainerCreating` 状态——kubelet 调用 CNI 失败。

```
kubelet 创建 Pod
  └── CRI: 容器已创建
  └── CNI: 为 Pod 分配 IP、配置网络接口
        │
        ├── 插件为本节点创建 veth pair + 路由
        └── 如果 Pod 在另一节点，插件负责跨节点隧道/路由
```

> 详细机制见同一目录下的 [k8s网络详细笔记](k8s网络详细笔记.md) 笔记。

**命令验证：**

```bash
# 根据使用的 CNI 不同，命令略有差异
kubectl get pod -n kube-system | grep -E 'calico|flannel|cilium'
```

---

## 五、组件间协作：一次 Pod 创建的完整链路

```
用户: kubectl apply -f nginx-deploy.yaml

  [1] kube-apiserver
      ├── 认证 → 授权 → 准入控制
      ├── 写入 etcd: Deployment{replicas: 3}
      └── 通知 Watcher → kube-controller-manager

  [2] Deployment Controller (kube-controller-manager 内)
      ├── 收到 Deployment 创建事件
      ├── 创建 ReplicaSet 对象 → apiserver → etcd
      └── 通知 → ReplicaSet Controller

  [3] ReplicaSet Controller (kube-controller-manager 内)
      ├── 收到 RS 创建事件
      ├── 创建 3 个 Pod 对象（spec 中 nodeName 为空）→ apiserver → etcd
      └── 通知 → kube-scheduler

  [4] kube-scheduler
      ├── Watch 到 nodeName 为空的新 Pod
      ├── 预选 → 优选 → 选出最佳 Node
      ├── 更新 Pod.spec.nodeName → apiserver → etcd
      └── 通知 → 目标 Node 的 kubelet

  [5] kubelet (目标 Node)
      ├── Watch 到绑定到本节点的 Pod
      ├── CRI → 调用 containerd 拉取镜像、创建容器
      ├── CNI → 调用网络插件分配 IP
      ├── CSI → 挂载存储（如有 PVC）
      ├── 启动容器
      └── 上报 Pod 状态: Running → apiserver → etcd

  [6] kube-proxy (对应 Service 存在时)
      ├── Watch 到新 Pod 的 EndpointSlice 变化
      └── 更新 iptables / IPVS 规则，将新 Pod IP 加入负载均衡

  Pod 正式提供服务
```

**信息流方向总结：**

```
用户/控制器 → apiserver → etcd (写入期望状态)
                                    │
                              Watch 通知
                                    │
          ┌──────────────┬──────────┼──────────┬──────────────┐
          ▼              ▼          ▼          ▼              ▼
     controller-     scheduler   kubelet     kube-proxy    CoreDNS
      manager
          │              │          │
          │              │          └── 上报实际状态 → apiserver → etcd
          │              │
          └── 向 etcd 写入新对象 ──→ apiserver → etcd
```

---

## 六、如何查看集群中运行了哪些组件

```bash
# 1. 控制平面组件（Static Pod，-n kube-system 可见）
kubectl get pod -n kube-system -o wide | grep -E 'apiserver|etcd|scheduler|controller'

# 2. 节点组件
kubectl get pod -n kube-system -l k8s-app=kube-proxy -o wide   # kube-proxy (DaemonSet)
systemctl status kubelet                                        # kubelet (系统服务)
crictl ps                                                       # 容器运行时管理的容器

# 3. 插件组件
kubectl get deploy -n kube-system coredns                       # CoreDNS
kubectl get pod -n kube-system | grep -E 'calico|flannel|cilium|weave'  # CNI

# 4. 汇总
kubectl get all -n kube-system
```

---

## 七、组件分布速查表

| 组件 | 哪来的 | 跑在控制平面 Node？ | 跑在 Worker Node？ | 运行形式 | 如果挂了会怎样 |
|------|--------|:---:|:---:|------|------|
| **kube-apiserver** | K8s 核心 | ✅ | — | Static Pod | 集群完全不可操作，已有容器不受影响 |
| **etcd** | K8s 核心 | ✅ | — | Static Pod | 无法写入新状态，已有容器不受影响（etcd 多节点时个别挂无害） |
| **kube-scheduler** | K8s 核心 | ✅ | — | Static Pod | 新 Pod 无法被调度（Pending），已有 Pod 不受影响 |
| **kube-controller-manager** | K8s 核心 | ✅ | — | Static Pod | 副本数不再保持、Service 不再更新 EndpointSlice…… |
| **cloud-controller-manager** | 可选 | ✅ | — | Static Pod / Deployment | 云资源操作失效（LB 创建/路由配置等） |
| **kubelet** | K8s 核心 | ✅ | ✅ | 系统服务 | 该 Node 脱离管理，Pod 无法创建/更新/健康检查 |
| **kube-proxy** | K8s 核心 | ✅ | ✅ | DaemonSet | Service ClusterIP 不可达，Pod 间直连 IP 不受影响 |
| **容器运行时** | 第三方 | ✅ | ✅ | 系统服务 | 该 Node 上所有容器停止响应，kubelet 报错 |
| **CoreDNS** | K8s 插件 | — | ✅ | Deployment | 集群内 DNS 解析失败（已有 TCP 连接不受影响） |
| **CNI 插件** | 第三方 | — | ✅ | DaemonSet + 二进制 | 新 Pod 卡在 ContainerCreating，已有 Pod 网络不受影响 |

---

## 八、小结

一个"全新可用的 K8s 集群"最少需要这些组件：

```
         控制平面（通常 3 个 Node）
         ├── kube-apiserver     ← 入口，所有操作必经之路
         ├── etcd                ← 状态存储，集群的"真相来源"
         ├── kube-scheduler      ← 为 Pod 选 Node
         └── kube-controller-manager ← 维持期望状态

         每个 Node（含控制平面 Node）
         ├── kubelet             ← 执行 Pod 指令，管理容器生命周期
         ├── kube-proxy          ← Service ClusterIP → Pod IP 转发
         └── 容器运行时           ← 真正创建/运行容器进程

         集群插件（以 Pod 运行）
         ├── CoreDNS             ← 服务发现（名称 → IP）
         └── CNI 网络插件        ← Pod 网络（分配 IP + 跨节点互通）
```

**记住一条核心原则**：只有 **apiserver** 直接写 etcd，其他所有组件都通过 apiserver 间接操作状态。这条原则解释了 K8s 架构中所有"为什么要这样设计"的问题。
