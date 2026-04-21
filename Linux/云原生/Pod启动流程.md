# 从 Deployment 到 Pod 的完整启动流程（Kubernetes 最新架构）

Pod 的创建过程是 Kubernetes 核心调度与编排机制的体现。在生产环境中，我们几乎不会单独创建 Pod，而是通过 Deployment 等高级控制器进行管理。从用户提交 Deployment 到最终 Pod 运行，需经历 API 接收、控制器协同、调度、节点执行四大阶段。

## 一、用户提交与控制器协同阶段（从 Deployment 到 Pod）

### 1. 提交 Deployment 定义
用户通过命令或 CI/CD 流水线提交 Deployment 配置：
```bash
kubectl apply -f deployment.yaml
```
`kubectl` 将 YAML/JSON 解析为结构化对象，通过 RESTful API 发送给 API Server。

### 2. API Server 处理请求
*   **认证与授权**：验证用户身份和权限（如 RBAC 检查）。
*   **准入控制**：通过 MutatingWebhook 和 ValidatingWebhook 校验资源配额（LimitRange/ResourceQuota）、安全策略（Pod Security Admission 等）并可能修改默认值。
*   **持久化存储**：将 Deployment 定义写入 etcd 数据库。
*   *关键细节*：etcd 采用 Raft 协议保证数据一致性，是集群状态的唯一真相来源。

### 3. Controller Manager 异步生成 Pod
Kubernetes 采用声明式 API 和异步控制循环机制（Control Loop）：
*   **Deployment Controller**：通过 List-Watch 监听到新的 Deployment 创建，计算所需状态，随后生成对应的 `ReplicaSet` 对象并写入 API Server。
*   **ReplicaSet Controller**：监听到新的 ReplicaSet 创建，发现当前可用 Pod 数量（0）不满足期望副本数（replicas），于是生成对应的 `Pod` 对象（状态为 `Pending`，`nodeName` 为空），并写入 API Server。

## 二、调度阶段：kube-scheduler 的智能决策

### 1. 调度器工作流程
*   **监听机制**：kube-scheduler 通过 List-Watch 持续监听 API Server 上 `nodeName` 为空且状态为 `Pending` 的 Pod 创建事件。
*   **预选阶段（Filter）**：
    *   剔除不满足硬件条件的节点（如 CPU/内存不足，检查 requests）。
    *   检查节点污点（Taint）与 Pod 容忍度（Toleration）是否匹配。
    *   检查节点亲和性（Node Affinity）和 Pod 反亲和性（Pod Anti-Affinity）。
*   **优选阶段（Score）**：
    *   在候选节点中打分（考虑镜像本地缓存、节点空闲资源比例、拓扑分布限制 TopologySpreadConstraints 等）。
    *   选择得分最高的节点（若有多节点同分则随机或轮询选择）。

### 2. 绑定决策（Bind）
*   将调度结果（即将 Pod 的 `spec.nodeName` 字段赋值）更新回 API Server，写入 etcd。
*   *生产排查*：可通过 `kubectl describe pod <pod-name>` 的 Events 列表查看调度器打分和决策详情。

## 三、节点执行阶段：Kubelet 的标准化操作

当 Pod 被绑定到特定 Node 后，目标节点上的 `kubelet` 接管后续工作。

### 1. 节点准入与环境准备
*   **节点准入检查**：Kubelet 会进行第二次资源核对（Node-level Admission），确认节点当前真实资源是否仍足以运行该 Pod（防调度器信息延迟）。
*   **创建 Pod 专属目录**：为 Pod 创建本地数据目录。
*   **存储卷挂载（CSI）**：
    *   若 Pod 定义了 PVC，Kubelet 会通过 CSI（Container Storage Interface）插件调用底层存储（如云盘、Ceph），将远程存储挂载到节点。
    *   将挂载点绑定到 Pod 专用目录：`/var/lib/kubelet/pods/<pod-uid>/volumes/`。

### 2. 创建 Pod 沙箱（Pod Sandbox）
*   Kubelet 通过 CRI（Container Runtime Interface）调用节点上的容器运行时（如 containerd 或 CRI-O）。
*   **启动 Pause 容器**：运行时首先拉取并启动 `pause` 镜像容器。`pause` 容器不执行任何业务逻辑，它的唯一作用是**占据并持有 Pod 的网络命名空间（Network Namespace）、IPC 和 PID 命名空间**，使后续加入的业务容器能共享这些底层隔离环境（实现 `localhost` 互通）。

### 3. 网络配置（CNI）
*   Kubelet 调用 CNI（Container Network Interface）插件（如 Calico, Cilium, Flannel）。
*   CNI 为 Pod 沙箱的网络命名空间分配 IP 地址。
*   配置宿主机上的 veth pair 虚拟网卡、路由和防火墙（iptables/eBPF）规则。

### 4. 容器启动流程（按顺序）
*   **拉取镜像**：Kubelet 根据 `imagePullPolicy` 决定是否通过 CRI 从镜像仓库拉取业务镜像。
*   **执行 Init Containers（初始化容器）**：
    *   如果定义了 Init 容器，它们会严格按照声明顺序依次启动并执行。
    *   只有前一个 Init 容器成功退出（Exit Code 0），下一个才会启动。常用于等待数据库就绪或初始化配置文件。
*   **启动 Main Containers（主业务容器）**：
    *   通过 CRI 接口并行或按序启动所有业务容器。
    *   配置环境变量、注入 ConfigMap/Secret、挂载存储卷。
    *   执行 ENTRYPOINT 和 CMD 指令（以及 postStart 钩子）。

### 5. 探针检测与状态上报
*   **Startup Probe（启动探针）**：*（较新版本引入）* 专为慢启动应用设计。在此探针成功前，禁用另外两种探针。成功后移交控制权。
*   **Liveness Probe（存活探针）**：持续检测容器是否存活，失败则 Kubelet 会重启该容器。
*   **Readiness Probe（就绪探针）**：检测容器是否准备好接收流量。失败则 Endpoint Controller 会将该 Pod IP 从 Service 的后端列表中摘除。
*   **状态闭环**：Kubelet 定期将 Pod 最新状态（`Running`）上报给 API Server 并存入 etcd。

## 四、Deployment -> Pod 创建流程图解

```text
用户 (kubectl / CI Pipeline)
  │
  ▼ [1. Apply Deployment]
API Server ───鉴权/准入───▶ etcd (存储 Deployment)
  │
  ▼ [2. Watch & Create RS]
Deployment Controller (在 kube-controller-manager 中)
  │
  ▼ [3. Watch & Create Pod]
ReplicaSet Controller ────▶ API Server ────▶ etcd (存储 Pending Pod)
  │
  ▼ [4. Schedule Pod]
kube-scheduler ───────────▶ 预选/优选算法 ──▶ etcd (更新 Pod 绑定 Node)
  │
  ▼ [5. Kubelet 执行]
目标 Node 的 Kubelet ──────▶ 发现被绑定到本节点的 Pod
  │
  ├──▶ [5.1] CSI 插件 ────▶ 挂载持久化存储 (Volumes)
  ├──▶ [5.2] CRI 接口 ────▶ 创建 Pause 容器 (Pod Sandbox)
  ├──▶ [5.3] CNI 插件 ────▶ 为 Sandbox 分配 IP，配置网络规则
  ├──▶ [5.4] CRI 接口 ────▶ 依次拉取镜像、运行 Init 容器
  └──▶ [5.5] CRI 接口 ────▶ 运行主容器 (Main Containers)，执行 Startup/Readiness/Liveness 探针
  │
  ▼ [6. 状态就绪]
Kubelet 上报状态 ─────────▶ API Server ────▶ etcd (更新为 Running)
```

## 五、最新架构下的关键组件变迁

*   **CRI（容器运行时接口）**：
    *   *重大改变*：Kubernetes 1.24+ 已彻底移除 `dockershim`。直接把 Docker 作为运行时的时代已经结束。
    *   *现代标准*：目前主流的 CRI 实现是 **`containerd`**（轻量、稳定、Docker 捐献的底层引擎）和 **`CRI-O`**（RedHat 主推，专为 K8s 打造）。
*   **CNI（容器网络接口）**：
    *   当前主流演进方向是基于 eBPF 技术的网络插件（如 **Cilium**），它能提供远超传统 iptables 的网络性能和深度安全可观测性。
*   **CSI（容器存储接口）**：
    *   树外（Out-of-Tree）标准，使得各种云厂商的块存储、文件存储可以独立更新插件版本，无需与 K8s 核心代码绑定。

## 六、常见问题与排查 (基于 Deployment)

*   **Pod 迟迟未出现（甚至没有 Pending 的 Pod）**
    *   *原因*：Deployment 或 RS 的配置有误（如镜像拉取秘钥 imagePullSecrets 缺失导致 RS 校验失败）。
    *   *排查*：`kubectl describe deployment <name>` 或 `kubectl describe rs <name>`
*   **Pod 状态卡在 Pending**
    *   *原因*：集群资源不足、无节点满足 NodeSelector/Affinity、节点均被 Taint 且无容忍度。
    *   *排查*：`kubectl describe pod <name>` (看 Events 里的 Scheduler 报错)
*   **Pod 状态卡在 ContainerCreating**
    *   *原因*：CSI 存储卷挂载失败/超时、CNI 分配 IP 失败、节点网络异常。
    *   *排查*：`kubectl describe pod <name>` 或查看节点日志 `journalctl -u kubelet`
*   **Pod 状态为 ImagePullBackOff**
    *   *原因*：镜像名/Tag写错、私有仓库未配置 Secret、节点无法访问公网。
*   **Pod 状态为 CrashLoopBackOff**
    *   *原因*：应用本身启动报错（如缺环境变量、连不上DB）、Liveness 探针配置不当导致不断重启。
    *   *排查*：`kubectl logs <pod-name>` 或 `kubectl logs <pod-name> --previous`

## 七、生产环境最佳实践

1.  **资源管理（QoS 保障）**：
    *   必须成对设置 `requests` 和 `limits`，避免应用 OOM 或互相挤兑。对于 Java 等应用，建议 CPU limits 不要设得太死，或者使用较新的 `CPU Manager` 策略。
2.  **探针配置（至关重要）**：
    *   **Startup Probe**：启动慢的应用（如老旧 Java SpringBoot）必须配置，防止在启动阶段被 Liveness 探针误杀。
    *   **Readiness Probe**：必须配置。如果不配，Pod 一旦 Running，Service 就会把流量打过来，但此时应用可能还没初始化完连接池，导致 502/504。
3.  **平滑升级与优雅退出（Graceful Shutdown）**：
    *   配置 `preStop` 钩子：在 Pod 被销毁前主动从注册中心下线或处理完堆积请求。
    *   处理 `SIGTERM` 信号：应用代码需要捕获终止信号，优雅关闭连接（而不是被强杀 `SIGKILL`）。
4.  **高可用分布**：
    *   使用 `topologySpreadConstraints` 或 `podAntiAffinity`，确保 Deployment 的多个副本打散在不同的可用区（AZ）或节点上，防止单点故障。
5.  **安全基线**：
    *   使用非 root 用户运行容器（`runAsNonRoot: true`）。
    *   通过 `securityContext.capabilities.drop: ["ALL"]` 移除多余的 Linux 权限。

---
> **总结**：在现代 Kubernetes 体系中，理解从 Deployment 到 Pod 的流转，能帮助我们在更宏观的视角（控制器模式）排查问题。K8s 已经从早期的 Docker 强绑定，走向了 CRI/CNI/CSI 完全标准化的云原生底层，掌握这些接口的职责划分是进阶 K8s 专家的必经之路。
