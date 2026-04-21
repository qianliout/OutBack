# Kubernetes Pod 状态与排查指南

在 Kubernetes 中，Pod 的生命周期状态主要通过 Phase（阶段）来宏观描述，同时也包含更细粒度的 Conditions（条件）和具体的 Reasons（原因）。理解这些状态对于排查问题和维护集群稳定性至关重要。

## 一、 核心状态 (Phase)

这是 `kubectl get pods` 命令中 STATUS 列显示的主要状态，代表了 Pod 在生命周期中的宏观位置。

| 状态 (Phase) | 含义 | 详细解读 |
| :--- | :--- | :--- |
| **Pending** (挂起) | 已接受，未运行 | Pod 已被创建但未进入 Running 状态。通常因为调度中（无合适节点）或准备中（拉取镜像、挂载存储）。 |
| **Running** (运行中) | 已调度，容器创建 | Pod 已绑定到节点且所有容器已创建。至少一个容器正在运行或启动/重启。注意：不代表应用一定健康。 |
| **Succeeded** (成功) | 所有任务完成 | Pod 中所有容器成功终止（退出码 0），不再重启。常见于 Job。 |
| **Failed** (失败) | 任务执行失败 | Pod 中所有容器已终止，且至少一个异常退出（退出码非 0）。表示应用崩溃或配置错误。 |
| **Unknown** (未知) | 状态失联 | 控制平面无法获取状态。通常因为节点与 API Server 通信中断。 |

## 二、 常见细分状态与原因 (Reasons)

在实际运维中，通过 `kubectl get pods -o wide` 或 `describe` 会看到更具体的状态：

**ContainerCreating**
属于 Pending 阶段。表示 Pod 已调度，Kubelet 正在创建容器环境。长时间卡住通常是镜像拉取慢或存储挂载失败。

**CrashLoopBackOff**
属于异常表现。表示容器启动后立即崩溃退出，Kubelet 尝试重启再次失败，进入退避等待。常见原因为配置错误、端口冲突、健康检查失败。

**ImagePullBackOff / ErrImagePull**
无法拉取镜像。可能是镜像名称错误、私有仓库凭证缺失或网络不通。

**Evicted (被驱逐)**
Pod 因节点资源压力（如磁盘不足、内存压力）被系统强制删除。

**Terminating**
Pod 正在被删除。API Server 已标记删除，Kubelet 正在执行优雅终止流程。

**Init:CrashLoopBackOff / Init:Error**
Init 容器启动失败。Init 容器失败会阻塞整个 Pod 的启动。

## 三、 细粒度健康条件 (Conditions)

通过 `kubectl describe pod` 查看 status.conditions 字段，了解具体健康状况：

**Initialized**: True 表示所有 Init 容器已成功完成。
**Ready**: True 表示 Pod 已准备好接收流量（通过 Readiness Probe）。False 则不会加入 Service 负载均衡。
**ContainersReady**: True 表示 Pod 内所有容器都已就绪。
**PodScheduled**: True 表示 Pod 已成功调度到某个节点。

## 四、 异常排查黄金法则

核心原则：永远先看 `kubectl describe`，再看日志或底层配置。

### 1. 状态：Pending (挂起)
含义：已经被 API Server 接受，但还没有被调度到任何节点上。
排查：执行 `kubectl describe pod <pod-name>` 查看底部 Events。
常见原因 1：提示 Insufficient cpu/memory。说明集群资源不足，需扩容或降低 Pod 资源请求。
常见原因 2：提示 node had taint。说明节点有污点，Pod 无容忍度，需调整调度策略。
常见原因 3：提示 unbound immediate PersistentVolumeClaims。说明依赖的 PVC 还没绑定成功，需检查存储后端。

### 2. 状态：ContainerCreating (创建中卡住)
含义：已调度到节点，但准备环境迟迟无法启动容器。
排查：执行 `kubectl describe pod <pod-name>` 查看 Events。
常见原因 1：提示 Failed to pull image。需检查镜像名、私有仓库凭证、网络。
常见原因 2：提示 MountVolume.SetUp failed。说明存储挂载失败，需检查存储插件或后端服务。
常见原因 3：提示 Failed create pod sandbox。说明 CNI 网络插件异常，无法分配 IP，需检查网络组件。

### 3. 状态：CrashLoopBackOff (无限重启)
含义：容器启动成功但立刻崩溃，陷入重启循环。
排查：先执行 `kubectl logs <pod-name> --previous` 查看上次崩溃日志，再看 describe。
常见原因 1：应用报错退出。如配置缺失、数据库断连，需根据日志修复代码或配置。
常见原因 2：OOMKilled（退出码 137）。内存超限被杀，需调大内存限制或优化代码。
常见原因 3：日志为空。可能是健康检查配置不当（启动慢被误杀）或启动命令写错。

### 4. 状态：ImagePullBackOff
含义：无法下载镜像。
排查：执行 `kubectl describe pod <pod-name>`。
常见原因：镜像名写错、私有仓库未配置 imagePullSecrets、节点无法访问外网仓库。

### 5. 状态：Evicted (被驱逐)
含义：Pod 被节点强行删除。
排查：执行 `kubectl describe pod <pod-name>`。
常见原因：节点磁盘空间不足（DiskPressure）或内存耗尽（MemoryPressure），需清理节点资源。

## 五、 优雅终止过程 (Graceful Shutdown)

直接强制杀掉 Pod 会导致请求报错（502）或数据丢失。Kubernetes 提供了优雅终止机制。

### 1. 终止的完整生命周期

第一步：更新状态与移除流量。API Server 将状态标记为 Terminating，并从 Service Endpoints 中移除 Pod IP，确保不再接收新流量。
第二步：执行 PreStop 钩子。如果配置了 lifecycle.preStop，Kubelet 会同步执行。常用于处理 Service 流量移除的延迟。
第三步：发送 SIGTERM 信号。向容器内主进程发送信号，通知应用开始清理工作并自行退出。
第四步：优雅等待期。Kubelet 开始倒计时，等待时间由 terminationGracePeriodSeconds 定义（默认 30 秒）。
第五步：强制终止。如果超过宽限期进程仍未退出，Kubelet 会发送 SIGKILL 信号强制杀死进程。
第六步：清理资源。彻底释放网络和存储资源，删除 Pod 对象。

### 2. 如何实现优雅关闭

应用层：代码必须捕获并处理 SIGTERM 信号。收到信号后停止接收新请求，等待现有请求处理完毕，关闭数据库连接，最后安全退出。
K8s 配置层 1：调整 terminationGracePeriodSeconds。默认 30 秒可能不够，根据业务耗时建议调整为 60 秒以上。
K8s 配置层 2：配置 PreStop 钩子。由于 Endpoints 移除存在延迟，可在 preStop 中执行 sleep（如 sleep 10），让 Pod 在收到 SIGTERM 前假死几秒，等待流量彻底切断。