# Kubernetes CrashLoopBackOff 故障排查完全指南

## 概述

CrashLoopBackOff 是 Kubernetes 中最常见的容器故障状态之一，表示容器启动后异常退出，kubelet 持续尝试重启的循环现象。**本质上这是 Kubernetes 的自我保护机制在生效**，而非故障本身。

理解这一点的关键：CrashLoopBackOff 不是 bug，而是 kubelet 检测到容器不健康后的正常响应。排查的目标是找到容器退出的真正原因。

---

## 第一章：核心概念与状态区分

### 1.1 容器生命周期状态

```
Container States:
├── Waiting         # 容器未运行（正在启动或被阻止）
├── Running         # 容器正在执行
└── Terminated      # 容器已完成执行
```

### 1.2 相关状态快速区分

| 状态 | 含义 | 排查重点 |
|------|------|----------|
| `CrashLoopBackOff` | 容器反复崩溃重启 | 应用层问题（日志、配置、资源） |
| `ImagePullBackOff` | 镜像拉取失败 | 镜像地址、凭证、仓库访问 |
| `ErrImagePull` | 镜像拉取错误 | 镜像名称、网络连通性 |
| `CreateContainerConfigError` | 配置错误 | ConfigMap/Secret 引用 |
| `InvalidImageName` | 镜像名称无效 | 镜像标签是否正确 |

### 1.3 退出码完整解读

```bash
kubectl get pod <pod-name> -n <namespace> -o jsonpath='{.status.containerStatuses[0].lastState.terminated.exitCode}'
```

| 退出码 | 含义 | 根因指向 |
|--------|------|----------|
| `0` | 正常退出 | 应用启动后主动退出，未保持运行 |
| `1` | 通用错误 | 应用内部异常 |
| 137 (128+9) | SIGKILL | 系统 OOM，被强制终止 |
| 139 (128+11) | SIGSEGV | 段错误，内存越界访问 |
| 143 (128+15) | SIGTERM | 优雅终止（正常重启流程） |
| 255 | 入口点错误 | 启动命令或参数错误 |

---

## 第二章：快速定位与信息收集

### 2.1 全局扫描异常 Pod

```bash
# 快速定位所有非 Running 状态的 Pod
kubectl get pods --all-namespaces --field-selector=status.phase!=Running

# 按重启次数排序，优先排查高频重启
kubectl get pods -A --sort-by='.status.containerStatuses.restartCount' | tail -20

# 针对特定 namespace
kubectl get pods -n <namespace> --sort-by='.status.containerStatuses.restartCount'

# 使用 label 筛选关键业务 Pod
kubectl get pods -n <namespace> -l app=<app-name> --sort-by='.status.containerStatuses.restartCount'
```

### 2.2 获取详细事件信息

```bash
# 完整事件输出（必查）
kubectl describe pod <pod-name> -n <namespace> | grep -A 30 "Events:"

# 精确获取 Events 部分
kubectl get events -n <namespace> --field-resolver involvedObject.name=<pod-name> --sort-by='.lastTimestamp'
```

**关键错误关键词**：
- `FailedCreatePodSandBox` → CNI/网络问题
- `FailedMount` → 存储挂载问题
- `OOMKilled` → 内存不足
- `Error` → 应用层异常
- `Unhealthy` → 探针检测失败

### 2.3 容器状态深度检查

```bash
# 查看容器最后终止状态详情
kubectl get pod <pod-name> -n <namespace> -o json | jq '.status.containerStatuses[] | {name, lastState, state}'

# 查看特定容器的重启历史
kubectl get pod <pod-name> -n <namespace> -o jsonpath='{range .status.containerStatuses[*]} Container: {.name}, Restart: {.restartCount}, Last Exit: {.lastState.terminated.exitCode}{"\n"}{end}'

# 查看容器资源实际使用（需 metrics-server）
kubectl top pod <pod-name> -n <namespace> --containers
```

### 2.4 日志获取（关键步骤）

```bash
# 当前日志
kubectl logs <pod-name> -n <namespace>

# 上一次崩溃的日志（最重要！）
kubectl logs <pod-name> -n <namespace> --previous

# 指定容器（多容器 Pod）
kubectl logs <pod-name> -n <namespace> -c <container-name> --previous

# 带时间戳和上下文
kubectl logs <pod-name> -n <namespace> --previous --tail=200 --timestamps

# 搜索关键错误上下文
kubectl logs <pod-name> -n <namespace> --previous | grep -A 30 -B 10 "Exception\|Error\|FATAL"
```

---

## 第三章：系统性排查路径

### 3.1 排查优先级决策树

```
CrashLoopBackOff 发生
    │
    ├─► Events 中有 OOMKilled？
    │       └─► 是 → 跳至「资源限制排查」
    │
    ├─► Events 中有 Mount 失败？
    │       └─► 是 → 跳至「存储与配置排查」
    │
    ├─► Events 中有探针失败记录？
    │       └─► 是 → 跳至「健康检查排查」
    │
    └─► Events 无明确错误
            └─► 跳至「应用层深度排查」
```

### 3.2 资源限制排查（首要检查项）

#### 内存问题

```bash
# 检查 Pod 资源限制配置
kubectl get pod <pod-name> -n <namespace> -o jsonpath='{.spec.containers[0].resources}'

# 检查内存限制与实际需求对比
kubectl describe pod <pod-name> -n <namespace> | grep -A 5 "Limits\|Requests"

# 节点内存压力
kubectl get nodes -o json | jq -r '.items[] | {name: .metadata.name, allocatable: .status.allocatable.memory, capacity: .status.capacity.memory}'
```

**典型症状**：
- `limits.memory` 设置 512Mi，但应用需要 1Gi → 必然 OOMKilled
- 容器实际内存使用接近 limits.memory 90% 以上
- dmesg 中有 oom-kill 记录

**解决方案**：
```yaml
resources:
  limits:
    memory: "1Gi"  # 设置为实际峰值的 1.2-1.5 倍
  requests:
    memory: "512Mi"  # 保障最小需求
```

#### CPU 问题

```bash
# 检查 CPU 限制
kubectl describe pod <pod-name> -n <namespace> | grep -E "cpu:|CPU"

# 查看 CPU 节流情况（高 CPU throttling 是常见问题）
kubectl exec -it <pod-name> -n <namespace> -- cat /sys/fs/cgroup/cpu/cpu.stat
```

**典型症状**：
- CPU limits 过低导致应用初始化超时
- 高并发场景下 CPU throttling 严重
- Java 应用（特别是 JVM）需要更长预热时间

### 3.3 健康检查探针排查（高频根因）

#### 探针配置检查

```bash
# 获取完整 Deployment 配置
kubectl get deployment <app-name> -n <namespace> -o yaml

# 检查探针配置（关键参数）
kubectl get pod <pod-name> -n <namespace> -o jsonpath='{.spec.containers[0].livenessProbe}'
kubectl get pod <pod-name> -n <namespace> -o jsonpath='{.spec.containers[0].readinessProbe}'
kubectl get pod <pod-name> -n <namespace> -o jsonpath='{.spec.containers[0].startupProbe}'
```

#### 探针参数详解

| 参数 | 作用 | 常见错误 |
|------|------|----------|
| `initialDelaySeconds` | 启动后多久开始探测 | 设置过短，应用未就绪就开始探测 |
| `periodSeconds` | 探测间隔 | 设置过频，增加负载 |
| `timeoutSeconds` | 超时时间 | 设置过短，慢查询被误杀 |
| `failureThreshold` | 连续失败次数 | 设置过小，非永久性故障被误判 |
| `successThreshold` | 成功后视为就绪 | livenessProbe 必须为 1 |

#### 典型问题场景

**场景 1：Java/.NET 等启动慢的应用**
```yaml
# 错误配置：应用需 60 秒启动，但 initialDelaySeconds=30
livenessProbe:
  httpGet:
    path: /health
    port: 8080
  initialDelaySeconds: 30  # ❌ 太短
  periodSeconds: 10

# 正确配置
startupProbe:  # 专门针对启动慢的应用
  httpGet:
    path: /health
    port: 8080
  initialDelaySeconds: 0
  periodSeconds: 5
  failureThreshold: 30  # 最多等待 150 秒
```

**场景 2：探针路径依赖外部服务**
```yaml
# ❌ 错误：/health 依赖数据库连接
livenessProbe:
  httpGet:
    path: /health
    port: 8080

# ✅ 正确：探针应只检测自身状态
livenessProbe:
  httpGet:
    path: /health/ready  # 只检查应用自身状态
    port: 8080
```

### 3.4 存储与配置依赖排查

#### 挂载状态检查

```bash
# 检查 Volumes 和 Mounts 配置
kubectl describe pod <pod-name> -n <namespace> | grep -A 20 "Mounts:\|Volumes:"

# 检查 PVC 绑定状态
kubectl get pvc -n <namespace>

# 检查 PV 状态
kubectl get pv <pv-name>

# 验证挂载点是否可访问
kubectl exec -it <pod-name> -n <namespace> -- ls -la /config
kubectl exec -it <pod-name> -n <namespace> -- df -h | grep config
```

#### 高频问题清单

| 问题类型 | 症状 | 解决方案 |
|----------|------|----------|
| ConfigMap 未挂载 | Events: "MountVolume.SetUp failed" | 检查 ConfigMap 名称和 namespace |
| Secret 未挂载 | "Invalid secret" | 验证 Secret 存在且类型正确 |
| PVC 未绑定 | "Pending" 状态超过 5 分钟 | 检查 StorageClass 和 PV 供给 |
| 权限不足 | "Permission denied" | 检查 securityContext 和 fsGroup |
| 只读文件系统 | "Read-only file system" | 检查 volumeMounts readOnly 设置 |

#### 嵌入式组件验证

```bash
# 验证关键端口是否监听（以 Milvus 为例）
kubectl exec -it <pod-name> -n <namespace> -- netstat -tuln | grep -E '2379|9000|6650'

# 验证 etcd 健康状态（嵌入式 etcd）
kubectl exec -it <pod-name> -n <namespace> -- etcdctl endpoint health

# 检查进程状态
kubectl exec -it <pod-name> -n <namespace> -- ps aux | grep -E 'etcd|minio|postgres'
```

### 3.5 网络问题排查

#### CNI 状态检查

```bash
# 检查 CNI 插件状态（以 Calico 为例）
calicoctl node status

# 检查 IP 分配情况
calicoctl ipam show --show-blocks

# 检查 iptables 规则数量（过多影响性能）
iptables-save | wc -l  # 超过 10 万条需优化

# 检查节点网络配置
kubectl get nodes -o wide
ip route show
ip addr show
```

#### 网络连通性验证

```bash
# Pod 内部健康检查
kubectl exec -it <pod-name> -n <namespace> -- curl -k https://localhost:8080/health

# Pod 间连通性测试
kubectl exec -it <pod-name> -n <namespace> -- ping <target-pod-ip>

# DNS 解析测试
kubectl exec -it <pod-name> -n <namespace> -- nslookup <service-name>
kubectl exec -it <pod-name> -n <namespace> -- dig <service-name>

# 端口连通性测试
kubectl exec -it <pod-name> -n <namespace> -- nc -zv <target-ip> <port>
```

#### NetworkPolicy 检查

```bash
# 检查是否有 NetworkPolicy 限制
kubectl get networkpolicy -n <namespace>

# 检查特定 Pod 的策略
kubectl describe pod <pod-name> -n <namespace> | grep -i policy
```

### 3.6 应用层深度排查

#### 交互式调试

```bash
# 基础调试 shell
kubectl exec -it <pod-name> -n <namespace> -- /bin/sh

# 使用 netshoot 工具进行高级调试
kubectl debug -it <pod-name> -n <namespace> --image=nicolaka/netshoot -- /bin/bash

# 在新节点启动调试 Pod
kubectl debug node <node-name> --image=nicolaka/netshoot -it -- cpulimit=0.5
```

#### 环境变量检查

```bash
# 查看所有环境变量
kubectl exec -it <pod-name> -n <namespace> -- env | sort

# 验证关键环境变量
kubectl exec -it <pod-name> -n <namespace> -- env | grep -E "PROFILE|ENV|MODE|HOSTNAME|PORT"

# 检查 ConfigMap/Secret 注入的环境变量
kubectl get pod <pod-name> -n <namespace> -o json | jq '.spec.containers[0].env'
```

#### 启动命令与权限检查

```bash
# 检查容器的启动命令
kubectl get pod <pod-name> -n <namespace> -o jsonpath='{.spec.containers[0].command}'

# 检查工作目录和文件系统
kubectl exec -it <pod-name> -n <namespace> -- ls -la /app
kubectl exec -it <pod-name> -n <namespace> -- pwd

# 检查用户和权限
kubectl exec -it <pod-name> -n <namespace> -- id
kubectl exec -it <pod-name> -n <namespace> -- cat /etc/passwd

# 检查日志目录写权限
kubectl exec -it <pod-name> -n <namespace> -- touch /var/log/test.log && rm /var/log/test.log
```

---

## 第四章：高级排查技巧

### 4.1 临时调试配置

当容器反复重启无法调试时，可注入临时容器：

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: <pod-name>-debug
  namespace: <namespace>
spec:
  containers:
  - name: debug
    image: busybox
    command: ["sleep", "3600"]
    resources: {}
  - name: main
    # 原有容器配置
```

使用：
```bash
kubectl debug <pod-name> -n <namespace> --share-processes --copy-to=debug-pod
kubectl exec -it debug-pod -n <namespace> -c main -- /bin/sh
```

### 4.2 系统内核日志检查

```bash
# 检查 OOM Killer 日志
dmesg -T | grep -i "out of memory"
journalctl -k | grep -i "oom-kill"

# 检查容器运行时日志
journalctl -u kubelet | tail -1000
journalctl -u containerd | tail -1000

# 检查特定容器的 cgroup 信息
cat /sys/fs/cgroup/memory/kubepods/burstable/pod<uid>/memory.oom_control
```

### 4.3 资源历史监控

```bash
# Prometheus 查询示例
# 容器内存使用率
container_memory_usage_bytes{pod="<pod-name>", namespace="<namespace>"} / container_spec_memory_limit_bytes{pod="<pod-name>", namespace="<namespace>"}

# 内存使用超过 80% 的 Pod
sum by (pod, namespace) (container_memory_usage_bytes) / sum by (pod, namespace) (container_spec_memory_limit_bytes) > 0.8
```

### 4.4 批量操作技巧

```bash
# 批量获取非 Running 状态 Pod 的事件
for pod in $(kubectl get pods -A --field-selector=status.phase!=Running -o jsonpath='{.items[*].metadata.name}'); do
  ns=$(kubectl get pod $pod -o jsonpath='{.metadata.namespace}')
  echo "=== $ns/$pod ==="
  kubectl describe pod $pod -n $ns | grep -E "Warning|Error|Failed" | head -5
done

# 批量查看重启次数最高的 Pod
kubectl get pods -A -o custom-columns='NAMESPACE:.metadata.namespace,NAME:.metadata.name,RESTARTS:.status.containerStatuses[0].restartCount,STATE:.status.phase' --sort-by='.RESTARTS' | tail -20
```

---

## 第五章：预防与最佳实践

### 5.1 配置设计原则

#### 资源限制设计

```yaml
# ✅ 推荐：基于实际监控数据设置
resources:
  requests:
    memory: "256Mi"
    cpu: "100m"
  limits:
    memory: "512Mi"  # 设置为 requests 的 2 倍
    cpu: "500m"

# ❌ 避免：无限制或随意设置
resources: {}
# 或
resources:
  limits:
    memory: "128Mi"  # 远低于实际需求
```

#### 探针配置规范

```yaml
# startupProbe：处理启动慢的应用（必需 for Java/Go/.NET）
startupProbe:
  httpGet:
    path: /ready
    port: 8080
  failureThreshold: 30  # 30 * periodSeconds(10) = 300s 最大启动时间
  periodSeconds: 10

# livenessProbe：检测应用是否存活（应简单，避免复杂检查）
livenessProbe:
  httpGet:
    path: /health/live
    port: 8080
  initialDelaySeconds: 30
  periodSeconds: 15
  failureThreshold: 3

# readinessProbe：检测应用是否就绪接收流量
readinessProbe:
  httpGet:
    path: /health/ready
    port: 8080
  initialDelaySeconds: 5
  periodSeconds: 10
  failureThreshold: 3
```

### 5.2 镜像管理规范

```bash
# ✅ 推荐：使用明确版本标签
image: myapp:1.2.3

# ❌ 避免：使用 latest 或无标签
image: myapp:latest
image: myapp

# 定期扫描镜像漏洞
trivy image myapp:1.2.3
```

### 5.3 部署前验证

```bash
# 本地模拟容器运行环境
docker run --rm \
  -v $(pwd)/config:/app/config:ro \
  -m 256m \
  --memory-swap=256m \
  --cpus=0.5 \
  myapp:1.2.3

# 验证健康检查端点
curl -f http://localhost:8080/health/live || exit 1
curl -f http://localhost:8080/health/ready || exit 1

# 验证日志输出到 stdout
docker logs <container-id>  # 应能看到应用日志
```

### 5.4 高可用设计

```yaml
# 配置 PodDisruptionBudget
apiVersion: policy/v1
kind: PodDisruptionBudget
metadata:
  name: app-pdb
  namespace: <namespace>
spec:
  minAvailable: 2  # 至少保持 2 个 Pod 可用
  selector:
    matchLabels:
      app: myapp

# 使用反亲和性分布 Pod
affinity:
  podAntiAffinity:
    requiredDuringSchedulingIgnoredDuringExecution:
    - labelSelector:
        matchLabels:
          app: myapp
      topologyKey: kubernetes.io/hostname
```

### 5.5 监控与告警体系

#### 关键告警规则

| 告警名称 | 条件 | 严重级别 |
|----------|------|----------|
| PodMemoryHighUsage | memory usage > 80% of limit | Warning |
| PodRestartHigh | restarts > 10 in 5min | Warning |
| PodNotReady | NotReady status > 2min | Critical |
| ContainerOOMKilled | OOMKilled event | Critical |

#### 推荐监控指标

```promql
# Pod 重启次数
rate(kube_pod_container_status_restarts_total[5m])

# 容器内存使用
container_memory_usage_bytes

# 容器 CPU throttling
rate(container_cpu_cfs_throttled_seconds_total[5m])

# Pod 状态分布
kube_pod_status_phase
```

---

## 第六章：常见场景快速修复

### 场景 1：Java 应用内存溢出

**症状**：退出码 137，dmesg 有 OOM 记录

**修复**：
```yaml
# 增加 JVM 堆内存限制（留出 off-heap 空间）
env:
- name: JAVA_OPTS
  value: "-Xmx512m -Xms256m"  # 不要用满 container memory limit
resources:
  limits:
    memory: "768Mi"  # JVM 堆 + 元空间 + 直接内存
```

### 场景 2：Python 应用启动超时

**症状**：探针 initialDelaySeconds 过短

**修复**：
```yaml
startupProbe:
  httpGet:
    path: /health
    port: 8000
  failureThreshold: 60  # 60 * 5s = 300s 启动时间
  periodSeconds: 5
```

### 场景 3：Node.js 应用端口冲突

**症状**：退出码 1，日志显示 "EADDRINUSE"

**修复**：
```yaml
env:
- name: PORT
  value: "8080"  # 确保环境变量与监听端口一致
```

### 场景 4：Go 应用多进程信号处理

**症状**：主进程无法正确处理 SIGTERM

**修复**：
```go
// 确保正确处理 SIGTERM
signal.Notify(sigCh, syscall.SIGTERM, syscall.SIGINT)
<-sigCh
// 优雅关闭
server.Shutdown(ctx)
```

### 场景 5：.NET Core 应用堆内存限制

**症状**：容器内存接近限制时被杀

**修复**：
```yaml
resources:
  limits:
    memory: "512Mi"
env:
- name: DOTNET_gcHeapHardLimit
  value: "420000000"  # 设置 GC 堆硬限制
```

---

## 总结

### 排查口诀

```
一查 Events 找线索
二看退出码定方向
三验资源防 OOM
四查探针防误杀
五检存储防挂载
六审日志找异常
```

### 核心原则

1. **CrashLoopBackOff 是自我保护机制**：容器在告诉你"我有问题"
2. **90% 的根因藏在 `--previous` 日志中**：不要只查看当前日志
3. **按优先级排查**：资源限制 > 探针配置 > 存储挂载 > 应用日志
4. **优雅设计是根本**：让应用正确报告自身健康状态

### 云原生应用设计精髓

最终目标不是"避免 CrashLoopBackOff"，而是让应用能够：

- **优雅启动**：正确配置 startupProbe，给足初始化时间
- **优雅关闭**：正确处理 SIGTERM，清理资源
- **准确报告**：健康检查只检测自身状态，不依赖外部依赖
- **资源可控**：明确声明资源需求，不贪婪不浪费

遵循这些原则，CrashLoopBackOff 将从"噩梦"变成"帮手"，成为保障应用可靠性的第一道防线。
