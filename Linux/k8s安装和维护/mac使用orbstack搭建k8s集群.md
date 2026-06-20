# mac 使用 OrbStack 搭建 K8s 集群

## 1. 目标

这篇笔记的目标不是在 mac 上快速起一个能用的 Kubernetes，而是尽量按照接近生产环境的思路，在 OrbStack 中先创建两台虚拟机，再基于这两台虚拟机手工搭建一个标准的 K8s 集群。

当前阶段先使用 2 个 VM，完成如下目标：

1. 使用 OrbStack 创建两台 Linux 虚拟机。
2. 明确 control-plane 和 worker 的角色划分。
3. 使用更接近生产环境的方式手工安装 Kubernetes。
4. 为后续新增 node、部署业务服务、补齐基础设施保留空间。
5. 后续每个实操章节都详细记录操作步骤、命令、验证方法和常见问题。
6. 后续每一步都说明如果换成真正的 Linux 主机，操作上是否有差异。
7. 让这套集群在 Mac 关机再开机后，能够通过脚本或标准命令快速恢复。

这套方案的定位是：

1. 适合学习 Kubernetes 的标准搭建流程。
2. 适合后续逐步扩容和演进。
3. 不追求一键成功，而是追求结构清晰、路径标准。
4. 不只记录结果，还记录每一步为什么这样做。

## 2. 当前环境

宿主机信息：

1. 机器：MacBook M1
2. 内存：64 GB
3. 虚拟化环境：OrbStack

这个硬件条件足够支撑一个本地的实验型准生产集群。即使后续增加到 3 到 5 个节点，也有较大的操作空间。

需要特别注意的一点：

1. M1 是 ARM64 架构。
2. 后续选择操作系统镜像、容器镜像、中间件镜像时，要优先确认是否支持 ARM64。

## 3. 为什么不用一键方案

像 kind、minikube、k3d 这类工具非常适合快速体验 Kubernetes，但不适合当前这个学习目标。

当前更适合的路线是：

1. 先有虚拟机。
2. 再配置基础系统。
3. 再安装容器运行时。
4. 再使用 kubeadm 初始化集群。
5. 再安装网络插件、入口组件和业务服务。

这样做的好处是：

1. 能真正理解节点、容器运行时、网络插件、控制平面的关系。
2. 后续增加 node 时，操作方式和真实环境更接近。
3. 出问题时更容易定位问题所在。

## 4. 集群设计

当前阶段使用 2 台虚拟机，规划如下：

1. `k8s-master-01`
2. `k8s-worker-01`

角色分工如下：

1. `k8s-master-01` 作为 control-plane 节点，负责运行 apiserver、controller-manager、scheduler、etcd 等核心组件。
2. `k8s-worker-01` 作为 worker 节点，负责承载业务 Pod。

这个阶段的集群特点如下：

1. 是单 control-plane 架构。
2. 不是高可用架构。
3. 但已经足够学习标准安装、节点加入、服务部署、网络接入等核心内容。

## 5. 为什么先做 2 个 VM

先用 2 个 VM 有几个现实优势：

1. 资源消耗更可控。
2. 拓扑足够简单，适合第一次完整走通搭建流程。
3. 已经能体现主节点和工作节点的职责差异。
4. 后续可以继续学习如何新增 worker 节点。

需要明确的是：

1. 2 节点集群不具备真正生产级高可用能力。
2. 当前重点是先学会标准安装和标准扩容。

## 6. 推荐的软件选型

为了更接近生产环境，建议采用下面的技术组合：

1. 操作系统：Ubuntu Server 24.04 ARM64
2. 容器运行时：containerd
3. Kubernetes 安装方式：kubeadm
4. CNI 网络插件：Calico
5. Ingress：ingress-nginx
6. 软件包管理：Helm

这套组合的原因如下：

1. Ubuntu Server 文档多，社区成熟，适合学习。
2. containerd 是当前主流选择，比 Docker 更贴近现在的 Kubernetes 运行时实践。
3. kubeadm 是学习标准集群搭建过程的最好入口。
4. Calico 是生产环境中常见的 CNI 方案。
5. ingress-nginx 适合本地实验环境，资料也足够丰富。

## 7. 资源规划建议

建议先按下面的资源分配进行：

1. `k8s-master-01`：4 vCPU，8 GB 内存，80 GB 磁盘
2. `k8s-worker-01`：4 vCPU，8 GB 内存，80 GB 磁盘

这只是起步配置，后面可以根据实际情况调整。

原因如下：

1. control-plane 需要运行多个核心组件，资源不能太小。
2. worker 节点后面还要承载业务服务，8 GB 内存会更从容。
3. 80 GB 磁盘能避免后续拉镜像、跑中间件时空间过紧。

如果后续要增加更多服务，例如数据库、监控、日志系统，可以继续追加内存和磁盘。

## 8. 网络规划建议

虽然是本地实验环境，也建议一开始就保持网络规划意识。

建议提前确定下面几件事：

1. 两台虚拟机的固定主机名。
2. 两台虚拟机的固定 IP。
3. Pod 网段。
4. Service 网段。
5. 后续对外暴露服务的方式。

规划原则如下：

1. 节点 IP 不要频繁变化。
2. Pod 网段不要和宿主机常用网段冲突。
3. Service 网段也不要和现有网络重复。
4. 后续如果要加 MetalLB，要预留可以分配的地址范围。

## 9. 搭建路线

建议按下面的顺序推进，而不是一上来就装 Kubernetes。

### 第一步：创建两台 VM

这一阶段要完成：

1. 选定统一的 Linux 发行版。
2. 创建 `k8s-master-01` 和 `k8s-worker-01`。
3. 给两台机器分配明确的 CPU、内存和磁盘。
4. 确认两台机器之间网络互通。

### 第二步：初始化操作系统

这一阶段要完成：

1. 配置主机名。
2. 配置时区。
3. 配置静态或稳定 IP。
4. 配置 `/etc/hosts`。
5. 安装常用系统工具。
6. 关闭 swap。
7. 配置内核参数和网络转发参数。

这一层非常关键，因为 kubeadm 安装前的很多问题都出在系统基础配置上。

### 第三步：安装 containerd

这一阶段要完成：

1. 安装 containerd。
2. 生成并调整 containerd 配置。
3. 配置 systemd cgroup 驱动。
4. 启动并设置开机自启。

这里的目标是让 Kubernetes 使用标准的容器运行时，而不是依赖一层额外的 Docker 兼容逻辑。

### 第四步：安装 kubeadm、kubelet、kubectl

这一阶段要完成：

1. 配置 Kubernetes 软件源。
2. 安装 kubeadm。
3. 安装 kubelet。
4. 安装 kubectl。
5. 锁定版本，避免自动升级带来不一致。

### 第五步：初始化 control-plane

在 `k8s-master-01` 上完成：

1. 使用 kubeadm init 初始化集群。
2. 明确 Pod 网段参数。
3. 保存 join 命令。
4. 配置当前用户的 kubeconfig。

### 第六步：安装 CNI 网络插件

这一阶段建议从 `Calico` 开始，因为上手更稳，排障资料也更多。

完成这一阶段后：

1. 节点状态会变成 Ready。
2. Pod 网络才真正可用。

### 第七步：让 worker 加入集群

在 `k8s-worker-01` 上执行 kubeadm join。

完成后需要验证：

1. 节点是否成功加入。
2. 节点状态是否正常。
3. CoreDNS 等基础组件是否正常运行。

### 第八步：安装基础能力组件

基础能力建议按下面顺序逐步补齐：

1. Metrics Server
2. Helm
3. ingress-nginx
4. MetalLB
5. StorageClass

注意：

1. 这些组件不需要第一天全部装完。
2. 更合理的方式是先让集群稳定，再逐步补齐。

### 第九步：部署一个真实服务

当基础组件稳定后，再开始部署业务服务。

建议从简单样例开始，例如：

1. 一个 Go API 服务
2. 一个前端服务
3. 一个 MySQL 或 PostgreSQL
4. 一个 Redis

然后逐步学习：

1. Deployment
2. Service
3. ConfigMap
4. Secret
5. Ingress
6. PVC

## 10. 当前阶段不追求的内容

为了让第一版方案更稳，当前先不追求下面这些能力：

1. 多 control-plane 高可用
2. 外部 etcd 集群
3. 完整的服务网格
4. 完整的日志平台
5. 完整的监控告警平台
6. 自动化 CI CD

这些内容都可以在后续集群稳定后再逐步增加。

## 11. 当前阶段的风险和限制

目前这套方案有几个天然限制：

1. 只有一个 control-plane，主节点故障后集群控制面会不可用。
2. 本地环境和真实云环境在网络、存储、负载均衡方面仍然有差异。
3. 某些镜像可能只对 AMD64 支持更完整，需要优先确认 ARM64 兼容性。
4. 如果后续部署较多中间件，本地资源虽然够用，但仍要关注磁盘和内存消耗。

## 12. 后续扩容思路

你后面还要学习怎么加 node，所以现在这套方案要提前预留扩容路径。

后续增加 worker 节点的核心流程大致如下：

1. 在 OrbStack 中创建新的 Linux VM。
2. 完成和现有节点一致的系统初始化。
3. 安装 containerd。
4. 安装 kubeadm 和 kubelet。
5. 在 master 上生成新的 join 命令。
6. 让新节点执行 kubeadm join。
7. 在集群中验证节点状态和调度情况。

也就是说，当前这篇笔记的第一阶段目标，是先把标准模板跑通。后面你每增加一个 node，本质上都是重复一套标准化流程。

## 13. 我对这套实验环境的定义

这套环境更准确的定义是：

1. 本地实验型准生产 Kubernetes 集群
2. 强调标准安装流程
3. 强调后续可扩容
4. 强调理解 Kubernetes 核心组件和节点职责

它不是：

1. 真正的生产高可用集群
2. 一键式体验环境

## 14. 下一步计划

接下来建议按下面顺序继续完善这篇笔记：

1. 先补充 OrbStack 中创建两台 VM 的具体操作步骤。
2. 再补充每台 Ubuntu 的初始化步骤。
3. 再补充 containerd 安装和配置。
4. 再补充 kubeadm 初始化集群的完整命令。
5. 最后补充 Ingress、MetalLB 和示例服务部署。

## 15. 结论

当前方案是合理的，而且非常适合作为学习路线的起点。

先做 2 个 VM，有三个明显优势：

1. 足够接近标准生产搭建流程。
2. 不会因为复杂度过高导致第一阶段卡死。
3. 后续新增 node 的路径非常自然。

下一篇内容可以直接进入实操部分，从 OrbStack 创建 `k8s-master-01` 和 `k8s-worker-01` 开始。

## 16. 后续文档编写规则

从下一节开始，这篇笔记不再只写“做什么”，而要尽量写成“可以直接照着做”的操作手册。

后续每一个实操章节都按下面的结构展开：

1. 本步骤的目标。
2. 在 OrbStack 虚拟机中的具体操作步骤。
3. 需要执行的完整命令。
4. 每条关键命令执行后的验证方法。
5. 常见报错或排查方向。
6. 如果是在真正的 Linux 主机上，和当前环境有什么不同。

也就是说，后面每一节都要尽量回答下面几个问题：

1. 为什么做这一步。
2. 具体在哪台机器上执行。
3. 具体执行什么命令。
4. 执行成功后应该看到什么现象。
5. 如果失败了，应该先检查哪里。
6. 如果不是 OrbStack，而是真实 Linux 主机，是否需要调整。

建议后续所有实操内容都带上下面这种说明方式：

1. 执行节点：`k8s-master-01` 或 `k8s-worker-01`
2. 执行身份：普通用户或 `root`
3. 作用说明：这一条命令解决什么问题
4. 预期结果：执行后应该看到什么
5. 真机差异：真实 Linux 是否一致

后续文档的一个重要原则是：

1. 在 OrbStack 虚拟机里执行的命令，尽量和真实 Linux 主机保持一致。
2. 这样后面从本地实验环境迁移到真实环境时，学习成本最低。

## 17. OrbStack VM 和真实 Linux 主机的主要差异

从 Kubernetes 节点内部看，大部分安装步骤其实是相同的。

也就是说，下面这些内容在 OrbStack 虚拟机里和真实 Linux 主机上，通常没有本质区别：

1. 安装 containerd
2. 安装 kubeadm、kubelet、kubectl
3. 关闭 swap
4. 配置内核参数
5. kubeadm init
6. kubeadm join
7. 安装 CNI
8. 部署 Deployment、Service、Ingress

真正有差异的主要在下面几个方面：

1. 节点的创建方式不同。
2. 节点的网络环境不同。
3. 节点的开机和恢复方式不同。
4. 存储方案的能力边界不同。
5. 对外暴露服务的方式可能不同。

具体来说：

1. 在 OrbStack 中，节点是通过创建虚拟机得到的。
2. 在真实 Linux 环境中，节点可能是物理机、云主机，或者由其他虚拟化平台创建的 VM。
3. 在 OrbStack 中，宿主机是 mac，关机后需要先恢复 OrbStack 和 VM。
4. 在真实 Linux 环境中，通常由物理服务器、云平台或虚拟化平台负责主机开机和 VM 自启动。
5. 在真实环境中，往往还会有更标准的负载均衡、块存储、网络隔离和安全策略。

所以这篇笔记后续每一步都要明确区分两层：

1. 节点内部的 Linux 操作是否通用。
2. 节点外部的基础设施行为是否依赖 OrbStack。

## 18. Mac 关机后快速恢复集群的目标

你提出的这个需求非常重要，因为它会直接影响这套集群是否适合长期使用。

当前要达到的目标不是“每次开机后重新搭一个集群”，而是：

1. Mac 开机后先恢复 OrbStack。
2. 再恢复两台 VM。
3. 再确认 containerd 和 kubelet 正常运行。
4. 再确认 control-plane 和 worker 节点恢复正常。
5. 再确认集群中的基础服务和业务服务自动恢复。

理想状态下，恢复过程应该尽量缩短为：

1. 启动 OrbStack
2. 启动两个 VM
3. 运行一个检查或恢复脚本
4. 观察集群和业务状态

## 19. 为了快速恢复，当前阶段就要提前做的设计

如果想在 Mac 关机后快速恢复，当前搭建阶段必须提前做好下面这些事情。

### 19.1 节点服务必须开机自启

在两台 Linux 节点中，至少要确保这些服务能够在节点启动后自动拉起：

1. `containerd`
2. `kubelet`

对应命令如下：

```bash
sudo systemctl enable containerd
sudo systemctl enable kubelet
```

这一步非常关键，因为 kubeadm 搭建的 control-plane 组件本质上依赖 kubelet 拉起静态 Pod。

如果 `kubelet` 没有自动恢复，那么 apiserver、controller-manager、scheduler、etcd 都不会正常回来。

### 19.2 集群资源必须声明式管理

后续部署业务服务时，尽量不要靠手工临时执行命令维持状态，而要把资源定义保存下来。

建议至少做到：

1. 所有 Kubernetes YAML 清单统一存放。
2. 所有 Helm 安装命令有固定 values 文件。
3. 关键中间件的配置文件可重复执行。

这样在极端情况下，即使某些资源异常，也可以快速重新执行：

```bash
kubectl apply -f <manifests-directory>
helm upgrade --install <release-name> <chart-name> -f <values-file>
```

### 19.3 数据不要只依赖临时目录

后续如果要部署数据库、Redis 或其他有状态服务，需要尽量避免把重要数据放在容易丢失的位置。

当前阶段至少要明确：

1. VM 磁盘本身要保留，不能随意删 VM。
2. Kubernetes 的持久化卷方案要提前规划。
3. 后续如果使用本地盘或 NFS，要清楚数据生命周期。

### 19.4 恢复动作尽量分层

建议把恢复动作拆成三层，而不是混成一个黑盒脚本。

第一层是宿主机层：

1. 启动 OrbStack。
2. 启动目标 VM。

第二层是节点层：

1. 检查 `containerd`。
2. 检查 `kubelet`。
3. 检查节点网络。

第三层是集群层：

1. 检查节点是否 Ready。
2. 检查系统 Pod 是否恢复。
3. 检查 Ingress、MetalLB、业务服务是否恢复。

## 20. 建议的恢复流程

下面是一套适合当前双节点实验环境的恢复思路。

### 20.1 宿主机恢复

当 Mac 开机后，先恢复 OrbStack 和虚拟机。

可以先查看 OrbStack 提供的命令能力：

```bash
orbctl status
orbctl list
```

如果需要启动 OrbStack 或启动指定机器，可以使用类似下面的命令：

```bash
orbctl start
orbctl start k8s-master-01
orbctl start k8s-worker-01
```

说明：

1. `orbctl start` 用于启动 OrbStack。
2. `orbctl start <machine-name>` 用于启动指定虚拟机。

### 20.2 节点服务恢复

虚拟机启动后，检查每个节点内部的关键服务。

可以分别进入机器执行：

```bash
orb -m k8s-master-01 sudo systemctl status containerd --no-pager
orb -m k8s-master-01 sudo systemctl status kubelet --no-pager
orb -m k8s-worker-01 sudo systemctl status containerd --no-pager
orb -m k8s-worker-01 sudo systemctl status kubelet --no-pager
```

如果服务没有起来，可以执行：

```bash
orb -m k8s-master-01 sudo systemctl restart containerd
orb -m k8s-master-01 sudo systemctl restart kubelet
orb -m k8s-worker-01 sudo systemctl restart containerd
orb -m k8s-worker-01 sudo systemctl restart kubelet
```

### 20.3 集群状态恢复检查

在 master 节点检查 Kubernetes 状态：

```bash
orb -m k8s-master-01 kubectl get nodes -o wide
orb -m k8s-master-01 kubectl get pods -A
orb -m k8s-master-01 kubectl get svc -A
```

重点观察：

1. 两个节点是否都是 `Ready`
2. `kube-system` 命名空间下的核心组件是否都正常
3. CoreDNS、CNI、Ingress 控制器是否已恢复

### 20.4 业务服务恢复

如果你的业务服务采用 Deployment、StatefulSet、DaemonSet 并且配置完整，理论上在节点恢复后，大部分服务会自动恢复。

如果你把所有资源都保存成清单或 Helm values，则还可以通过下面的命令进行一次声明式校准：

```bash
orb -m k8s-master-01 kubectl apply -f <manifests-directory>
orb -m k8s-master-01 helm upgrade --install <release-name> <chart-name> -f <values-file>
```

## 21. 恢复脚本示例

下面给出一个适合当前实验环境的恢复脚本示例。后面等节点名、清单目录和 Helm 发布名最终确定后，可以再细化。

脚本示例文件名可以叫：

`restore-k8s-lab.sh`

```bash
#!/usr/bin/env bash
set -euo pipefail

echo "[1/5] start OrbStack"
orbctl start

echo "[2/5] start VMs"
orbctl start k8s-master-01
orbctl start k8s-worker-01

echo "[3/5] check node services"
orb -m k8s-master-01 sudo systemctl is-active --quiet containerd || orb -m k8s-master-01 sudo systemctl start containerd
orb -m k8s-master-01 sudo systemctl is-active --quiet kubelet || orb -m k8s-master-01 sudo systemctl start kubelet
orb -m k8s-worker-01 sudo systemctl is-active --quiet containerd || orb -m k8s-worker-01 sudo systemctl start containerd
orb -m k8s-worker-01 sudo systemctl is-active --quiet kubelet || orb -m k8s-worker-01 sudo systemctl start kubelet

echo "[4/5] wait and verify cluster"
sleep 15
orb -m k8s-master-01 kubectl get nodes -o wide
orb -m k8s-master-01 kubectl get pods -A

echo "[5/5] reconcile manifests if needed"
# orb -m k8s-master-01 kubectl apply -f /home/ubuntu/k8s-manifests
# orb -m k8s-master-01 helm upgrade --install ingress-nginx ingress-nginx/ingress-nginx -f /home/ubuntu/values/ingress-nginx.yaml

echo "restore finished"
```

这个脚本的核心作用不是重新安装集群，而是：

1. 启动宿主机上的虚拟化环境。
2. 启动虚拟机。
3. 检查关键系统服务。
4. 校验 Kubernetes 状态。
5. 必要时重新 apply 清单。

## 22. 如果换成真正的 Linux 主机，恢复方式有什么不同

如果将来不是在 mac + OrbStack 上运行，而是在真正的 Linux 主机或云主机上运行，恢复逻辑的核心不变，但入口会不同。

主要差异如下：

1. 不再需要 `orbctl start` 这类 OrbStack 命令。
2. 如果是物理机，通常由机器本身的开机过程恢复。
3. 如果是云主机，通常由云平台负责实例启动。
4. 如果是其他虚拟化平台，通常由 Proxmox、VMware、KVM 等平台负责 VM 自启动。
5. 节点内部依然是检查 `containerd`、`kubelet`、`kubectl get nodes` 这一套逻辑。

也就是说：

1. OrbStack 相关命令只属于宿主机基础设施层。
2. Linux 节点内部的恢复动作，本质上和真实环境是高度一致的。

## 23. 后续实操章节的新增要求

从下一节开始，每个操作章节都要额外补充两块内容：

1. 本步骤在真实 Linux 主机上的差异说明。
2. 本步骤对“关机后快速恢复”的影响说明。

例如：

1. 在安装 containerd 时，要明确说明为什么必须 `enable`。
2. 在安装 kubelet 时，要明确说明为什么必须让它随系统自动恢复。
3. 在部署业务时，要明确说明哪些资源应该用 YAML 或 Helm 固化。
4. 在部署有状态服务时，要明确说明数据持久化路径和恢复边界。
