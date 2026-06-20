---
title: mac 使用 OrbStack 搭建 K8s 集群实操安装手册
tags:
  - k8s
  - kubernetes
  - orbstack
  - mac
  - arm64
aliases:
  - OrbStack 双节点 K8s 实操手册
---

# mac 使用 OrbStack 搭建 K8s 集群实操安装手册

## 1. 文档目标

这篇笔记记录的是一次真实执行过的安装过程，不是纯理论方案。

这篇文档要解决的问题：

1. 在 MacBook M1 上使用 OrbStack 准备两个 Linux 节点。
2. 用 `kubeadm + containerd + Calico` 搭建一个双节点 Kubernetes 集群。
3. 记录每一步的命令、目的、验证方法和预期结果。
4. 记录本次过程中真实遇到的问题和解决办法。
5. 说明如果换成真正的 Linux 主机，哪些地方会不同。
6. 总结在 Linux 主机上安装时容易踩的坑。

## 2. 本次安装结果

本次最终安装结果如下：

1. 集群类型：单 control-plane，单 worker。
2. Kubernetes 版本：`v1.33.13`
3. 容器运行时：`containerd 2.2.1`
4. CNI：`Calico v3.30.4`
5. 最终状态：两个节点均为 `Ready`

本次最终节点状态如下：

```text
NAME            STATUS   ROLES           AGE     VERSION
k8s-master-01   Ready    control-plane   5m29s   v1.33.13
k8s-worker-01   Ready    <none>          4m23s   v1.33.13
```

本次最终系统 Pod 状态如下：

```text
NAME                                       READY   STATUS    RESTARTS   AGE
calico-kube-controllers-6cb4cc57b9-f7f28   1/1     Running   0          57s
calico-node-5rmxc                          1/1     Running   0          57s
calico-node-lc6mx                          1/1     Running   0          57s
coredns-674b8bbfcf-57tbv                   1/1     Running   0          5m22s
coredns-674b8bbfcf-6lpkd                   1/1     Running   0          5m22s
etcd-k8s-master-01                         1/1     Running   0          5m29s
kube-apiserver-k8s-master-01               1/1     Running   0          5m29s
kube-controller-manager-k8s-master-01      1/1     Running   0          5m29s
kube-proxy-c96sk                           1/1     Running   0          4m23s
kube-proxy-lgjwv                           1/1     Running   0          5m22s
kube-scheduler-k8s-master-01               1/1     Running   0          5m29s
```

## 3. 本次环境信息

宿主机环境：

1. 机器：MacBook M1
2. 内存：64 GB
3. 虚拟化工具：OrbStack

本次节点规划：

1. `k8s-master-01`
2. `k8s-worker-01`

本次节点 IP：

1. `k8s-master-01`：`192.168.139.167`
2. `k8s-worker-01`：`192.168.139.177`

本次实际使用的软件组合：

1. 操作系统：Ubuntu 24.04 ARM64
2. 容器运行时：containerd
3. 集群初始化方式：kubeadm
4. 网络插件：Calico

## 4. 非常重要的前置说明

虽然本文标题写的是“创建两个虚拟机”，但 OrbStack 的 Linux machine 从系统内部看更接近容器化 Linux machine，而不是传统意义上的完整虚拟机。

本次在节点内部查看到的信息如下：

```text
Virtualization: lxc
Operating System: Ubuntu 24.04.4 LTS
Architecture: arm64
```

这意味着：

1. 用它学习 `kubeadm`、节点初始化、容器运行时、CNI 和服务部署是没有问题的。
2. 但它和真正的 KVM、VMware、云主机、物理机仍然不完全一样。
3. 如果后面你要进一步模拟更真实的生产基础设施，要把这一层差异考虑进去。

## 5. 第一步：确认 OrbStack 状态

执行位置：

1. 宿主机 macOS

执行命令：

```bash
orbctl status
orbctl list
```

命令作用：

1. 确认 OrbStack 是否运行。
2. 确认当前已有的 Linux machine。

预期结果：

1. `orbctl status` 输出 `Running`
2. `orbctl list` 能列出已有机器

本次实际看到的关键结果：

```text
$ orbctl status
Running
```

## 6. 第二步：创建两个节点

执行位置：

1. 宿主机 macOS

本次实际执行命令如下：

```bash
orbctl create --memory 8G --cpus 4 --disk 80G ubuntu:noble k8s-master-01
orbctl create --memory 8G --cpus 4 --disk 80G ubuntu:noble k8s-worker-01
```

命令说明：

1. `--memory 8G` 给每个节点分配 8 GB 内存。
2. `--cpus 4` 给每个节点分配 4 个 vCPU。
3. `--disk 80G` 给每个节点分配 80 GB 磁盘上限。
4. `ubuntu:noble` 表示 Ubuntu 24.04。

创建后检查命令：

```bash
orbctl list
orbctl info k8s-master-01
orbctl info k8s-worker-01
```

本次实际结果如下：

```text
k8s-master-01   running  ubuntu  noble  arm64  192.168.139.167
k8s-worker-01   running  ubuntu  noble  arm64  192.168.139.177
```

如果是在真正的 Linux 主机上：

1. 这里不会用 `orbctl create`。
2. 节点可能来自物理机、云主机、KVM、VMware、Proxmox 或其他平台。
3. 节点准备完成后，从“系统初始化”开始，后续步骤基本一致。

## 7. 第三步：确认节点可登录

执行位置：

1. 宿主机 macOS

检查命令：

```bash
orb -m k8s-master-01 whoami
orb -m k8s-master-01 hostnamectl
orb -m k8s-worker-01 whoami
orb -m k8s-worker-01 hostnamectl
```

本次实际观察到：

1. 默认用户是和宿主机同名的 `liuqianli`
2. 两个节点 hostname 已经分别是 `k8s-master-01` 和 `k8s-worker-01`

这一步的意义：

1. 确认节点已经可操作。
2. 确认后面不需要再手动改 hostname。

## 8. 第四步：做基础系统初始化

这一步在两个节点都要做。

执行位置：

1. `k8s-master-01`
2. `k8s-worker-01`

本次实际执行逻辑如下：

1. 更新软件包索引。
2. 安装基础工具。
3. 关闭 swap。
4. 写入内核模块配置。
5. 写入 sysctl 配置。
6. 配置 `/etc/hosts`

在每个节点执行的命令可以整理为：

```bash
sudo bash -lc '
set -euo pipefail

apt-get update
DEBIAN_FRONTEND=noninteractive apt-get install -y \
  apt-transport-https \
  ca-certificates \
  curl \
  gpg \
  software-properties-common \
  jq

swapoff -a || true
sed -ri "s@^([^#].*\sswap\s+.*)$@# \1@" /etc/fstab

cat >/etc/modules-load.d/k8s.conf <<EOF
overlay
br_netfilter
EOF

modprobe overlay
modprobe br_netfilter

cat >/etc/sysctl.d/99-kubernetes-cri.conf <<EOF
net.bridge.bridge-nf-call-iptables = 1
net.bridge.bridge-nf-call-ip6tables = 1
net.ipv4.ip_forward = 1
EOF

sysctl --system
'
```

然后写入主机名解析：

```bash
sudo bash -lc '
grep -q "192.168.139.167 k8s-master-01" /etc/hosts || echo "192.168.139.167 k8s-master-01" >> /etc/hosts
grep -q "192.168.139.177 k8s-worker-01" /etc/hosts || echo "192.168.139.177 k8s-worker-01" >> /etc/hosts
'
```

为什么要做这一步：

1. Kubernetes 依赖内核网络参数。
2. `br_netfilter` 和 `ip_forward` 是常见前置要求。
3. 关闭 swap 是 kubeadm 的标准要求。
4. 提前配好 `hosts` 更利于节点之间排查问题。

验证命令：

```bash
sysctl net.ipv4.ip_forward
sysctl net.bridge.bridge-nf-call-iptables
sysctl net.bridge.bridge-nf-call-ip6tables
```

预期结果：

```text
net.ipv4.ip_forward = 1
net.bridge.bridge-nf-call-iptables = 1
net.bridge.bridge-nf-call-ip6tables = 1
```

## 9. 第五步：安装 containerd

这一步在两个节点都要做。

执行命令：

```bash
sudo bash -lc '
set -euo pipefail

DEBIAN_FRONTEND=noninteractive apt-get install -y containerd

mkdir -p /etc/containerd
containerd config default >/etc/containerd/config.toml
sed -i "s/SystemdCgroup = false/SystemdCgroup = true/" /etc/containerd/config.toml

systemctl enable --now containerd
'
```

为什么要做这一步：

1. Kubernetes 当前推荐直接使用 `containerd`
2. `SystemdCgroup = true` 更符合 kubelet 与 systemd 的配合方式
3. `enable --now` 能让节点重启后自动恢复 containerd

验证命令：

```bash
containerd --version
systemctl is-enabled containerd
systemctl is-active containerd
```

本次实际结果：

```text
containerd github.com/containerd/containerd/v2 2.2.1
enabled
active
```

## 10. 第六步：安装 kubeadm、kubelet、kubectl

这一步在两个节点都要做。

执行命令：

```bash
sudo bash -lc '
set -euo pipefail

mkdir -p -m 755 /etc/apt/keyrings
curl -fsSL https://pkgs.k8s.io/core:/stable:/v1.33/deb/Release.key | gpg --dearmor -o /etc/apt/keyrings/kubernetes-apt-keyring.gpg
chmod 644 /etc/apt/keyrings/kubernetes-apt-keyring.gpg

echo "deb [signed-by=/etc/apt/keyrings/kubernetes-apt-keyring.gpg] https://pkgs.k8s.io/core:/stable:/v1.33/deb/ /" >/etc/apt/sources.list.d/kubernetes.list
chmod 644 /etc/apt/sources.list.d/kubernetes.list

apt-get update
DEBIAN_FRONTEND=noninteractive apt-get install -y kubelet kubeadm kubectl
apt-mark hold kubelet kubeadm kubectl
systemctl enable kubelet
'
```

为什么要做这一步：

1. `kubeadm` 负责初始化和加入集群。
2. `kubelet` 负责节点上的 Pod 生命周期。
3. `kubectl` 负责管理集群。
4. `apt-mark hold` 防止自动升级造成版本漂移。

验证命令：

```bash
kubeadm version -o short
systemctl is-enabled kubelet
systemctl is-active kubelet || true
```

本次实际结果：

```text
v1.33.13
enabled
inactive
```

这里 `kubelet` 在初始化前是 `inactive`，这不一定是问题。

原因是：

1. kubelet 启动后会等待合适的配置。
2. 在 `kubeadm init` 或 `kubeadm join` 之后才会进入正常工作状态。

## 11. 第七步：初始化 control-plane

这一步只在 `k8s-master-01` 上执行。

执行命令：

```bash
sudo kubeadm init \
  --apiserver-advertise-address=192.168.139.167 \
  --pod-network-cidr=192.168.0.0/16 \
  --cri-socket=unix:///run/containerd/containerd.sock \
  --node-name=k8s-master-01
```

命令说明：

1. `--apiserver-advertise-address` 指定 apiserver 对外通告地址。
2. `--pod-network-cidr` 要和后面 CNI 配置匹配。
3. `--cri-socket` 明确指定 containerd。
4. `--node-name` 固定节点名，避免后续歧义。

本次初始化成功后，关键输出如下：

```text
Your Kubernetes control-plane has initialized successfully!
```

同时会输出配置 `kubectl` 的命令：

```bash
mkdir -p $HOME/.kube
sudo cp -i /etc/kubernetes/admin.conf $HOME/.kube/config
sudo chown $(id -u):$(id -g) $HOME/.kube/config
```

还会输出本次 worker 加入命令：

```bash
kubeadm join 192.168.139.167:6443 --token b9pnot.2ar04lrtwf7dzsz2 \
  --discovery-token-ca-cert-hash sha256:d362554c8581866f02f0e6dd725c8628392e916e7b77ffc8c9d9dc84faae9b48
```

注意：

1. 这个 token 是有时效的。
2. 以后如果过期，需要重新生成 join 命令。

重新生成方式：

```bash
kubeadm token create --print-join-command
```

## 12. 第八步：配置 master 上的 kubectl

这一步只在 `k8s-master-01` 上执行。

执行命令：

```bash
mkdir -p $HOME/.kube
sudo cp /etc/kubernetes/admin.conf $HOME/.kube/config
sudo chown $(id -u):$(id -g) $HOME/.kube/config
```

验证命令：

```bash
kubectl get nodes -o wide
```

这个阶段通常会看到 master 还是 `NotReady`。

原因是：

1. 此时 CNI 还没有安装。
2. Pod 网络还不可用。

## 13. 第九步：安装 Calico

这一步只在 `k8s-master-01` 上执行。

本次实际执行命令：

```bash
kubectl apply -f https://raw.githubusercontent.com/projectcalico/calico/v3.30.4/manifests/calico.yaml
```

为什么选择 Calico：

1. 文档多。
2. 社区成熟。
3. 在生产环境中使用广泛。
4. 对学习网络插件机制比较友好。

## 14. 第十步：让 worker 加入集群

这一步只在 `k8s-worker-01` 上执行。

本次实际执行命令：

```bash
sudo kubeadm join 192.168.139.167:6443 \
  --token b9pnot.2ar04lrtwf7dzsz2 \
  --discovery-token-ca-cert-hash sha256:d362554c8581866f02f0e6dd725c8628392e916e7b77ffc8c9d9dc84faae9b48 \
  --cri-socket=unix:///run/containerd/containerd.sock \
  --node-name=k8s-worker-01
```

执行后回到 master 验证：

```bash
kubectl get nodes -o wide
kubectl get pods -A -o wide
```

## 15. 本次真实遇到的问题

### 15.1 问题一：Calico 镜像拉取失败

本次最关键的问题出现在 Calico 启动阶段。

故障现象：

1. 节点一直是 `NotReady`
2. `calico-node` 一直起不来
3. `coredns` 一直 `Pending`

本次实际报错如下：

```text
Failed to pull image "docker.io/calico/cni:v3.30.4": failed to pull and unpack image "docker.io/calico/cni:v3.30.4": failed to resolve image: failed to do request: Head "https://registry-1.docker.io/v2/calico/cni/manifests/v3.30.4": net/http: TLS handshake timeout
```

这说明：

1. 不是 Kubernetes 配置本身错了。
2. 是从 `docker.io` 拉取 Calico 镜像时超时了。

### 15.2 问题一的解决方案

本次解决方法不是换 CNI，而是绕过 `docker.io`，先从 `quay.io` 预拉镜像，再打回 Calico 清单默认期望的标签。

在 `k8s-master-01` 上执行：

```bash
sudo bash -lc '
set -euo pipefail

ctr -n k8s.io images pull quay.io/calico/node:v3.30.4
ctr -n k8s.io images tag quay.io/calico/node:v3.30.4 docker.io/calico/node:v3.30.4

ctr -n k8s.io images pull quay.io/calico/cni:v3.30.4
ctr -n k8s.io images tag quay.io/calico/cni:v3.30.4 docker.io/calico/cni:v3.30.4

ctr -n k8s.io images pull quay.io/calico/kube-controllers:v3.30.4
ctr -n k8s.io images tag quay.io/calico/kube-controllers:v3.30.4 docker.io/calico/kube-controllers:v3.30.4
'
```

在 `k8s-worker-01` 上执行：

```bash
sudo bash -lc '
set -euo pipefail

ctr -n k8s.io images pull quay.io/calico/node:v3.30.4
ctr -n k8s.io images tag quay.io/calico/node:v3.30.4 docker.io/calico/node:v3.30.4

ctr -n k8s.io images pull quay.io/calico/cni:v3.30.4
ctr -n k8s.io images tag quay.io/calico/cni:v3.30.4 docker.io/calico/cni:v3.30.4
'
```

然后重建 Calico Pod：

```bash
kubectl delete pod -n kube-system -l k8s-app=calico-node --force --grace-period=0
kubectl delete pod -n kube-system -l k8s-app=calico-kube-controllers --force --grace-period=0
```

之后再次检查：

```bash
kubectl get nodes
kubectl get pods -n kube-system
```

本次最终恢复成功。

## 16. 本次排查思路总结

如果你后面遇到节点 `NotReady`，建议按下面顺序排查：

1. `kubectl get nodes`
2. `kubectl get pods -A`
3. `kubectl get pods -n kube-system`
4. `kubectl describe pod -n kube-system <pod-name>`
5. `systemctl status containerd`
6. `systemctl status kubelet`
7. `journalctl -u kubelet -xe`

本次之所以能快速定位问题，是因为看了 `calico-node` 的 `Events`，而不是盲猜。

## 17. 如果换成真正的 Linux 主机，哪些地方会不同

从节点内部看，下面这些步骤几乎是一样的：

1. 关闭 swap
2. 配置内核参数
3. 安装 containerd
4. 安装 kubeadm、kubelet、kubectl
5. 执行 `kubeadm init`
6. 执行 `kubeadm join`
7. 安装 Calico

真正不同的主要在这些地方：

1. 节点创建方式不同。
2. 网络环境不同。
3. 存储环境不同。
4. 镜像可达性不同。
5. 开机恢复方式不同。

在真实 Linux 主机上：

1. 你不会用 `orbctl create`
2. 节点通常来自物理机、云主机或其他虚拟化平台
3. 你更可能面对固定网卡名、静态路由、安全组、防火墙、DNS 等问题

## 18. 在真正的 Linux 主机上常见踩坑

### 18.1 忘记关闭 swap

现象：

1. kubeadm preflight 检查失败

处理：

```bash
swapoff -a
sed -ri "s@^([^#].*\sswap\s+.*)$@# \1@" /etc/fstab
```

### 18.2 没有打开 `ip_forward`

现象：

1. CNI 启动后网络异常
2. Pod 间通信不正常

处理：

```bash
sysctl net.ipv4.ip_forward
```

如果结果不是 `1`，就要修复 sysctl 配置。

### 18.3 containerd 的 cgroup 驱动不匹配

现象：

1. kubelet 异常
2. Pod 创建失败

处理重点：

1. 检查 `/etc/containerd/config.toml`
2. 确认 `SystemdCgroup = true`

### 18.4 kubelet 没有设置为开机自启

现象：

1. 节点重启后控制面起不来
2. kubeadm 初始化过的节点恢复异常

处理：

```bash
sudo systemctl enable kubelet
sudo systemctl enable containerd
```

### 18.5 Pod 网段和现有网络冲突

现象：

1. Pod 网络异常
2. Service 路由异常
3. 访问路径莫名其妙不通

处理建议：

1. 在初始化集群前就规划好 Pod CIDR 和 Service CIDR
2. 避免和现有办公网、宿主机网段冲突

### 18.6 镜像仓库访问不稳定

现象：

1. 节点一直 `NotReady`
2. kube-system 中网络插件起不来
3. 报 `ErrImagePull`、`ImagePullBackOff`

处理建议：

1. 优先看 `kubectl describe pod`
2. 不要一上来怀疑 kubeadm
3. 必要时提前 `ctr images pull`
4. 必要时替换镜像源或预拉镜像

### 18.7 防火墙或安全组限制

在真实 Linux 环境里尤其常见。

重点端口包括：

1. `6443`
2. `10250`
3. CNI 所需端口
4. 节点间 overlay 网络所需端口

如果是在云上，还要检查：

1. 安全组
2. VPC 路由
3. Network ACL

## 19. 本次安装完成后的常用验证命令

查看节点：

```bash
kubectl get nodes -o wide
```

查看所有 Pod：

```bash
kubectl get pods -A -o wide
```

查看系统组件：

```bash
kubectl get pods -n kube-system
```

查看节点详细信息：

```bash
kubectl describe node k8s-master-01
kubectl describe node k8s-worker-01
```

查看 kubelet 状态：

```bash
sudo systemctl status kubelet --no-pager
```

查看 containerd 状态：

```bash
sudo systemctl status containerd --no-pager
```

## 20. 本次安装过程的关键结论

这次安装有几个非常重要的结论：

1. `Mac + OrbStack + kubeadm + containerd + Calico` 这条路线是可行的。
2. `ARM64` 环境本身没有阻止这次安装成功。
3. 真正导致卡住的主要问题不是 kubeadm，而是镜像拉取。
4. 安装 K8s 时一定要保留排查证据，尤其是 `kubectl describe pod` 的 `Events`
5. 节点能否真正 `Ready`，很大程度上取决于 CNI 是否正常启动。

## 21. 下一步建议

在这篇实操手册基础上，后续建议继续补下面几个章节：

1. 安装 `ingress-nginx`
2. 安装 `MetalLB`
3. 部署一个示例服务
4. 增加新 worker 节点
5. 写一套宿主机关机后快速恢复集群的脚本

## 22. 关联笔记

这篇笔记对应的方案设计笔记是：

`[[mac使用orbstack搭建k8s集群]]`
