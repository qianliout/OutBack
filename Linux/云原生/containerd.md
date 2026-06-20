# containerd 核心原理与实战笔记

> **定位**：CNCF 毕业项目，行业标准的容器运行时。Docker 在 2017 年将其核心引擎捐献给了 CNCF，此后 containerd 独立发展，成为 K8s 默认的容器运行时。

---

## 目录

- [一、containerd 在容器生态中的位置](#一containerd-在容器生态中的位置)
- [二、containerd vs Docker：逐项对比](#二containerd-vs-docker逐项对比)
- [三、核心架构](#三核心架构)
- [四、containerd 的 namespace 概念](#四containerd-的-namespace-概念)
- [五、三种客户端工具对比](#五三种客户端工具对比)
- [六、常用命令速查（ctr）](#六常用命令速查ctr)
- [七、常用命令速查（nerdctl）](#七常用命令速查nerdctl)
- [八、构建镜像](#八构建镜像)
- [九、运行容器全流程解析](#九运行容器全流程解析)
- [十、数据目录结构](#十数据目录结构)
- [十一、K8s 中的 containerd](#十一k8s-中的-containerd)
- [十二、故障排查](#十二故障排查)
- [十三、小结：containerd 的核心价值](#十三小结containerd-的核心价值)

---

## 一、containerd 在容器生态中的位置

### 1.1 完整链路

```
用户界面层       docker CLI          nerdctl / ctr
                   │                    │
镜像构建层       dockerd  ────────── BuildKit
                   │                    │
容器运行时         └─ unix socket ──── containerd
                                           │
shim 层                             containerd-shim-runc-v2
                                           │
OCI 运行时                                runc
                                           │
内核                              Linux Namespace + Cgroups
```

**关键理解**：Docker 去掉 `dockerd` 后，就是 containerd + runc。`dockerd` 提供的额外功能是：镜像构建、`docker compose`、Docker Hub 集成、用户友好的 CLI。

### 1.2 K8s 路径

```
kubelet  ── CRI gRPC ── containerd (内置 CRI plugin) ── shim ── runc
```

K8s 不需要 `dockerd`——kubelet 通过 CRI 接口直接调用 containerd。从 K8s 1.24 起，`dockershim` 被移除，containerd 成为事实标准。

### 1.3 一句话定义

> **containerd 是管理容器完整生命周期的 daemon**：拉取镜像、管理镜像层（snapshot）、创建/启动/停止容器、管理容器进程的 stdin/stdout。它**不负责**构建镜像，也不提供用户可以直接用的 CLI。

---

## 二、containerd vs Docker：逐项对比

| 维度 | Docker（docker-ce） | containerd |
|------|--------------------|------------|
| **组成部分** | dockerd + containerd + runc | containerd + runc |
| **架构复杂度** | 高（一大坨功能在一个 daemon 里） | 低（职责单一） |
| **镜像构建** | ✅ `docker build` | ❌ 需要外部工具（nerdctl build / BuildKit） |
| **镜像仓库集成** | ✅ Docker Hub 内置 | ❌ 需要自己配置 |
| **CLI 工具** | `docker` | `ctr`（调试用）、`nerdctl`（Docker 兼容） |
| **Compose** | ✅ `docker compose` | ❌ containerd 本身不支持，nerdctl 支持 |
| **OCI 兼容** | 部分（历史包袱） | 完全 OCI 标准 |
| **systemd cgroup** | 默认 cgroupfs（需显式配置） | 默认 systemd cgroup（K8s 友好） |
| **K8s 支持** | 需要 dockershim（1.24 已移除） | **原生 CRI 插件**，K8s 默认推荐 |
| **资源占用** | 较重（docker daemon 约 200MB+） | 轻量（daemon 约 50-80MB） |
| **许可证** | Docker CE 曾有企业限制 | Apache 2.0 |

**简单判断什么时候用哪个**：

| 场景 | 推荐 |
|------|------|
| 个人开发、`docker compose up` 一把梭 | Docker |
| K8s 节点 | containerd |
| CI/CD 中跑容器、不需要 Docker Hub | containerd + nerdctl |
| 边缘设备、资源受限 | containerd |

---

## 三、核心架构

### 3.1 containerd 内部组件

```
┌──────────────────────────────────────────────────┐
│                  containerd daemon                │
│                                                   │
│  ┌──────────┐  ┌────────────┐  ┌──────────────┐  │
│  │  CRI     │  │  Services  │  │  Event/Task  │  │
│  │  Plugin  │  │  (gRPC)    │  │  Manager     │  │
│  └──────────┘  └────────────┘  └──────────────┘  │
│         │             │               │           │
│         ▼             ▼               ▼           │
│  ┌───────────────────────────────────────────┐   │
│  │           Snapshotter (overlayfs)          │   │
│  └───────────────────────────────────────────┘   │
│  ┌───────────────────────────────────────────┐   │
│  │           Content Store (blobs)            │   │
│  └───────────────────────────────────────────┘   │
│  ┌───────────────────────────────────────────┐   │
│  │           Metadata (BoltDB)                │   │
│  └───────────────────────────────────────────┘   │
└──────────────────────┬───────────────────────────┘
                       │
              ┌────────▼────────┐
              │  shim process   │  ← 每个容器一个 shim
              └────────┬────────┘
                       │
              ┌────────▼────────┐
              │      runc       │  ← 创建容器后退出
              └────────┬────────┘
                       │
              ┌────────▼────────┐
              │  容器进程 (PID)  │
              └─────────────────┘
```

### 3.2 各组件的职责

| 组件 | 职责 | 生命周期 |
|------|------|----------|
| **containerd daemon** | gRPC 服务端，协调整体流程，管理镜像、快照、容器元数据 | 随系统启动，常驻 |
| **containerd-shim-runc-v2** | 每个容器一个 shim 进程。作用：① 让 runc 可以安全退出而不影响容器进程；② 接管容器的 stdin/stdout/stderr 流；③ 向 containerd 报告容器退出状态 | 随容器创建而创建，随容器退出而退出 |
| **runc** | OCI 运行时，根据 `config.json` 和 `rootfs` 创建容器进程。工作完成后退出 | 临时进程，创建完容器就退出 |
| **Snapshotter** | 管理文件系统层的快照（准备 rootfs），默认用 overlayfs | 常驻（containerd 的一部分） |
| **Content Store** | 存储镜像层的原始 blob 数据（压缩的 tar.gz 文件），按 SHA256 寻址 | 常驻 |

### 3.3 Shim 为什么重要？

这是 containerd 设计中最精妙的部分。考虑一个问题：

> runc 创建了容器进程，如果 runc 自己退出，容器进程怎么办？

答案是 **shim**。runc 创建完容器后可以立即退出，容器的父进程变成 **shim**（而不是 containerd 自身）。这样的好处：

1. **containerd 重启不影响容器**：containerd daemon 升级或重启时，shim 和容器进程不受影响。containerd 重启后可以重新连接已有的 shim。
2. **stdin/stdout 流不中断**：shim 作为容器和 containerd 之间的"管道"，即使 containerd 暂时不可用，日志流也不会丢。
3. **一个容器一个 shim**：隔离故障域。一个 shim 挂掉只影响一个容器。

```
无 shim 时（脆弱）：
containerd ── runc ── 容器进程 (runc 退出后，容器变孤儿)

有 shim 时（健壮）：
containerd ── shim ── runc ── 容器进程
                │               │
                │      runc 创建完容器后退出
                │               │
                └── 容器父进程就是 shim ──┘
                     containerd 重启 → shim 继续持有容器
```

---

## 四、containerd 的 namespace 概念

containerd 有自己的 **namespace** 概念——这不是 Linux 内核的 Namespace，而是 **containerd 内部用来隔离镜像和容器资源的逻辑分区**。

### 4.1 为什么需要 containerd namespace？

同一台宿主机上，不同的系统可能共用 containerd。例如：

- K8s 通过 CRI 使用 containerd，不希望看到 `ctr` 手动创建的容器
- 用户手动 `ctr` 拉取调试镜像，不影响 K8s 的资源列表

containerd namespace 就是这个隔离边界。

### 4.2 常用 namespace

| Namespace | 使用者 | 内容 |
|-----------|--------|------|
| `default` | `ctr` / `nerdctl` 默认 | 用户手动拉取的镜像和容器 |
| `k8s.io` | K8s CRI plugin | K8s 管理的所有 Pod 镜像和容器 |
| `moby` | Docker Engine（当 Docker 使用 containerd 时） | Docker 管理的资源 |

### 4.3 操作示例

```bash
# ctr 默认操作 default namespace
ctr images ls                    # 列出 default namespace 的镜像

# 指定 namespace（K8s 场景最常用）
ctr -n k8s.io images ls          # 列出 K8s 使用的镜像
ctr -n k8s.io containers ls      # 列出 K8s 的容器
ctr -n k8s.io snapshots ls       # 列出 K8s 的快照

# nerdctl 也有 namespace 支持
nerdctl -n k8s.io images ls
```

**关键区别**：一台 K8s 节点上，`ctr images ls` 返回空（没在 `default` 里放过东西），但 `ctr -n k8s.io images ls` 会列出所有 Pod 使用的镜像。很多人第一次接触 containerd 时被这个坑过。

---

## 五、三种客户端工具对比

### 5.1 工具定位

| 工具 | 全称 | 定位 | 语法 | 推荐场景 |
|------|------|------|------|----------|
| **ctr** | containerd CLI | containerd 自带调试工具，不对用户友好 | 自定义 | 调试 containerd 自身、操作快照和 blob |
| **crictl** | CRI CLI | K8s 社区维护，只操作 CRI 接口暴露的容器 | 类似 docker | K8s 节点上调试 Pod（看不到手动创建的容器） |
| **nerdctl** | nerdctl | containerd 项目的 Docker-compatible CLI | 完全兼容 docker | **日常使用首选**，支持 Compose、Build、Run |

### 5.2 命令范围对比

| 操作 | ctr | crictl | nerdctl |
|------|:---:|:---:|:---:|
| 拉取镜像 | `ctr image pull` | `crictl pull` | `nerdctl pull` |
| 列出镜像 | `ctr images ls` | `crictl images` | `nerdctl images` |
| 运行容器 | `ctr run` | `crictl run` | `nerdctl run` |
| 构建镜像 | ❌ | ❌ | ✅ `nerdctl build` |
| Compose | ❌ | ❌ | ✅ `nerdctl compose up` |
| 管理快照 | ✅ `ctr snapshots ls` | ❌ | ❌ |
| 管理 blobs | ✅ `ctr content ls` | ❌ | ❌ |
| 查看 K8s Pod | 需要 `-n k8s.io` | ✅ `crictl pods` | ❌ |
| 命名空间切换 | `-n` | ❌（只操作 K8s 的） | `--namespace` |

### 5.3 工具选择建议

```
如果你是 K8s 节点管理员 → crictl（排查 Pod 问题）
如果你是开发者想用 containerd 替代 Docker → nerdctl
如果你在调试 containerd 本身 → ctr
```

---

## 六、常用命令速查（ctr）

### 6.1 镜像管理

```bash
# 拉取镜像
ctr image pull docker.io/library/nginx:alpine

# 列出镜像
ctr images ls

# 查看镜像详情
ctr images info docker.io/library/nginx:alpine

# 导出镜像为 tar
ctr images export nginx-alpine.tar docker.io/library/nginx:alpine

# 从 tar 导入镜像
ctr images import nginx-alpine.tar

# 打标签
ctr images tag docker.io/library/nginx:alpine nginx:local

# 删除镜像
ctr images rm docker.io/library/nginx:alpine
```

> **注意**：`ctr image pull` 需要写完整地址（`docker.io/library/nginx:alpine`），不能简写 `nginx:alpine`。这是 ctr 和 docker 的一个重要区别。

### 6.2 容器运行

```bash
# 方式一：一键运行（pull + create + start）
ctr run --rm -t docker.io/library/nginx:alpine mynginx

# 方式二：分步操作（理解容器生命周期）
ctr image pull docker.io/library/nginx:alpine
ctr container create docker.io/library/nginx:alpine mynginx
ctr task start -d mynginx                  # -d 表示后台运行

# 列出容器
ctr containers ls

# 列出运行中的任务（task）
ctr task ls

# 在容器中执行命令
ctr task exec -t --exec-id shell1 mynginx /bin/sh

# 暂停 / 恢复容器
ctr task pause mynginx
ctr task resume mynginx

# 杀掉容器
ctr task kill mynginx

# 删除容器
ctr container rm mynginx
```

### 6.3 快照与内容管理

```bash
# 列出快照树（查看镜像分层）
ctr snapshots ls

# 查看快照的父子关系
ctr snapshots tree

# 查看快照详情
ctr snapshots info <snapshot-id>

# 列出 raw blob 内容
ctr content ls
```

### 6.4 命名空间操作

```bash
# 列出所有 containerd namespace
ctr namespaces ls

# 创建 namespace
ctr namespaces create mytest

# 在指定 namespace 操作
ctr -n mytest images pull docker.io/library/alpine:latest
ctr -n mytest images ls

# 删除 namespace（需先清理里面的资源）
ctr namespaces rm mytest
```

---

## 七、常用命令速查（nerdctl）

`nerdctl` 的语法刻意与 `docker` 保持一致，大多数命令可以直接替换。如果你会 docker，基本不需要重新学习。

### 7.1 安装

```bash
# 从 GitHub Releases 下载（推荐 latest）
wget https://github.com/containerd/nerdctl/releases/download/v2.0.0/nerdctl-2.0.0-linux-amd64.tar.gz
sudo tar xzf nerdctl-2.0.0-linux-amd64.tar.gz -C /usr/local/bin/
```

### 7.2 镜像管理

```bash
# 拉取镜像（和 docker 完全一样的语法）
nerdctl pull nginx:alpine
nerdctl pull registry.k8s.io/pause:3.9

# 列出镜像
nerdctl images

# 推送镜像
nerdctl tag nginx:alpine myregistry.com/nginx:alpine
nerdctl push myregistry.com/nginx:alpine

# 保存与加载
nerdctl save -o nginx.tar nginx:alpine
nerdctl load -i nginx.tar

# 删除镜像
nerdctl rmi nginx:alpine
```

### 7.3 容器运行

```bash
# 运行容器（后台运行，端口映射）
nerdctl run -d -p 8080:80 --name mynginx nginx:alpine

# 查看运行中的容器
nerdctl ps

# 查看所有容器（含已停止）
nerdctl ps -a

# 进入容器
nerdctl exec -it mynginx /bin/sh

# 查看容器日志
nerdctl logs -f mynginx

# 停止 / 删除容器
nerdctl stop mynginx
nerdctl rm mynginx

# 查看容器资源使用
nerdctl stats
```

### 7.4 Compose

```bash
# 启动 Compose 项目
nerdctl compose up -d

# 查看 Compose 状态
nerdctl compose ps

# 停止 Compose 项目
nerdctl compose down

# 查看 Compose 日志
nerdctl compose logs -f
```

### 7.5 其他常用命令

```bash
# 登录镜像仓库
nerdctl login registry.mycompany.com

# 查看磁盘使用
nerdctl system df

# 清理未使用的资源
nerdctl system prune -a

# 查看 containerd 版本信息
nerdctl version

# 在指定 containerd namespace 操作
nerdctl --namespace k8s.io images ls
```

---

## 八、构建镜像

### 8.1 containerd 本身不能构建镜像

这是一个经常被误解的点：containerd 只管**运行**容器，不管**构建**镜像。构建镜像需要 BuildKit 或其他工具。

### 8.2 方案一：nerdctl build（推荐）

```bash
# 和 docker build 完全一样
nerdctl build -t myapp:v1 .

# 指定 Dockerfile
nerdctl build -f Dockerfile.prod -t myapp:v1 .

# 构建并推送到仓库
nerdctl build -t myregistry.com/myapp:v1 .
nerdctl push myregistry.com/myapp:v1
```

底层原理：`nerdctl build` 内置了 BuildKit 支持。首次运行会下载 `buildkitd` 容器镜像，后续构建在 BuildKit 守护进程中执行。

### 8.3 方案二：docker build + import

如果环境中已有 Docker，可以用 Docker 构建然后导入 containerd：

```bash
# 用 Docker 构建
docker build -t myapp:v1 .

# 导出为 tar
docker save myapp:v1 -o myapp.tar

# 导入 containerd
ctr images import myapp.tar
# 或者
nerdctl load -i myapp.tar
```

### 8.4 方案三：buildctl（BuildKit 原生客户端）

```bash
# 启动 buildkitd
sudo systemctl start buildkit

# 用 buildctl 构建
buildctl build \
  --frontend=dockerfile.v0 \
  --local context=. \
  --local dockerfile=. \
  --output type=image,name=myapp:v1,push=false
```

这是最底层的方式，一般不需要直接用——`nerdctl build` 封装了这些。

### 8.5 构建方案对比

| 方案 | 复杂度 | 适用场景 |
|------|:---:|------|
| `nerdctl build` | 低 | **首选**，和 `docker build` 一样简单 |
| `docker build` + `ctr import` | 中 | 迁移阶段，已有 Docker 工作流 |
| `buildctl` | 高 | 需要 BuildKit 高级特性（cache export、multi-arch） |

---

## 九、运行容器全流程解析

### 9.1 ctr 三步流程：体现容器生命周期

containerd 把"运行容器"拆成了三个独立步骤，这和 Docker 的 `docker run` 一键完成不同。理解这三个步骤有助于理解容器运行时的工作原理：

```
ctr image pull    →  拉取镜像，解压到 content store，创建 committed snapshots
ctr container create  →  基于镜像快照创建容器的 rootfs（active snapshot）+ 生成 OCI spec
ctr task start    →  调用 shim → runc 真正创建容器进程
```

### 9.2 详细步骤拆解

```bash
# 第 0 步：拉取镜像
ctr image pull docker.io/library/alpine:latest

# 查看镜像拉取后创建的快照（committed snapshot）
ctr snapshots ls | grep alpine
# 输出示例：
# sha256:abc123...  COMMITTED  (这就是镜像层的 rootfs)

# ──────────────────────────────────────

# 第 1 步：创建容器（但不运行）
ctr container create docker.io/library/alpine:latest myalpine

# 此时发生了什么？
# ① containerd 为 myalpine 创建一个 ACTIVE snapshot（可写层），
#    其 parent 指向镜像层的 COMMITTED snapshot
# ② 生成 OCI config.json（进程参数、环境变量、挂载点等）
# ③ 容器元数据写入 BoltDB
# ④ 此时容器进程还没有被创建！

# 可以看到容器已注册
ctr containers ls | grep myalpine

# ──────────────────────────────────────

# 第 2 步：真正启动容器进程
ctr task start -d myalpine

# 此时发生了什么？
# ① containerd fork 出 containerd-shim-runc-v2 进程
# ② shim fork 出 runc
# ③ runc 根据 config.json 创建 Namespace + Cgroups + rootfs pivot_root
# ④ runc 启动容器进程（alpine 的 /bin/sh 或 ENTRYPOINT）
# ⑤ runc 退出
# ⑥ shim 接管容器进程的 stdin/stdout/stderr，并向 containerd 报告状态

ctr task ls | grep myalpine
# 输出示例：
# TASK      PID      STATUS
# myalpine  12345    RUNNING
```

### 9.3 snapshot → container → task 三层模型

这是 containerd 最核心的设计模型：

```
Snapshot（文件系统层）
  │
  │  container create 基于 snapshot 创建
  ▼
Container（静态定义：rootfs + OCI spec + 元数据）
  │
  │  task start 将静态定义"激活"为运行中的进程
  ▼
Task（动态实体：运行中的进程 + stdin/stdout 流）
```

| 概念 | 对应什么 | 状态 | 可以做什么 |
|------|---------|------|-----------|
| **Snapshot** | 文件系统层（COMMITTED = 只读镜像层，ACTIVE = 容器可写层） | 静态 | `ctr snapshots ls`，快照挂载 |
| **Container** | 容器的"蓝图"（用什么镜像、挂什么 volume、什么环境变量） | 静态 | `ctr containers ls`，创建时生成 OCI spec |
| **Task** | 容器的"运行时"（实际的进程、PID、IO 流） | 动态 | `ctr task ls`，exec、kill、pause、resume |

**为什么这么设计？**

- **解耦文件系统和进程**：同一个 container 定义可以被多次 task start（类似 `docker start` 重复启动已停止的容器）。
- **Task 退出 ≠ Container 消失**：容器进程 crash 了，container 定义还在，可以重新 `task start`。
- **K8s 的优雅对应**：Pod 重启容器时不需要重建整个文件系统，只需要基于现有 container 定义创建新的 task。

---

## 十、数据目录结构

> 详细的存储模型分析见 [Fs.md  Part 3](./Fs.md)。这里给出快速目录速览。

### 10.1 一级目录

```bash
/var/lib/containerd/
├── io.containerd.content.v1.content/
│   └── blobs/                     # 镜像层的原始数据（tar.gz），按 SHA256 命名
│       └── sha256/
├── io.containerd.snapshotter.v1.overlayfs/
│   └── snapshots/                 # 解压后的文件系统快照
│       ├── 1/fs/                  # committed snapshot：镜像只读层
│       ├── 2/fs/                  # active snapshot：容器可写层
│       └── ...
├── io.containerd.metadata.v1.bolt/
│   └── meta.db                    # BoltDB 元数据库——镜像/容器/快照之间的关联
└── io.containerd.runtime.v2.task/
    └── default/                   # 每个容器一个目录，存放 shim 的 socket 和状态
        └── <container-id>/
```

### 10.2 目录对应的 containerd 概念

| 目录 | 对应概念 | 对应 `ctr` 子命令 |
|------|----------|-------------------|
| `content/blobs/sha256/` | Content Store（原始镜像数据） | `ctr content ls` |
| `snapshotter/overlayfs/snapshots/` | Snapshot Store（解压后的文件系统层） | `ctr snapshots ls` |
| `metadata/bolt/meta.db` | 元数据库（镜像→blob→快照的映射） | （无直接命令，通过其他命令间接操作） |
| `runtime.v2.task/` | Task 运行时（shim 通信 socket） | `ctr task ls` |

### 10.3 磁盘空间排查

```bash
# 查看 containerd 占用的磁盘空间
du -sh /var/lib/containerd/

# 查看各部分分别占用多少
du -sh /var/lib/containerd/io.containerd.content.v1.content/
du -sh /var/lib/containerd/io.containerd.snapshotter.v1.overlayfs/

# 使用 nerdctl 清理
nerdctl system df                   # 查看各资源占用
nerdctl system prune -a             # 清理未使用的资源
```

---

## 十一、K8s 中的 containerd

### 11.1 CRI 插件机制

containerd 内置了 **CRI plugin**（`io.containerd.grpc.v1.cri`），使其能够作为 K8s 的容器运行时。它的工作原理是：

```
kubelet
  │
  │  gRPC (CRI 协议: RunPodSandbox / CreateContainer / StartContainer ...)
  ▼
containerd CRI plugin
  │
  │  将 CRI 概念翻译成 containerd 内部概念:
  │    PodSandbox  →  containerd container (Pause)
  │    Container   →  containerd container (业务容器)
  │    Image       →  containerd image (存在 k8s.io namespace)
  ▼
containerd core (snapshotter, content, task manager)
```

### 11.2 CRI 与 containerd 概念的映射

| CRI 概念 | containerd 内部概念 | containerd namespace |
|----------|--------------------|---------------------|
| `PodSandbox` | Container (Pause 容器) | `k8s.io` |
| `Container` | Container (业务容器) | `k8s.io` |
| `Image` | Image | `k8s.io` |
| `RunPodSandbox` | 创建 Pause 容器 + CNI 配网 | — |
| `CreateContainer` | `ctr container create` | — |
| `StartContainer` | `ctr task start` | — |

### 11.3 在 K8s 节点上用 crictl 调试

```bash
# crictl 默认连接到 containerd 的 CRI socket
# 配置文件: /etc/crictl.yaml
#   runtime-endpoint: unix:///run/containerd/containerd.sock

# 查看 Pod
crictl pods

# 查看容器
crictl ps
crictl ps -a                     # 含已停止的

# 查看镜像
crictl images

# 拉取镜像
crictl pull nginx:alpine

# 查看容器日志
crictl logs <container-id>

# 进入容器（需要先找到 container-id）
crictl exec -it <container-id> /bin/sh

# 查看 Pod 详情
crictl inspectp <pod-id>
```

### 11.4 containerd 的 CRI 配置

```bash
# 查看 containerd 的默认配置
containerd config default > /etc/containerd/config.toml

# K8s 关键配置项
# [plugins."io.containerd.grpc.v1.cri"]
#   sandbox_image = "registry.k8s.io/pause:3.9"    # Pause 镜像
#   [plugins."io.containerd.grpc.v1.cri".containerd.runtimes.runc]
#     runtime_type = "io.containerd.runc.v2"
#     [plugins."io.containerd.grpc.v1.cri".containerd.runtimes.runc.options]
#       SystemdCgroup = true                         # K8s 要求用 systemd cgroup
```

---

## 十二、故障排查

### 12.1 查看 containerd 状态

```bash
# 检查 containerd 是否运行
sudo systemctl status containerd

# 查看 containerd 日志
sudo journalctl -u containerd -f     # -f 实时跟踪
sudo journalctl -u containerd -n 100 # 最近 100 行
sudo journalctl -u containerd --since "10 minutes ago"

# 查看 containerd 版本
containerd --version
ctr version
```

### 12.2 常见问题与排查

#### 问题 1：镜像拉取失败

```bash
# 查看错误日志
journalctl -u containerd | grep -i "pull\|error\|failed"

# 手动测试拉取
ctr image pull docker.io/library/alpine:latest

# 常见原因
# - 镜像仓库不可达（网络/防火墙/DNS）
# - 镜像仓库需要认证（需要配置 registry auth）
# - 磁盘空间不足
```

**配置私有仓库认证**：

```bash
# nerdctl 方式（和 docker login 一样）
nerdctl login myregistry.company.com

# 手动配置 containerd（K8s 场景）
# 在 /etc/containerd/config.toml 中：
# [plugins."io.containerd.grpc.v1.cri".registry.configs."myregistry.com".auth]
#   username = "xxx"
#   password = "xxx"
```

#### 问题 2：容器无法启动

```bash
# 查看失败容器
ctr containers ls
ctr task ls

# 查看容器详情
ctr containers info <container-id>

# 查看 shim 是否存活
ps aux | grep shim

# 手动启动容器测试
ctr run --rm -t docker.io/library/alpine:latest debug-container
```

#### 问题 3：快照耗尽

```bash
# 查看快照占用
ctr snapshots ls | wc -l

# 清理未使用的快照
# nerdctl 方式
nerdctl system prune -a

# ctr 方式：删除不再使用的镜像和容器
ctr images rm <unused-image>
ctr container rm <stopped-container>

# 检查磁盘使用
df -h /var/lib/containerd
```

#### 问题 4：shim 进程僵死

```bash
# 查找所有 shim 进程
ps aux | grep containerd-shim

# 如果一个容器已经不存在但 shim 还在，手动清理
# 1. 找到僵死 shim 的 PID 和对应的容器 ID
ps aux | grep "containerd-shim" | grep -v grep

# 2. 检查该容器是否在 containerd 中注册
ctr task ls

# 3. 如果容器已删除但 shim 残留，kill 它
# （先确认容器确实已不存在）
sudo kill <shim-pid>
```

### 12.3 开启 debug 日志

```bash
# 编辑 /etc/containerd/config.toml
# [debug]
#   level = "debug"

# 或者通过环境变量启动
sudo CONTAINERD_LOG_LEVEL=debug containerd

# 重启后观察详细日志
sudo systemctl restart containerd
sudo journalctl -u containerd -f
```

---

## 十三、小结：containerd 的核心价值

### 13.1 和 Docker 的关系

```
Docker (2015):  dockerd + containerd + runc = 单体大而全
                   │
                   │  2017: Docker 将 containerd 捐给 CNCF
                   ▼
containerd (2024): containerd + runc = 专注运行时, 轻量标准
```

containerd 是 Docker 把"运行时"这一层拆出来独立发展的产物。Docker 自己现在也用的是 containerd。

### 13.2 为什么 K8s 选择 containerd？

| 原因 | 说明 |
|------|------|
| **原生 CRI** | 不需要 `dockershim` 这层适配，直接 gRPC 调用 |
| **systemd cgroup 默认支持** | Docker 默认 cgroupfs，与 K8s 的 systemd cgroup 驱动有冲突风险 |
| **资源占用低** | containerd daemon 比 dockerd 轻得多（50MB vs 200MB+） |
| **重启不影响容器** | shim 架构保证 containerd 升级/重启时容器不中断 |
| **Apache 2.0 许可证** | 无商业限制 |
| **更清晰的数据模型** | snapshot → container → task 三层模型，比 Docker 的耦合设计更健壮 |

### 13.3 工具速查矩阵

| 你想干什么 | 用什么命令 |
|-----------|-----------|
| 日常开发，替代 docker | `nerdctl run/build/compose/…` |
| K8s 节点上排查 Pod | `crictl pods/ps/logs/exec` |
| 调试 containerd 内部状态 | `ctr snapshots/content/images ls` |
| 查看 containerd 日志 | `journalctl -u containerd` |
| 清理磁盘空间 | `nerdctl system prune -a` |
| 查看配置 | `containerd config dump` |
