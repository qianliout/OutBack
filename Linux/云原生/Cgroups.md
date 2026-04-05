# Linux Namespaces 与 Cgroups 深度学习笔记

---

## 一、 核心概念入门 (Introduction)

### 1. 为什么需要 Namespaces 和 Cgroups？

在现代软件架构中，我们追求更高的资源利用率、更快的应用部署速度和更可靠的环境一致性。这推动了从庞大的单体应用，到笨重的虚拟机，再到轻量级容器的技术演进。

容器技术之所以能够实现这些目标，其核心依赖于Linux内核提供的两大基石：

- **Namespaces (命名空间)**：**实现资源视图的隔离 (Isolation)**。
  - **作用**：让运行在Namespace中的进程感觉自己拥有一个独立的系统环境。例如，一个进程在自己的PID Namespace中看到的进程ID（PID）是从1开始的，它无法看到宿主机上或其他Namespace中的真实进程。同样，它可以拥有独立的网络配置（IP地址、路由表）、主机名、挂载点等。
  - **类比**：想象一栋大楼（宿主机），每个Namespace就像是其中的一个房间。房间里的人（进程）看不到其他房间的情况，他们有自己的房间号（PID）、自己的门牌（主机名）、自己的内部网络（网络设备）。

- **Cgroups (Control Groups, 控制组)**：**实现对进程组使用资源的限制 (Limitation)、审计和控制**。
  - **作用**：解决了资源滥用的问题。即使进程被隔离，如果某个进程无限制地消耗CPU或内存，依然会影响宿主机上其他进程的稳定运行。Cgroups就是为此而生，它可以将一组进程聚合起来，并统一对它们所使用的CPU、内存、I/O等资源进行配额和限制。
  - **类比**：继续上面的例子，Cgroups就是这栋大楼的中央空调和电表系统。物业可以为每个房间（Cgroup）设定用电额度（CPU/内存限制）、水流速度（I/O限制），并记录用量（资源审计）。

**协同工作**：Namespaces和Cgroups联手构建了容器的根基。**Namespaces负责“画地为牢”，构建出一个看似独立的王国；而Cgroups则负责为这个王国设定“资源预算”，确保它不会过度消耗，影响到邻国。**

### 2. 与虚拟机 (VM) 的对比

| 特性 | 虚拟机 (Virtual Machine) | 容器 (Container) |
| :--- | :--- | :--- |
| **隔离级别** | **硬件级隔离**，通过Hypervisor虚拟化一整套硬件（CPU, 内存, 磁盘） | **进程级隔离**，共享宿主机内核，隔离进程的视图和资源 |
| **内核** | 每个VM都有自己独立的客户机操作系统(Guest OS)和内核 | 所有容器**共享宿主机内核** |
| **资源开销** | **高**。需要为每个VM分配GB级的内存和磁盘空间，存在Hypervisor的开销 | **低**。容器只是一个特殊的进程，开销极小，通常为MB级 |
| **启动速度** | **慢** (分钟级)。需要启动一个完整的操作系统 | **快** (秒级甚至毫秒级)。本质上是启动一个进程 |
| **性能损耗** | **较大**。存在硬件虚拟化和指令翻译的开销 | **极小**。接近原生进程性能，因为没有额外的虚拟化层 |
| **部署密度** | **低**。一台物理机上通常只能运行几个到十几个VM | **高**。一台物理机上可以轻松运行成百上千个容器 |

---

## 二、 深入理解 Linux Namespaces

### 1. Namespace API 概览

Linux主要通过三个系统调用和/proc文件系统来操作Namespace：

- **`clone(fn, child_stack, flags, arg)`**: 创建一个新进程。通过在`flags`参数中指定`CLONE_NEW*`系列标志位，可以在创建子进程的同时，为它创建并加入新的Namespace。这是创建容器（新进程+新Namespace）最主要的方式。
  - 例如: `clone(..., CLONE_NEWPID | CLONE_NEWNET, ...)` 会创建一个新进程，该进程位于新的PID和Network Namespace中。

- **`setns(fd, nstype)`**: 将**当前进程**加入一个**已经存在**的Namespace。`fd`是一个指向`/proc/[pid]/ns/[type]`文件的文件描述符。这常用于调试，例如`nsenter`命令就是基于此实现，让我们能"进入"一个正在运行的容器的Namespace。

- **`unshare(flags)`**: 将**当前进程**从父进程共享的Namespace中分离，并加入一个新的Namespace。这允许一个现有进程改变其执行上下文，而无需创建新进程。`unshare`命令就是这个系统调用的封装。

- **`/proc/[pid]/ns/` 目录**:
  - 系统上的每个进程，在其对应的`/proc/[pid]`目录下，都有一个`ns`子目录。
  - `ls -l /proc/$$/ns` ( `$$`代表当前进程ID)
  ```
  lrwxrwxrwx 1 root root 0 Jan 1 10:00 cgroup -> 'cgroup:[4026531835]'
  lrwxrwxrwx 1 root root 0 Jan 1 10:00 ipc -> 'ipc:[4026531839]'
  lrwxrwxrwx 1 root root 0 Jan 1 10:00 mnt -> 'mnt:[4026531840]'
  lrwxrwxrwx 1 root root 0 Jan 1 10:00 net -> 'net:[4026531956]'
  lrwxrwxrwx 1 root root 0 Jan 1 10:00 pid -> 'pid:[4026531836]'
  lrwxrwxrwx 1 root root 0 Jan 1 10:00 user -> 'user:[4026531837]'
  lrwxrwxrwx 1 root root 0 Jan 1 10:00 uts -> 'uts:[4026531838]'
  ```
  - 这里的每个链接都指向一个Namespace的唯一标识符（inode number）。如果两个进程的某个ns链接指向相同的标识符，说明它们位于同一个该类型的Namespace中。

### 2. 内核实现原理

- **`task_struct`**: 在Linux内核中，每个进程都由一个 `task_struct` 结构体描述。它是内核管理进程的核心数据结构，包含了进程状态、调度信息、文件描述符表等所有信息。其中就包含一个指向 `nsproxy` 结构体的指针。
- **`nsproxy`**: 这个结构体专门用来打包和管理进程所属的多个Namespace实例的指针（如`uts_ns`, `ipc_ns`, `mnt_ns`, `pid_ns_for_children`, `net_ns`）。
- **写时复制 (Copy-on-Write)**: 当使用 `fork()` 或 `clone()` 创建新进程时：
  - 如果**没有**指定 `CLONE_NEW*` 标志，子进程的`task_struct->nsproxy`将直接指向父进程的`nsproxy`结构体，意味着它们共享所有Namespace。
  - 如果指定了 `CLONE_NEW*` 标志（例如`CLONE_NEWUTS`），内核会为子进程**创建一个新**的`uts_namespace`结构体实例，然后复制父进程的`nsproxy`，并将其中`uts_ns`指针替换为指向这个新实例的地址。这就是所谓的“写时复制”，只复制发生变化的Namespace部分。
- **虚拟化视图**: 当一个进程执行系统调用（如`gethostname()`、`ps`、`ifconfig`）时，内核代码会获取当前进程的`task_struct`，通过`nsproxy`找到它所属的Namespace，然后基于这个Namespace的上下文来返回结果。例如，`gethostname()`会返回当前UTS Namespace的主机名，而不是全局主机名。这就在内核层面实现了“视图隔离”。

### 3. 七种主要的 Namespace 详解

（需要root权限执行以下`unshare`命令）

- **UTS (UNIX Time-sharing System) Namespace**: 隔离主机名和NIS域名。
  - **实践**:
    ```bash
    # 1. 查看当前主机名
    hostname
    # 2. 在新的UTS Namespace中启动一个shell
    sudo unshare --uts --fork /bin/bash
    # 3. 在新shell中修改主机名
    hostname -b my-new-hostname
    # 4. 查看新主机名，会发现已改变
    hostname
    # 5. exit退出后，回到原shell，主机名未变
    hostname
    ```
  - **应用**: 每个容器可以有自己独立的`hostname`，便于在网络中识别。

- **Mount (mnt) Namespace**: 隔离文件系统挂载点。
  - **实践**:
    ```bash
    # 在新的Mount Namespace中启动shell，并确保挂载事件不传播
    sudo unshare --mount --fork --propagation unchanged /bin/bash
    # 在新shell中，创建一个挂载点并挂载一个tmpfs
    mkdir /mytmp
    mount -t tmpfs none /mytmp
    # 挂载在新shell中可见
    ls /mytmp
    # exit退出后，在原shell中/mytmp目录依然存在，但没有挂载内容
    ls /mytmp
    ```
  - **应用**: 这是实现容器拥有独立根文件系统(rootfs)的基础。容器可以`chroot`到一个新的根目录，并且它内部的`mount`/`umount`操作不会影响到宿主机。

- **PID (Process ID) Namespace**: 隔离进程ID树。
  - **实践**:
    ```bash
    # 在新的PID Namespace中启动shell
    sudo unshare --pid --fork --mount-proc /bin/bash
    # 在新shell中查看进程
    ps aux
    ```
    你会发现，bash进程的PID是1，并且只能看到这个Namespace内部的进程。`--mount-proc`是必要的，因为它会为新的PID Namespace挂载一个新的`/proc`文件系统。
  - **应用**: 容器内的init进程（通常是应用的入口程序）PID为1，拥有管理子进程的特权，且无法看到或影响宿主机上的其他进程。

- **Network (net) Namespace**: 隔离网络栈（网络设备、IP地址、路由表、防火墙规则、端口等）。
  - **实践**:
    ```bash
    # 在新的Network Namespace中启动shell
    sudo unshare --net --fork /bin/bash
    # 在新shell中查看网络接口，会发现只有一个lo环回接口，并且是down状态
    ip addr
    # 激活它
    ip link set lo up
    ip addr
    ```
    新创建的Net NS是完全隔离的。要让它和外界通信，通常需要使用`veth pair`（虚拟网卡对）技术，一端留在宿主机，另一端“插”到这个Namespace中。
  - **应用**: Docker容器网络模型（bridge, host, none, container）的基石。每个容器都有自己独立的IP地址和端口空间。

- **IPC (Inter-Process Communication) Namespace**: 隔离System V IPC对象（信号量、共享内存、消息队列）和POSIX消息队列。
  - **实践**:
    ```bash
    # 在宿主机上创建IPC对象
    ipcmk -Q
    ipcs -q
    # 在新的IPC Namespace中启动shell
    sudo unshare --ipc --fork /bin/bash
    # 在新shell中查看，看不到宿主机的IPC对象
    ipcs -q
    ```
  - **应用**: 防止容器内的进程与宿主机或其他容器的进程发生IPC干扰。

- **User Namespace**: 隔离用户和组ID (UID/GID)。
  - **作用**: 允许一个非root用户在容器内“看起来”像root用户。这是实现**Rootless Containers（无根容器）**的关键。
  - **原理**: 它建立了一个从Namespace内部UID到宿主机UID的映射。例如，可以将Namespace内的UID 0 (root) 映射到宿主机上的UID 1000 (一个普通用户)。这样，即使进程在容器内以root身份运行，它在宿主机上实际的权限也只是一个普通用户，极大地提升了安全性。
  - **实践**:
    ```bash
    # 在新的User和UTS Namespace中启动shell
    unshare --user --uts --fork /bin/bash
    # 在新shell中查看用户ID，会发现是nobody(65534)，因为没有建立映射
    id
    # 修改主机名，会因为权限不足而失败
    hostname my-test
    ```
    配置User NS映射比较复杂，通常由容器运行时自动处理。

- **Cgroup Namespace**: 隔离Cgroup视图。
  - **作用**: 当进程在一个Cgroup NS中查看`/proc/self/cgroup`文件时，它看到的路径是相对于它在宿主机上所属的Cgroup树的路径，而不是从宿主机的cgroupfs根目录开始的绝对路径。
  - **应用**: 主要用于让容器内的`systemd`等系统管理工具能正确地管理其自身的Cgroup子树，增强了容器的独立性和兼容性。

### 4. 实战工具

- **`unshare`**: 创建并执行一个程序在新的Namespace中。`unshare [options] [program [arguments]]`
- **`nsenter`**: 进入一个或多个已存在的Namespace，并执行指定程序。 `nsenter --target [pid] --[namespace_type] [program]`
  - **示例**: `sudo nsenter --target <container_pid> --net --pid /bin/bash` 可以进入一个正在运行的容器的网络和PID空间，非常适合调试。
- **`lsns`**: 列出系统中当前存在的所有Namespace及其类型、归属进程等信息。

---

## 三、 深入理解 Control Groups (Cgroups)

### 1. Cgroups 核心概念

- **任务 (Task)**: 在Cgroups的语境下，就是一个系统进程。
- **控制组 (Control Group)**: 一组按照某种标准划分的进程。Cgroups允许你将大量进程组织起来，并以组为单位进行资源控制。在cgroup文件系统中，一个控制组就是一个目录。
- **层级 (Hierarchy)**: Cgroups的核心设计。一个层级是由一系列Cgroup组成的树状结构。系统可以有多个独立的层级。
- **子系统/控制器 (Subsystem/Controller)**: 真正负责实现资源控制的模块。每个子系统代表一种可控制的资源。
  - `cpu`: 控制CPU时间分配。
  - `memory`: 控制内存使用量。
  - `blkio`: 控制块设备（磁盘、SSD）的I/O速度。
  - `pids`: 控制一个组内的进程数量。
  - ...等等。

### 2. 内核实现原理

- **虚拟文件系统 (`cgroupfs`)**: Cgroups通过一个名为`cgroupfs`的特殊虚拟文件系统暴露给用户空间。管理员或程序通过在这个文件系统上进行标准的文件和目录操作（`mkdir`, `echo`, `cat`）来与内核进行交互。
  - `mount -t cgroup` 可以看到cgroup的挂载点，通常在`/sys/fs/cgroup`。
  - **创建控制组**: `mkdir /sys/fs/cgroup/memory/my-group`
  - **配置限制**: `echo "100M" > /sys/fs/cgroup/memory/my-group/memory.limit_in_bytes`
  - **添加进程**: `echo $$ > /sys/fs/cgroup/memory/my-group/cgroup.procs`

- **内核挂钩 (Kernel Hooks)**: Cgroups的控制能力并非凭空而来，而是其各个子系统在内核处理资源的关键路径上“植入”了检查点（挂钩）。
  - **`cpu` 控制器**: 在**进程调度器**的逻辑中加入挂钩。每次调度器选择下一个要运行的进程时，会检查该进程所属的Cgroup的CPU配额，决定是否分配CPU时间以及分配多少。
  - **`memory` 控制器**: 在**内存管理**模块中加入挂钩，特别是在**缺页中断**（page fault）的处理函数中。当进程试图访问一块新的内存时，会触发缺页中断，此时内存控制器会检查进程已使用的内存是否超过其Cgroup的限制，如果超过，则触发OOM (Out Of Memory) Killer。
  - **`pids` 控制器**: 在 `fork` 系统调用的内核实现中加入挂钩。每当有进程试图创建子进程时，会检查其Cgroup的进程数是否已达上限。

- **层级规则**: 进程加入某个Cgroup，会自动受到该Cgroup及其所有父级Cgroup的限制。这使得资源控制可以进行精细的层级管理。

### 3. Cgroups V1 vs. Cgroups V2

| 特性 | Cgroups V1 | Cgroups V2 |
| :--- | :--- | :--- |
| **层级** | **多层级**。`cpu`和`memory`子系统可以分别挂载在不同的层级树上 | **单一统一层级**。所有可用的控制器都挂载在同一个层级树上 |
| **控制器管理** | 灵活但混乱。一个进程可能同时属于多个不同层级的Cgroup | **清晰**。一个进程只属于一个Cgroup。在层级内部启用/禁用控制器 |
| **接口** | 文件名杂乱，如`tasks` vs `cgroup.procs` | 接口更统一、规范，如`cgroup.procs`, `cgroup.threads` |
| **生态支持** | 传统默认，广泛支持 | **未来趋势**。Docker(19.03+), systemd, Kubernetes(1.22+推荐)等现代工具已全面支持并推荐使用 |

**核心区别**: V2的单一层级模型解决了V1中因多层级导致的管理复杂性和逻辑冲突问题，是更先进的设计。

### 4. 关键子系统 (Controllers) 详解 (以V1为例，V2接口类似)

#### **`cpu` 子系统**
- **`cpu.shares`** (相对值): 按比例分配CPU。如果A组的shares是1024，B组是2048，那么当CPU繁忙时，B组获得的CPU时间将是A组的两倍。这是K8s `requests.cpu`的实现基础。
- **`cpu.cfs_period_us` & `cpu.cfs_quota_us`** (绝对值): 在一个`period`（周期，微秒）内，该组最多只能使用`quota`（配额，微秒）的CPU时间。
  - **示例**: 限制使用一个CPU核心的50% -> `period`设为100000 (100ms), `quota`设为50000 (50ms)。
  - 这是K8s `limits.cpu`的实现基础。

#### **`memory` 子系统**
- **`memory.limit_in_bytes`** (硬限制): Cgroup中所有进程的总内存使用（物理内存+swap）不能超过此值。一旦超出，内核会立即触发OOM Killer，杀死组内某个进程。这是K8s `limits.memory`的实现基础。
- **`memory.soft_limit_in_bytes`** (软限制): 内核会尽力将内存使用保持在此值以下，但不是强制的。只有当系统内存紧张时，才会优先回收超过软限制的Cgroup的内存。
- **`memory.oom_control`**: `oom_kill_disable`设为1可以禁用OOM Killer，但不推荐，可能导致系统僵死。

#### **实践：限制进程内存使用**
```bash
# 1. 创建一个memory cgroup
sudo mkdir /sys/fs/cgroup/memory/my-mem-group

# 2. 设置100MB的硬限制
sudo sh -c 'echo 104857600 > /sys/fs/cgroup/memory/my-mem-group/memory.limit_in_bytes'

# 3. 启动一个后台程序，持续申请200MB内存
stress --vm 1 --vm-bytes 200M &
STRESS_PID=$!

# 4. 将该进程的PID加入cgroup
sudo sh -c 'echo $STRESS_PID > /sys/fs/cgroup/memory/my-mem-group/cgroup.procs'

# 5. 很快，你会看到 "Killed" 的输出，因为stress进程因超出内存限制被OOM杀死了
#    可以通过 dmesg | grep Killed 命令查看内核日志
```

---

## 四、 融会贯通：Namespaces + Cgroups = 容器

### 1. 手动构建一个简易容器的步骤

以下脚本展示了如何仅用`unshare`, `cgroup`和`chroot`等基本命令，构建一个有独立主机名、进程空间、文件系统和资源限制的"容器"：

```bash
#!/bin/bash
# 需要busybox静态二进制文件 和 一个名为rootfs的目录

# 1. 创建Cgroup用于资源限制
sudo mkdir -p /sys/fs/cgroup/cpu/my_container
sudo mkdir -p /sys/fs/cgroup/memory/my_container
sudo sh -c 'echo 512 > /sys/fs/cgroup/cpu/my_container/cpu.shares' # 限制CPU
sudo sh -c 'echo 100M > /sys/fs/cgroup/memory/my_container/memory.limit_in_bytes' # 限制内存

# 2. 将当前shell进程加入Cgroup (子进程会继承)
sudo sh -c "echo $$ > /sys/fs/cgroup/cpu/my_container/cgroup.procs"
sudo sh -c "echo $$ > /sys/fs/cgroup/memory/my_container/cgroup.procs"

# 3. 使用 unshare 创建新的 Namespaces
sudo unshare --fork --pid --mount-proc -u -i -m --propagation private /bin/bash -c "
  # 内部执行的命令

  # 4. 切换根文件系统
  mount --bind rootfs rootfs # 使rootfs成为一个挂载点
  cd rootfs
  pivot_root . old_rootfs
  cd /
  umount -l /old_rootfs
  rmdir /old_rootfs

  # 5. 设置新主机名
  hostname 'my-tiny-container'

  # 6. 启动一个shell，你现在就在'容器'里了！
  /bin/sh
"

# 清理Cgroups
sudo rmdir /sys/fs/cgroup/cpu/my_container
sudo rmdir /sys/fs/cgroup/memory/my_container
```
这个过程虽然简陋，但完整地演示了容器的本质：**一个被Namespace隔离、被Cgroups限制的特殊进程**。

### 2. OCI 运行时 (如 runc) 的角色

我们不可能每次都手动执行上面的复杂脚本。**OCI (Open Container Initiative) 运行时** 就是将这个过程标准化、产品化的工具。`runc`是其中最著名的实现。

- `runc`是一个轻量级的命令行工具，它读取一个符合OCI规范的配置文件（`config.json`），然后准确地执行所有必要的`clone`, `setns`, `pivot_root`操作，并配置好Cgroups，最终拉起一个真正的容器进程。
- **Docker, containerd, CRI-O**这些更高级的容器管理器，在需要创建容器时，最终都会调用`runc`这样的底层OCI运行时来完成核心的创建工作。

---

## 五、 在云原生生态中的核心应用

### 1. 容器运行时 (如 Docker, containerd)

- **`docker run`命令的背后**:
  - `docker run -it --hostname web01 --memory 512m nginx`
  - **隔离的基础**: Docker Daemon接收到命令后，会为新容器准备好一整套独立的Namespace（UTS, PID, Net, ...）。
  - **资源限制的执行者**: 同时，它会在`/sys/fs/cgroup/memory/docker/[container-id]/`下创建一个Cgroup，并向`memory.limit_in_bytes`文件写入`536870912`（512MB）。然后调用`runc`来启动Nginx进程，并将其PID加入该Cgroup。

### 2. 容器编排 (Kubernetes)

Kubernetes将Namespace和Cgroup的应用提升到了一个全新的高度，使其成为大规模集群资源管理和调度策略的基石。

- **Pod模型的基石**:
  - Pod是K8s中最小的部署单元，它可以包含一个或多个容器。Pod的设计巧妙地利用了Namespace共享机制。
  - 每个Pod启动时，会先创建一个非常小的、几乎不消耗资源的**Pause容器**。Pause容器创建了Pod所需的一系列Namespace（特别是Network, IPC, UTS）。
  - 随后，Pod中定义的所有业务容器被创建，并被配置为**加入（`setns`）到Pause容器的Namespace**中。
  - **结果**:
    - 所有业务容器共享同一个Network Namespace，因此它们可以通过`localhost`互相通信，共享端口空间，如同在一台物理机上。
    - 它们也共享IPC Namespace和UTS Namespace。
    - PID Namespace默认不共享，但可以配置共享(`shareProcessNamespace: true`)。
  - **结论**: **Pause容器就是Pod网络和进程间通信模型的“锚点”，Pod之所以能像一个独立的“逻辑主机”，正是得益于这种Namespace的精巧运用。**

- **服务质量 (QoS) 的保障**:
  - Kubernetes通过`requests`和`limits`两个字段来管理容器资源，并据此将Pod划分为三种QoS等级。这一切最终都由节点上的`kubelet`组件转换为对Cgroup的配置。
    - **`spec.containers.resources.limits`**:
      - `cpu`: "1" -> `cpu.cfs_quota_us` = 100000, `cpu.cfs_period_us` = 100000 (100% of one core)
      - `memory`: "500Mi" -> `memory.limit_in_bytes` = 524288000
      - `limits`定义了**硬限制**，超出后进程会被“惩罚”（CPU节流或内存OOM杀死）。
    - **`spec.containers.resources.requests`**:
      - `cpu`: "0.5" -> `cpu.shares` = 512 (K8s中1 core = 1024 shares)
      - `memory`: "256Mi"
      - `requests`主要用于**调度**（确保节点有足够资源）和在资源竞争时作为**分配权重**。
  - **QoS等级与Cgroup的关系**:
    - **Guaranteed**: `requests` == `limits` (且不为0)。资源得到完全保障，优先级最高。
    - **Burstable**: `requests` < `limits`。资源有一定保障，但可能被压缩。`kubelet`会根据`requests`配置`cpu.shares`。
    - **BestEffort**: `requests`和`limits`都未设置。优先级最低，当节点资源耗尽时最先被驱逐。

- **多租户与资源切片**:
  - `kubelet`会在宿主机上为所有Pod创建一个顶级的Cgroup（如`/kubepods`）。在这个Cgroup下，会根据QoS等级再创建子Cgroup（如`/kubepods/burstable`）。每个Pod会再有自己的Cgroup。
  - `.../kubepods.slice/kubepods-burstable.slice/kubepods-burstable-pod<UID>.slice/`
  - 这种层级结构使得K8s可以对整个节点、某类Pod、某个Pod进行精细的资源控制和统计。在多租户场景下，可以为不同租户的Pod设置不同的根Cgroup，从而实现集群级别的资源切片和隔离。

---

## 六、 附录与进阶

### 1. 其他 Linux 容器技术

Namespaces和Cgroups构成了容器的“墙”和“屋顶”，但一个完整的安全堡垒还需要“门禁”和“监控”。

- **Seccomp (Secure Computing Mode)**: 像一个“白名单”，限制一个进程**可以使用的系统调用**。Docker默认提供一个Seccomp配置文件，禁止了如`reboot`等约44个危险的系统调用，防止容器内的进程影响宿主机。
- **AppArmor/SELinux**: 提供了**强制访问控制 (MAC)**。它们可以限制进程对**文件、目录、网络端口**等资源的访问权限。例如，可以定义一个AppArmor Profile，只允许Web服务器进程读取`/var/www`目录，写入`/var/log`目录，监听80端口。
- **Capabilities**: 将传统的`root`用户的超级权限（all-powerful）细粒度地划分为一系列独立的“能力”。例如，一个进程只需要绑定到1024以下的端口，那么只需赋予它`CAP_NET_BIND_SERVICE`这个Capability即可，而无需给予完整的root权限。容器运行时会为容器默认剥离大量非必需的Capabilities，只保留一小部分。

这四项技术（Namespaces, Cgroups, Seccomp, Capabilities）共同构成了现代Linux容器安全的核心支柱。

### 2. 推荐工具与资源

- **`dive`**: 一个用于探索Docker/OCI镜像、层内容和发现减小镜像大小方法的工具。
- **`ctop`**: 容器版的`top`命令，可以实时显示多个容器的资源使用情况。
- **官方文档**: `man namespaces`, `man cgroups`, `man unshare`等永远是第一手的最权威资料。
- **书籍**: 《Docker - 从入门到实践》、《Kubernetes权威指南》、《深入剖析Kubernetes》等。
