# Docker 文件系统存储笔记（基于 Overlay2）

## 1. 总览：Docker 的数据根目录

默认情况下，Docker 将所有数据保存在 **`/var/lib/docker`** 下。  
使用 `overlay2` 存储驱动时，核心子目录有：

```
/var/lib/docker/
├── overlay2/                # 存储所有层（镜像层 + 容器层）的实际文件数据
├── image/overlay2/          # 镜像的元数据（分层关系、配置、仓库映射等）
│   ├── distribution/        # 远程仓库信息
│   ├── imagedb/             # 镜像配置和清单 (content/sha256)
│   ├── layerdb/             # 每个层的元数据（大小、父链、diff_id）
│   └── repositories.json    # 镜像名:tag -> manifest 映射
└── ...
```

---

## 2. 镜像的保存方式（只读层）

### 2.1 分层存储
- Docker 镜像由一系列**只读层**堆叠而成，每一层只记录与下一层的**差异**（类似 Git 的增量）。
- 拉取镜像时，Docker 会逐层下载并解压到 `/var/lib/docker/overlay2/<layer_id>/diff/` 目录。

### 2.2 镜像层目录结构（示例）
```
/var/lib/docker/overlay2/<layer_id>/
├── diff/           # 该层的文件系统内容（新增/修改/删除的文件）
├── link            # 该层的短链接名（用于缩短路径）
├── lower           # 指向父层（下层）的链接文件（仅非底层存在）
└── work/           # 保留（通常为空，仅当该层作为挂载点时使用）
```

### 2.3 镜像元数据
- **layerdb**：记录每个层的 `diff_id`、大小、父层指针等。
- **imagedb**：存储镜像的 JSON 配置，其中 `RootFS.Layers` 字段列出了该镜像包含的所有层（按顺序）。
- **repositories.json**：将 `镜像名:tag` 映射到具体镜像的 digest。

### 2.4 示例命令
```bash
# 查看镜像包含哪些层（从上到下）
docker image inspect ubuntu:latest | jq '.[].RootFS.Layers'

# 查看某个层的实际存储路径
docker image inspect ubuntu:latest | jq '.[].GraphDriver.Data'
```

---

## 3. 容器运行时的文件系统（添加可写层）

### 3.1 容器层的生成
当执行 `docker run` 时，Docker 会：
1. 基于镜像的所有只读层，创建一个**新的可写层**（容器层）。
2. 使用 OverlayFS 将只读层和可写层联合挂载到一个**合并目录**，作为容器的根文件系统。

### 3.2 容器层的目录结构
```
/var/lib/docker/overlay2/<container_id>/
├── diff/           # 容器的可写层（UpperDir） – 存储所有修改
├── link            # 短链接名
├── lower           # 文本文件，记录所有下层镜像层的短链接路径（用 : 分隔）
├── merged/         # 联合挂载点（容器看到的完整根文件系统）
└── work/           # OverlayFS 工作目录（用于原子操作）
```

### 3.3 核心字段解释（来自 `docker inspect` 的 `GraphDriver.Data`）

| 字段        | 对应路径                                      | 作用                                           |
|-------------|-----------------------------------------------|------------------------------------------------|
| **LowerDir**| 多个 `.../diff` 目录（冒号分隔）              | 所有下层只读镜像层（从底到顶顺序）            |
| **UpperDir**| `/var/lib/docker/overlay2/<id>/diff`          | 容器的可写层，存放所有文件修改                 |
| **WorkDir** | `/var/lib/docker/overlay2/<id>/work`          | OverlayFS 内部临时工作目录（原子操作）        |
| **MergedDir**| `/var/lib/docker/overlay2/<id>/merged`       | 联合挂载点，容器进程的根目录 `/`              |

---

## 4. OverlayFS 工作原理与目录联系

### 4.1 四个核心组件的关系
OverlayFS 通过以下挂载命令将多个目录合并为一个视图：
```bash
mount -t overlay overlay \
      -o lowerdir=<LowerDir>,upperdir=<UpperDir>,workdir=<WorkDir> \
      <MergedDir>
```

- **LowerDir**：只读层，可多个（例如镜像各层）。
- **UpperDir**：可写层，只有一个（容器层）。
- **WorkDir**：辅助目录，与 UpperDir 同一文件系统，用于支持 rename 等原子操作。
- **MergedDir**：最终呈现的联合文件系统，容器内看到的 `/`。

### 4.2 文件读写规则（写时复制 CoW）
- **读文件**：优先从 UpperDir 查找；若不存在，则从 LowerDir 中从右到左（优先级从高到低）依次查找，返回第一个找到的。
- **修改文件**：若文件在 UpperDir 中不存在，则从 LowerDir 复制到 UpperDir（CoW），然后再修改。
- **删除文件**：在 UpperDir 中创建一个同名 **whiteout 文件**（`.wh.<filename>`），在合并视图中“遮盖”掉下层同名文件。
- **新建文件**：直接在 UpperDir 中创建。

### 4.3 特殊文件 `lower` 的作用
路径：`/var/lib/docker/overlay2/<id>/lower`（**这是一个文件**）  
内容：记录当前层所依赖的所有 lower 层的**短链接路径**，例如：
```
l/ABC:l/DEF:l/GHI
```
其中 `l/ABC` 实际指向 `/var/lib/docker/overlay2/l/ABC`，而 `l/ABC` 又是一个指向实际层 `diff/` 目录的软链接。  
作用：给 OverlayFS 的 `lowerdir=` 参数提供路径列表，并避免挂载参数过长（Linux 对路径长度有限制）。

### 4.4 目录 `work` 的作用
路径：`/var/lib/docker/overlay2/<id>/work`（**是一个目录**）  
作用：OverlayFS 内部用于实现原子操作的工作区（如复制、重命名等）。内核会在此处生成临时文件，**用户不应手动修改**。`work` 目录不会出现在容器的 `merged` 视图中。

### 4.5 短链接目录 `/var/lib/docker/overlay2/l/`
- 该目录下存放许多**指向各层 `diff/` 目录的短软链接**（链接名很短，如 `ABCDEF`）。
- 目的：缩短路径长度，避免 `lowerdir` 参数因路径过长而超限。

---

## 5. 常用调试与查看命令

```bash
# 查看容器的联合挂载参数
docker inspect <container_id> | jq '.[].GraphDriver.Data'

# 查看容器 merged 目录内容（需要 root）
sudo ls -l /var/lib/docker/overlay2/<container_id>/merged

# 查看容器的可写层变化
sudo ls -l /var/lib/docker/overlay2/<container_id>/diff

# 查看镜像各层的实际占用
docker system df -v

# 进入容器的“物理”可写层（比 docker exec 更底层）
sudo nsenter -t <container_pid> -m ls /   # 需要先获取容器 pid
```

---

## 6. 总结对比表

| 对象       | 存储位置（示例）                               | 读写性     | 生命周期                 |
|------------|------------------------------------------------|------------|--------------------------|
| 镜像层     | `/var/lib/docker/overlay2/<layer_id>/diff/`   | 只读       | `docker rmi` 时删除      |
| 容器可写层 | `/var/lib/docker/overlay2/<container_id>/diff/`| 读写       | `docker rm` 时删除       |
| lower 文件 | `/var/lib/docker/overlay2/<id>/lower`         | 只读（配置）| 与层/容器同生命周期      |
| work 目录  | `/var/lib/docker/overlay2/<id>/work/`         | 内核使用   | 与层/容器同生命周期      |
| merged 目录| `/var/lib/docker/overlay2/<id>/merged/`       | 联合视图   | 容器停止后仍存在（可重新挂载）|
| 短链接池   | `/var/lib/docker/overlay2/l/`                 | 软链接     | Docker 自动管理           |

---

## 7. 关键概念速记

- **镜像 = 一组只读层**，每层是一个 `diff` 目录。
- **容器 = 镜像只读层 + 一个可写层（UpperDir）**，通过 OverlayFS 联合挂载到 `merged`。
- **写时复制**：修改文件时，先将文件从 lower 复制到 upper，再修改。
- **whiteout 文件**：`.wh.xxx` 用于删除下层文件。
- **lower 文件**：记录联合挂载需要的所有 lower 层路径（短链接形式）。
- **work 目录**：OverlayFS 原子操作的工作空间，用户不要碰。
- **`/var/lib/docker/overlay2/l/`**：存放短链接，防止路径过长。

> 💡 理解 OverlayFS 的这四个目录（lower, upper, work, merged）以及 Docker 如何组织它们，是掌握 Docker 存储层的核心。

---

**附：学习路径建议**  
1. 手动拉取一个镜像，用 `docker image inspect` 查看层列表。  
2. 进入 `/var/lib/docker/overlay2`，观察实际的层目录和 `diff` 内容。  
3. 运行一个容器，用 `docker inspect` 查看 `GraphDriver.Data`，然后 `ls -l` 对应的 `lower` 文件、`diff`、`merged`。  
4. 在容器内创建、修改、删除文件，再到主机上查看 `diff` 目录的变化（whiteout 文件）。  
5. 尝试 `mount | grep overlay` 查看主机上实际挂载的 OverlayFS 设备。

掌握了这些，你对 Docker 文件系统的理解就达到了“知其所以然”的程度。