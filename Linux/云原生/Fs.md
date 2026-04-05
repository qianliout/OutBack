# Docker & Containerd 容器分层与存储深度解析

---

## Part 1: Docker 镜像与分层核心概念 (Docker Image & Layers)

### 1.1. 什么是 Docker 镜像 (Image)? 

> **定义**：Docker 镜像是一个用于创建 Docker 容器的**只读模板**。它打包了应用程序及其运行所需的所有依赖项，包括代码、运行时、库、环境变量和配置文件。

-   **镜像与容器的关系**：
    -   **镜像是静态定义 (Static Definition)**：它是一个不可变的实体，包含了应用程序运行所需的一切。
    -   **容器是运行时实例 (Runtime Instance)**：它是基于镜像创建的、可运行的进程环境。可以从单个镜像创建多个相互隔离的容器实例。当容器被创建时，会在镜像之上添加一个[[#1.4. 容器层 (Container Layer)]]。

### 1.2. 什么是镜像分层 (Layer)? 

> **定义**：镜像并非一个单一的大文件，而是由一系列**文件系统变更集**（changeset）堆叠而成。每一个变更集就是一层 (Layer)。

这类似于图像处理软件中的图层，每个操作都创建一个新层。

-   **层的特性**：
    *   **只读性 (Read-only)**：镜像的每一层在构建完成后都是不可修改的。这保证了镜像的一致性和可重复性。
    *   **堆叠性 (Stackable)**：镜像是多层文件系统的联合，通过[[#Part 2: 核心机制：CoW 与 OverlayFS 深度解析|存储驱动]]技术，将这些层以堆叠方式聚合，最终形成一个统一的文件系统视图。
    *   **唯一性与内容寻址 (Content-addressable)**：每一层都通过其内容的哈希值（SHA256校验和）来唯一标识。如果多个镜像使用了内容完全相同的层，那么在物理存储上只会保存一份，实现了高效的存储和传输。这也构成了[[#1.3. Dockerfile 与层的关系|Docker 构建缓存]]的基础。

### 1.3. Dockerfile 与层的关系

`Dockerfile` 是用于构建 Docker 镜像的定义文件，其中的大多数指令都会创建一个新的层。

-   **`FROM` 指令**：
    -   `FROM ubuntu:22.04`
    -   这条指令指定了基础镜像。它不是创建一个新层，而是将 `ubuntu:22.04` 镜像已有的所有层引入到当前镜像中。

-   **`RUN`, `COPY`, `ADD` 指令**：
    -   这些指令会在当前所有层之上添加一个新的层。
    -   `RUN apt-get update && apt-get install -y nginx`：该指令执行后，所有新安装的文件和对文件系统的修改都会被记录到一个新的层中。
    -   `COPY ./app /app`：该指令会将本地的 `app` 目录复制到镜像的 `/app` 目录，这个复制操作本身会形成一个新层。

-   **构建缓存 (Build Cache)**：
    -   Docker 在构建镜像时，会检查 `Dockerfile` 中的下一条指令。如果该指令和它所依赖的父层（之前的层）与之前某次构建完全相同，Docker 就会直接使用缓存中的层，而不是重新执行该指令。
    -   **缓存失效**：一旦某条指令因为内容变更（如 `COPY` 的源文件被修改）或命令本身不同而无法使用缓存，其后的所有指令都必须重新执行。这就是为什么[[#4.2. 优化镜像大小和构建速度|优化 Dockerfile 指令顺序]]如此重要。

### 1.4. 容器层 (Container Layer)

-   当使用 `docker run` 基于镜像创建容器时，Docker 会在镜像所有只读层的顶部添加一个**可写层**，也称为**容器层**。
-   所有对容器文件系统的运行时修改都发生在此层，例如创建、修改或删除文件。
-   这个机制由[[#2.1. 写时复制 (Copy-on-Write, CoW) 策略|写时复制 (Copy-on-Write, CoW) 策略]]实现，确保了底层镜像是不可变的。
-   当容器被删除时（`docker rm`），其对应的可写层也会被一并删除，而底层的镜像层不受任何影响。

---

## Part 2: 核心机制：CoW 与 OverlayFS 深度解析

### 2.1. 写时复制 (Copy-on-Write, CoW) 策略

CoW 是 Docker 实现存储效率和分层管理的核心机制。

-   **核心思想**：允许多个进程或实例共享同一份只读数据。只有当某个实例需要修改这份数据时，系统才会复制一份副本给该实例进行修改。这避免了不必要的复制，大大节省了存储空间。

-   **CoW 在 Docker 中的应用 (以 `overlay2` 为例)**：
    -   **文件读取 (Read)**：
        1.  从顶部的容器可写层 (`upperdir`) 开始查找文件。
        2.  如果找到，则直接读取并返回。
        3.  如果未找到，则从上至下依次在镜像的只读层 (`lowerdir`) 中查找。
        4.  找到后，读取并返回。
        5.  如果所有层都找不到，则返回“文件不存在”错误。

    -   **文件修改/写入 (Write/Modify)**：
        1.  当要修改一个存在于 `lowerdir` 的文件时，驱动会触发“写时复制”操作。
        2.  将该文件从其所在的只读层完整地复制到顶部的可写层 (`upperdir`)。
        3.  容器的修改操作将作用于可写层中的这个**副本**。
        4.  从此，该容器对该文件的所有读取都将直接命中可写层中的已修改版本，只读层中的原始文件被“隐藏”。

    -   **文件删除 (Delete)**：
        1.  当要删除一个存在于 `lowerdir` 的文件时，由于只读层不可修改，无法真正删除。
        2.  驱动会在可写层 (`upperdir`) 中创建一个特殊的**“白障” (whiteout) 文件**。
        3.  该文件是一种标记，用于在统一文件系统视图中隐藏底层的文件。因此，在容器内部，该文件将不再可见，但它实际上仍然存在于底层的镜像层中。

### 2.2. 存储驱动 (Storage Driver) 与 `overlay2`

存储驱动是 Docker 的核心组件，它负责实现分层、堆叠和 CoW 机制。`overlay2` 是目前 Linux 系统上推荐的、性能最好且使用最广泛的驱动，它利用了 Linux 内核的 **OverlayFS** 联合文件系统技术。

-   **OverlayFS 核心目录结构**：
    *   `lowerdir`：只读层，可以有多个，对应 Docker 的镜像层。
    *   `upperdir`：可写层，只有一个，对应 Docker 的容器层。
    *   `mergeddir`：统一视图，是 `lowerdir` 和 `upperdir` 联合挂载的结果，呈现给容器的最终文件系统。
    *   `workdir`：内部工作目录，用于完成原子操作，对用户透明。

-   **深入理解 `workdir` 的作用 (原子性保证)**
    > `workdir` 是 OverlayFS 的一个关键内部组件，它的存在是为了**保证文件操作的原子性**，防止在操作过程中出现数据不一致或损坏的状态。

    标准的 OverlayFS 操作，如将文件从 `lowerdir` “复制到” `upperdir`（即 "copy-up"），并非真正的瞬时操作。如果直接在 `upperdir` 中创建并写入文件，一旦中途发生断电或系统崩溃，`upperdir` 中就会留下一个不完整的、已损坏的文件。

    `workdir` 通过充当一个**“准备区”**或**“中转站”**来解决这个问题：
    1.  **准备阶段**：当需要执行 copy-up 操作时，OverlayFS 首先在 `workdir` 内部完成所有的准备工作（例如，创建一个完整的硬链接或文件副本）。
    2.  **原子移动**：一旦 `workdir` 中的文件准备就绪，OverlayFS 会使用一个**原子操作**（通常是 `rename(2)` 系统调用）将其瞬间移动到 `upperdir` 中。`rename` 操作在同一个文件系统内移动文件是原子的，这意味着它要么完全成功，要么根本没发生，不会出现中间状态。
    3.  **最终结果**：由于这个机制，`mergeddir` 中看到的文件状态只会从“不存在于 `upperdir`”瞬间切换到“完整存在于 `upperdir`”，从而避免了任何读取到部分写入文件的可能性。

    简而言之，`workdir` 是 OverlayFS 实现健壮性的幕后功臣，确保了即使在非正常情况下，文件系统的状态也能保持一致。

### 2.3. 动手实践：亲手查看和验证 OverlayFS

**注意**：以下命令需要在 Docker 主机上以 `root` 或 `sudo` 权限执行。

1.  **启动一个容器并获取其信息**
    ```bash
    # 启动一个简单的容器
    docker run -d --name test-overlay alpine sleep 3600

    # 查看 Docker 为这个容器配置的 OverlayFS 目录
    docker inspect test-overlay | grep -i "dir"
    ```
    您会看到类似下面的输出，这正是 OverlayFS 的关键路径：
    ```json
    "LowerDir": "/var/lib/docker/overlay2/l/…",
    "UpperDir": "/var/lib/docker/overlay2/…/diff",
    "MergedDir": "/var/lib/docker/overlay2/…/merged",
    "WorkDir": "/var/lib/docker/overlay2/…/work"
    ```
    *   `LowerDir` 指向只读层链。
    *   `UpperDir` 对应容器可写层，路径通常为 `.../{id}/diff`。
    *   `MergedDir` 是容器看到的统一文件系统，路径通常为 `.../{id}/merged`。
    *   `WorkDir` 是内部工作目录，路径通常为 `.../{id}/work`。

2.  **验证写时复制 (Copy-on-Write)**
    *   **在容器内创建一个新文件：**
        ```bash
        docker exec test-overlay touch /app/new_file.txt
        ```
        此时查看主机的 `UpperDir` 路径，您会在其下的 `/app` 目录中看到 `new_file.txt`。
        ```bash
        UPPER_DIR=$(docker inspect test-overlay | jq -r '.[0].GraphDriver.Data.UpperDir')
        sudo ls $UPPER_DIR/app
        # new_file.txt
        ```

    *   **在容器内修改一个基础镜像已有的文件：**
        ```bash
        docker exec test-overlay sh -c "echo 'hello overlay' >> /etc/hostname"
        ```
        再次查看主机的 `UpperDir`，您会发现 `etc/hostname` 文件被复制了上来，并且包含了修改后的内容。
        ```bash
        sudo cat $UPPER_DIR/etc/hostname
        # <容器短ID>
        # hello overlay
        ```
        这清晰地展示了 CoW 过程：原始文件从 `LowerDir` 被复制到 `UpperDir` 进行修改，底层保持不变。

---

## Part 3: 运行时对比：Docker vs. Containerd 的分层管理

`containerd` 作为更底层的容器运行时，其组织方式与传统 Docker Engine 有所不同，概念更清晰，也更符合云原生标准。

### 3.1. Docker (`overlay2` 驱动) 的层级组织

当使用 `dockerd` 时，所有数据都集中在 `/var/lib/docker/overlay2`。

-   **物理结构**:
    -   **`{hash-id}`**: 代表一个层的目录，其下的 `diff` 目录存放实际文件。
    -   **`l/`**: 存放短名称符号链接，指向 `{hash-id}` 目录，以避免路径过长问题。
    -   **`lower` 文件**: 在上层目录中，记录其下层（`lowerdir`）的链接ID。
-   **核心理念**: 层级关系主要通过文件系统自身结构（`lower` 文件）来表达。元数据与存储耦合较深。

### 3.2. Containerd (`overlayfs` 快照器) 的深度解析

`containerd` 使用“快照 (snapshot)”模型来管理层，设计上更为先进和解耦。

-   **物理目录结构**:
    -   **内容存储 (Content Store)**: `/var/lib/containerd/io.containerd.content.v1.content/blobs/`
        -   这里存放着镜像的真正原始数据——**内容寻址的 `blobs`**（通常是 tar.gz 压缩包），以其 SHA256 哈希值命名。一个 blob 就对应一个镜像层。
    -   **快照存储 (Snapshot Store)**: `/var/lib/containerd/io.containerd.snapshotter.v1.overlayfs/snapshots/`
        -   **`{number}`**: 每个数字命名的目录代表一个**快照**。快照是 **blob 内容解压后的文件系统树**。
        -   **`fs/`**: 在每个快照目录中，`fs` 目录存放着该快照的实际文件内容（相当于 Docker 的 `diff` 目录）。
        -   **`parents`**: 在上层快照目录中，此文件记录其父级快照的数字 ID。

-   **元数据管理 (The Source of Truth)**:
    -   所有关于镜像、内容（blobs）和快照之间关系的**元数据**都存储在一个独立的 BoltDB 数据库中：`/var/lib/containerd/io.containerd.metadata.v1.bolt/meta.db`。
    -   这个数据库是所有操作的“唯一事实来源”。它精确记录了哪个镜像由哪些 blob 组成，以及这些 blob 被解压成了哪些快照。这种**元数据与存储分离**的设计，比 Docker 依赖文件系统来推断关系的方式更加健壮和清晰。

-   **快照的生命周期与状态**:
    `containerd` 中的快照有明确的状态，反映其用途：
    1.  **`PREPARED`**: 一个临时的、可写的快照。当需要进行 copy-up 等操作时，会先创建一个 `PREPARED` 快照作为中转。
    2.  **`COMMITTED`**: 一个只读的、已固化的快照。**镜像的每一层都对应一个 `COMMITTED` 快照**。它们像积木一样堆叠起来，构成完整的镜像。
    3.  **`ACTIVE`**: 一个可写的快照，用作**正在运行的容器的可写层**。它总是以一个 `COMMITTED` 快照（镜像的顶层）作为其父级。

-   **ImageFS vs. ContainerFS 的清晰分离**:
    `containerd` 的模型天然地实现了 `imagefs` 和 `containerfs` 的概念分离，这对于 Kubernetes 等编排系统至关重要。
    -   **ImageFS**: 指存放只读 `COMMITTED` 快照的文件系统。这是镜像数据所在的地方。
    -   **ContainerFS**: 指为每个容器创建的 `ACTIVE` 快照及其父级 `COMMITTED` 快照链所组成的统一文件系统。
    -   **优势**: 这种分离允许进行更精细化的管理和安全设置。例如，可以将 `ImageFS` 挂载在独立的、只读的磁盘上，以防止任何对基础镜像的意外篡固。

### 3.3. 查询与对比总结

使用 `ctr` 工具可以查询 `containerd` 的内部信息 (Kubernetes 环境通常使用 `k8s.io` 命名空间)。

- **追踪示例**:
  ```bash
  # 1. 列出镜像，找到 busybox
  sudo ctr -n k8s.io images ls | grep busybox

  # 2. 查看镜像的 manifest，获取其各层内容的 digest (哈希)
  IMAGE="docker.io/library/busybox:latest"
  sudo ctr -n k8s.io images info $IMAGE | jq '.manifest.layers[].digest'

  # 3. 列出所有快照，观察 Parent -> ID 的链条关系
  # COMMITTED 快照代表镜像层，ACTIVE 快照代表容器层
  sudo ctr -n k8s.io snapshots ls
  ```

| 特性 | Docker (`overlay2`) | Containerd (`overlayfs` snapshotter) |
| :--- | :--- | :--- |
| **核心理念** | 基于文件系统的关系表达 | **基于独立元数据的关系管理** |
| **元数据管理** | 耦合于文件系统结构 (`lower` 文件) | **独立的 BoltDB 数据库 (`meta.db`)** |
| **存储分离** | 混合存储，概念上不清晰 | **清晰的 `content` 和 `snapshot` 分离** |
| **镜像/容器隔离** | 混合在同一目录 | **清晰的快照模型 (Committed vs. Active)** |
| **生态对接** | 传统 Docker 模式 | **云原生标准 (CRI)**, 被 K8s 广泛采用 |
| **查询工具** | `docker inspect` | `ctr` (snapshots, images, content) |

**结论**: `containerd` 的方式是更现代、更健壮、更符合云原生场景的设计。它通过清晰的元数据管理和快照模型，解决了 Docker 早期设计中一些模糊和耦合过紧的问题。

---

## Part 4: 实践与最佳策略

### 4.1. 查看镜像分层

-   **`docker image history <image_name>`**: 清晰地展示镜像是如何由 `Dockerfile` 中的指令一步步构建起来的，以及每一层的大小。
-   **`docker image inspect <image_name>`**: 提供关于镜像的全部元数据，包括每一层的 SHA256 哈希值 (`RootFS.Layers`)。

### 4.2. 优化镜像大小和构建速度

-   **合并指令以减少层数**：将多个 `RUN` 命令用 `&&` 合并在一条指令中，并及时清理缓存，以减少层数。
    ```dockerfile
    # 推荐
    RUN apt-get update && apt-get install -y curl \
        && rm -rf /var/lib/apt/lists/*
    ```

-   **合理安排 Dockerfile 指令顺序**：将不经常变动的依赖项（如 `npm install`）放在前面，将经常变动的源码 (`COPY . .`) 放在后面，以最大化利用构建缓存。
    ```dockerfile
    # 推荐
    COPY package*.json ./
    RUN npm install
    COPY . .
    ```

-   **使用 `.dockerignore` 文件**：忽略不需要打包进镜像的文件（如 `.git`, `node_modules`, `README.md`），以减小构建上下文和镜像体积。

-   **使用多阶段构建 (Multi-stage Builds)**：对于编译型语言尤其有效。在一个 `build` 阶段编译出二进制文件，然后在最终阶段使用一个极简的基础镜像，只复制必要的产物。
    ```dockerfile
    # ---- Build Stage ----
    FROM golang:1.19 AS build
    WORKDIR /src
    COPY . .
    RUN CGO_ENABLED=0 go build -o /app .

    # ---- Final Stage ----
    FROM alpine:latest
    COPY --from=build /app /app
    CMD ["/app"]
    ```

### 4.3. 共享与效率

-   **磁盘空间**：多个镜像如果基于相同的父镜像，它们将在磁盘上共享这些基础层。
-   **网络传输**：拉取（pull）或推送（push）镜像时，已存在的层不会被重复传输，节省网络带宽和时间。
