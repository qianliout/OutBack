# Docker 

### 一、概念与基础

1.  **什么是 Docker？它与传统虚拟机（VM）的主要区别是什么？**

    *   **Docker** 是一个开源的应用容器引擎，它允许开发者将应用及其依赖打包到一个轻量级、可移植的容器中，然后发布到任何流行的 Linux 或 Windows 机器上，也可以实现虚拟化。容器是完全使用沙箱机制，相互之间不会有任何接口。

    *   **与传统虚拟机（VM）的主要区别**：
        *   **架构层级**：VM 在宿主机操作系统之上构建了一个完整的虚拟硬件层（Hypervisor），并在其上运行一个全新的客户机操作系统（Guest OS）。而 Docker 容器直接运行在宿主机操作系统之上，共享宿主机的内核。
        *   **资源占用**：由于不需要额外的客户机操作系统，Docker 容器比 VM 更轻量，启动更快（秒级 vs 分钟级），占用的系统资源（CPU、内存、磁盘空间）也更少。
        *   **性能**：Docker 容器的性能接近原生，因为它们直接利用宿主机的内核。而 VM 由于增加了虚拟硬件和客户机操作系统层，会有一定的性能损耗。
        *   **隔离性**：VM 提供了更强的隔离性，因为它们有独立的内核。Docker 容器共享宿主机内核，隔离性相对较弱，主要依赖于 Linux 的 Namespace 和 Cgroups 技术。
        *   **可移植性**：Docker 容器在不同的环境中具有极高的一致性，解决了“在我电脑上能跑”的问题。VM 也可以移植，但由于体积庞大，移植和分发不如 Docker 方便。

2.  **请解释 Docker 的架构，包括其主要组件（Docker Client, Docker Daemon, Docker Registry等）。**

    Docker 采用 C/S (Client/Server) 架构。

    *   **Docker Client (客户端)**：用户与 Docker 交互的命令行工具，如 `docker build`, `docker run` 等命令都是通过客户端发送给 Docker Daemon 的。
    *   **Docker Daemon (守护进程)**：也称为 Docker Engine，是 Docker 的核心服务，运行在宿主机上。它负责监听并处理来自客户端的 API 请求，管理 Docker 对象，如镜像、容器、网络和数据卷。
    *   **Docker Registry (镜像仓库)**：用于存储和分发 Docker 镜像的地方。最著名的公共仓库是 Docker Hub。用户可以从仓库拉取（pull）镜像，也可以将自己构建的镜像推送（push）到仓库。

3.  **什么是 Docker 镜像（Image）和容器（Container）？它们之间的关系是什么？**

    *   **Docker 镜像 (Image)**：一个只读的模板，包含了创建 Docker 容器所需的文件系统和配置。它由一系列的层（Layers）组成，每一层代表了 Dockerfile 中的一条指令。镜像可以用来创建容器。
    *   **Docker 容器 (Container)**：镜像的运行实例。容器是可读写的，它在镜像的只读层之上增加了一个可写层（容器层）。容器包含了应用程序及其所有依赖，与宿主机和其他容器隔离。

    *   **关系**：可以把镜像看作是面向对象编程中的 **类（Class）**，而容器则是这个类的 **实例（Instance）**。一个镜像可以创建出多个相互隔离的容器。

4.  **Dockerfile 的主要作用是什么？请列举几个常用的 Dockerfile 指令。**

    *   **作用**：Dockerfile 是一个文本文件，包含了一系列用户可以调用来自动构建 Docker 镜像的指令。它定义了环境、应用程序以及如何运行它。

    *   **常用指令**：
        *   `FROM`：指定基础镜像，必须是第一条指令。
        *   `RUN`：在镜像构建过程中执行命令（如安装软件、更新包）。
        *   `COPY` / `ADD`：将文件或目录从宿主机复制到镜像中。`ADD` 功能更强，可以处理 URL 和解压 tar 文件。
        *   `WORKDIR`：设置工作目录，后续的 `RUN`, `CMD`, `ENTRYPOINT` 等指令都在此目录下执行。
        *   `EXPOSE`：声明容器在运行时监听的端口。
        *   `CMD`：提供容器启动时的默认命令。如果 `docker run` 提供了命令，`CMD` 会被覆盖。
        *   `ENTRYPOINT`：配置容器启动时执行的命令，不会被 `docker run` 的参数轻易覆盖，而是将 `docker run` 的参数作为 `ENTRYPOINT` 的参数。
        *   `ENV`：设置环境变量。

5.  **什么是联合文件系统（UnionFS）？它在 Docker 中起什么作用？**

    *   **联合文件系统 (UnionFS)** 是一种分层、轻量级并且高性能的文件系统，它支持将不同目录（分支）的内容联合挂载（union mount）到一个统一的视图中。

    *   **在 Docker 中的作用**：Docker 使用 UnionFS 作为其存储驱动（如 AUFS, OverlayFS），来实现镜像和容器的分层。
        *   **分层存储**：Docker 镜像由多个只读层构成。当基于一个镜像创建新镜像时，只需在原有层之上添加新的层，实现了层的复用，节省了磁盘空间。
        *   **写时复制 (Copy-on-Write)**：当容器启动时，它在只读的镜像层之上添加一个可写的容器层。当容器需要修改一个文件时，该文件会从下方的只读层复制到可写层进行修改，而原始文件保持不变。这使得容器的创建和销毁非常快速，并且多个容器可以共享同一个基础镜像而不会相互影响。

### 二、镜像与容器

1.  **如何构建一个 Docker 镜像？（请说出命令并解释过程）**

    *   **命令**：`docker build -t <image_name>:<tag> <path_to_dockerfile_directory>`

    *   **过程**：
        1.  **准备 Dockerfile**：在项目根目录下创建一个名为 `Dockerfile` 的文件，其中包含构建镜像所需的指令。
        2.  **执行构建命令**：在包含 `Dockerfile` 的目录（或指定其路径）下运行 `docker build` 命令。
        3.  **Docker Daemon 执行构建**：
            *   客户端将构建上下文（通常是 `Dockerfile` 所在的目录及其所有文件）发送给 Docker Daemon。
            *   Daemon 逐行解析 `Dockerfile` 中的指令。
            *   对于每条指令，Daemon 会执行相应的操作，并生成一个新的镜像层。
            *   如果某一层已经存在于本地缓存中（例如，之前构建时生成的），Docker 会直接使用缓存，从而加快构建速度。
            *   所有指令执行完毕后，会生成一个最终的镜像，并使用 `-t` 参数指定的名称和标签进行标记。

2.  **`docker build` 命令中的 `-t` 参数是什么意思？**

    `-t` 是 `--tag` 的缩写，用于为构建的镜像指定一个 **名称（name）** 和可选的 **标签（tag）**。格式通常是 `repository/image_name:tag`。例如：`my-app:1.0`。这使得镜像更易于识别和管理。

3.  **`docker run` 和 `docker start` 命令有什么区别？**

    *   `docker run`：用于 **创建并启动** 一个 **新** 的容器。它会基于指定的镜像创建一个容器，然后启动它。
    *   `docker start`：用于 **启动** 一个 **已经存在但已停止** 的容器。它不会创建新容器，只是将一个处于停止状态的容器重新运行起来。

4.  **如何进入一个正在运行的 Docker 容器？（至少说出两种方法）**

    *   **方法一：`docker exec`** (推荐)
        *   命令：`docker exec -it <container_id_or_name> /bin/bash`
        *   说明：这是最常用的方法。它会在正在运行的容器中启动一个新的进程（如此处的 `/bin/bash`），并创建一个交互式的 TTY 终端。退出该终端不会导致容器停止。

    *   **方法二：`docker attach`**
        *   命令：`docker attach <container_id_or_name>`
        *   说明：此命令会连接到容器的主进程（PID 1）的标准输入、输出和错误流。**注意**：如果从这个终端退出，会导致容器的主进程结束，从而使容器停止运行。因此，它不适合用于简单的交互式 shell 访问。

5.  **如何查看正在运行的容器和所有容器（包括已停止的）？**

    *   **查看正在运行的容器**：`docker ps` 或 `docker container ls`
    *   **查看所有容器**：`docker ps -a` 或 `docker container ls -a`

6.  **如何删除一个容器和镜像？强制删除的命令是什么？**

    *   **删除容器**：
        *   普通删除（容器需先停止）：`docker rm <container_id_or_name>`
        *   强制删除（可删除正在运行的容器）：`docker rm -f <container_id_or_name>`

    *   **删除镜像**：
        *   普通删除（需先删除所有基于该镜像的容器）：`docker rmi <image_id_or_name>`
        *   强制删除（可删除被容器引用的镜像）：`docker rmi -f <image_id_or_name>`

7.  **如何查看容器的日志？（`docker logs` 命令及其常用选项）**

    *   **基本命令**：`docker logs <container_id_or_name>`

    *   **常用选项**：
        *   `-f` 或 `--follow`：实时跟踪日志输出，类似于 `tail -f`。
        *   `--tail <number>`：只显示最后 N 行日志。
        *   `-t` 或 `--timestamps`：为每条日志添加时间戳。
        *   `--since <timestamp>`：显示指定时间戳之后的日志（如 `--since 2023-10-27T10:00:00`）。
        *   `--until <timestamp>`：显示指定时间戳之前的日志。

8.  **当运行一个容器时，`-d` 和 `-it` 参数分别代表什么含义？**

    *   `-d` (`--detach`)：**后台运行**。容器将在后台启动并运行，同时打印出容器 ID。这对于长时间运行的服务（如 web 服务器）非常有用。
    *   `-it`：这是两个参数的组合：
        *   `-i` (`--interactive`)：保持标准输入（STDIN）打开，允许与容器进行交互。
        *   `-t` (`--tty`)：分配一个伪终端（pseudo-TTY），通常与 `-i` 结合使用，提供一个交互式的 shell 环境。
        *   合在一起，`-it` 用于启动一个需要用户交互的容器，例如进入一个 shell。

9.  **如何将本地文件复制到正在运行的容器中，或从容器中复制文件到本地？**

    使用 `docker cp` 命令。

    *   **从本地复制到容器**：
        `docker cp <local_path> <container_id_or_name>:<container_path>`
        *   示例：`docker cp ./app.conf my-container:/etc/app.conf`

    *   **从容器复制到本地**：
        `docker cp <container_id_or_name>:<container_path> <local_path>`
        *   示例：`docker cp my-container:/var/log/app.log ./logs/`

10. **什么是 `.dockerignore` 文件？它有什么作用？**

    *   `.dockerignore` 文件是一个文本文件，其语法类似于 `.gitignore`。
    *   **作用**：在执行 `docker build` 时，构建上下文（build context）会被发送到 Docker Daemon。`.dockerignore` 文件用于 **排除** 那些不需要包含在构建上下文中的文件和目录。
    *   **好处**：
        *   **减少镜像大小**：避免将不必要的文件（如 `.git` 目录、日志文件、本地依赖 `node_modules` 等）打包进镜像。
        *   **加快构建速度**：减少发送到 Daemon 的数据量。
        *   **避免缓存失效**：如果一些频繁变动但与构建无关的文件被包含进来，可能会导致 Docker 缓存失效，从而减慢构建。
        *   **安全性**：防止将敏感文件（如密钥、密码文件）意外地复制到镜像中。

### 三、网络与存储

1.  **Docker 有哪几种网络模式？请简要说明它们的特点（none, host, bridge, container）。**

    *   **`bridge` (桥接模式，默认)**：Docker 会创建一个名为 `docker0` 的虚拟网桥。每个容器都会被分配一个独立的网络命名空间和一个 IP 地址，并通过 `docker0` 网桥与宿主机和其他容器通信。这是最常用的模式。
    *   **`host` (主机模式)**：容器不会创建自己的网络命名空间，而是直接共享宿主机的网络栈。容器的 IP 地址就是宿主机的 IP 地址，容器内监听的端口也直接暴露在宿主机上。性能最好，但隔离性差。
    *   **`container` (容器模式)**：新创建的容器共享另一个已经存在的容器的网络命名空间（IP、端口等）。它们之间可以通过 `localhost` 直接通信。
    *   **`none` (无网络模式)**：容器拥有自己的网络命名空间，但不进行任何网络配置。它只有一个 `lo` (loopback) 网络接口，与外部完全隔离。

2.  **如何创建一个自定义的 Docker 网络？**

    使用 `docker network create` 命令。最常用的是创建一个自定义的 `bridge` 网络。

    `docker network create --driver bridge my-custom-network`

3.  **如何让两个容器通过自定义网络进行通信？**

    1.  **创建自定义网络**：
        `docker network create my-app-net`
    2.  **启动容器并连接到该网络**：
        `docker run -d --name service-a --network my-app-net my-service-a-image`
        `docker run -d --name service-b --network my-app-net my-service-b-image`
    3.  **通信**：在 `service-a` 容器内，可以直接通过容器名 `service-b` 来访问它，Docker 内置的 DNS 会自动解析。例如 `ping service-b`。

4.  **如何将宿主机的端口映射到容器内的端口？（`-p` 参数的使用）**

    使用 `docker run` 命令的 `-p` 或 `--publish` 参数。

    *   **格式**：`-p <host_port>:<container_port>`
    *   **示例**：`docker run -d -p 8080:80 nginx`
        *   这会将宿主机的 `8080` 端口映射到容器的 `80` 端口。访问宿主机的 `http://localhost:8080` 就会访问到 Nginx 容器的 `80` 端口。

    *   **其他用法**：
        *   ` -p <container_port>`：随机映射一个宿主机端口到容器指定端口。
        *   ` -p <ip>:<host_port>:<container_port>`：指定绑定到宿主机的特定 IP 地址。

5.  **Docker 的数据持久化有哪几种方式？请简要说明。**

    主要有三种方式：

    *   **数据卷 (Volumes)**：**推荐方式**。数据卷是由 Docker 管理的、存储在宿主机文件系统特定部分（如 `/var/lib/docker/volumes/`）的目录。它完全由 Docker 控制，与容器的生命周期解耦。
    *   **绑定挂载 (Bind Mounts)**：将宿主机上的任意文件或目录直接挂载到容器中。路径由用户完全控制，但可能涉及权限问题，且与宿主机的特定文件结构紧密耦合。
    *   **临时文件系统挂载 (tmpfs Mounts)**：将数据存储在宿主机的内存中，不会持久化到磁盘。适用于存储临时、敏感且不需要持久化的数据。

6.  **什么是 Docker 数据卷（Volume）？它与绑定挂载（bind mount）有什么区别？**

    *   **数据卷 (Volume)** 是 Docker 用于持久化数据的首选机制。

    *   **区别**：
        *   **管理方式**：
            *   **Volume**：由 Docker 创建和管理，与宿主机的文件系统结构解耦。用户只需关心卷的名称。
            *   **Bind Mount**：用户必须指定宿主机上的确切路径，与宿主机文件系统紧密耦合。
        *   **可移植性**：
            *   **Volume**：更具可移植性。因为 Dockerfile 或 Compose 文件中只定义卷的名称，不关心它在宿主机上的具体位置，这使得应用在不同环境中的部署更加一致。
            *   **Bind Mount**：可移植性差，因为它依赖于宿主机上存在的特定目录结构。
        *   **性能**：在某些操作系统（如 macOS 和 Windows）上，Volume 的性能通常优于 Bind Mount，因为 Docker 可以对其进行优化。
        *   **自动填充**：如果将一个 Volume 挂载到一个非空目录的容器中，Volume 会被预先填充该目录的内容。而 Bind Mount 则会覆盖容器内的目录。
        *   **安全性**：Volume 只能由 Docker 进程访问，相对更安全。Bind Mount 允许容器访问宿主机的任意文件系统，如果配置不当，可能带来安全风险。

7.  **如何创建一个数据卷并将其挂载到容器中？**

    *   **方法一：`docker run` 时自动创建**
        使用 `-v` 或 `--mount` 参数。如果指定的卷不存在，Docker 会自动创建它。
        `docker run -d --name my-container -v my-data-volume:/path/in/container my-image`
        这里 `my-data-volume` 就是数据卷的名称。

    *   **方法二：先创建数据卷，再挂载**
        1.  **创建数据卷**：`docker volume create my-data-volume`
        2.  **挂载到容器**：`docker run -d --name my-container -v my-data-volume:/path/in/container my-image`

    *   **使用 `--mount` 参数 (更明确)**：
        `docker run -d --name my-container --mount source=my-data-volume,target=/path/in/container my-image`

### 四、编排与集群 (Docker Compose & Swarm/Kubernetes)

1.  **什么是 Docker Compose？它的主要用途是什么？**

    *   **Docker Compose** 是一个用于定义和运行多容器 Docker 应用程序的工具。
    *   **主要用途**：通过一个单独的 `docker-compose.yml` 文件来配置应用服务，然后使用一条命令 (`docker-compose up`) 就可以创建并启动所有服务。它极大地简化了开发、测试和单机环境下的多服务部署流程。

2.  **请解释 Docker Compose 文件的基本结构（version, services, volumes, networks等）。**

    一个 `docker-compose.yml` 文件通常包含以下顶级键：

    *   `version`：指定 Compose 文件格式的版本（如 `"3.8"`）。虽然在新版中已非必需，但仍是好的实践。
    *   `services`：核心部分，定义了应用包含的各个服务（容器）。
        *   每个服务都是一个子键，如 `web`, `db`。
        *   在每个服务下，可以定义 `image`, `build`, `ports`, `volumes`, `networks`, `environment` 等，这些配置与 `docker run` 的参数类似。
    *   `volumes`：定义具名数据卷，供 `services` 引用。这使得数据卷的管理更加清晰。
    *   `networks`：定义自定义网络，供 `services` 连接。Compose 默认会创建一个桥接网络，但也可以在这里定义更复杂的网络拓扑。

3.  **如何通过 Docker Compose 一键部署和管理多个服务？**

    1.  **创建 `docker-compose.yml` 文件**：在项目根目录定义好所有服务、网络和数据卷。
    2.  **启动所有服务**：
        *   `docker-compose up`：在前台启动所有服务，并聚合显示所有容器的日志。
        *   `docker-compose up -d`：在后台启动所有服务。
    3.  **管理服务**：
        *   `docker-compose ps`：查看所有服务的状态。
        *   `docker-compose logs -f <service_name>`：查看特定服务的日志。
        *   `docker-compose stop`：停止所有服务，但不删除容器。
        *   `docker-compose start`：重新启动已停止的服务。
        *   `docker-compose down`：停止并 **删除** 所有服务的容器、网络。如果加上 `-v`，还会删除数据卷。
        *   `docker-compose build`：重新构建服务的镜像。
        *   `docker-compose exec <service_name> <command>`：在指定服务容器内执行命令。

4.  **什么是 Docker Swarm？它与 Kubernetes 相比有什么特点和优缺点？（基础问题）**

    *   **Docker Swarm** 是 Docker 官方提供的容器编排工具，它将多个 Docker 主机组成一个集群，让用户可以像管理单个 Docker 主机一样管理整个集群。

    *   **与 Kubernetes (K8s) 的比较**：
        *   **特点和优点 (Swarm)**：
            *   **简单易用**：与 Docker Engine 紧密集成，学习曲线平缓，配置和管理非常简单。
            *   **轻量级**：对资源要求较低，部署快速。
            *   **开箱即用**：作为 Docker 的一部分，无需额外安装复杂的组件。
        *   **缺点 (Swarm)**：
            *   **功能相对有限**：相比 K8s，Swarm 在自动扩缩容、服务发现、存储编排、自我修复等高级功能上较弱。
            *   **社区和生态系统较小**：K8s 已经成为事实上的行业标准，拥有庞大的社区和丰富的生态工具。
            *   **可定制性差**：Swarm 的功能比较固定，不如 K8s 灵活和可扩展。

        *   **总结**：Swarm 适合中小型、对编排需求不复杂的应用。Kubernetes 则适用于大型、复杂、需要高度自动化和可扩展性的生产环境。

5.  **如何在 Swarm 中创建一个服务（service）？**

    1.  **初始化 Swarm 集群**（在一个 manager 节点上执行）：
        `docker swarm init`
    2.  **创建服务**：
        使用 `docker service create` 命令。
        *   **示例**：创建一个名为 `my-web` 的 Nginx 服务，包含 3 个副本，并将宿主机的 8080 端口映射到服务的 80 端口。
          `docker service create --name my-web --replicas 3 -p 8080:80 nginx`

    *   **管理服务**：
        *   `docker service ls`：列出所有服务。
        *   `docker service ps my-web`：查看服务的任务（容器）分布情况。
        *   `docker service scale my-web=5`：将服务副本数扩展到 5 个。
        *   `docker service rm my-web`：删除服务。

### 五、安全与最佳实践

1.  **在 Dockerfile 中编写指令时，有哪些最佳实践可以减少镜像大小和提高安全性？**

    *   **使用 `.dockerignore`**：排除不必要的文件，从源头上减少构建上下文的大小。
    *   **选择合适的基础镜像**：使用官方、经过验证的轻量级基础镜像，如 `alpine`、`slim` 版本，避免使用庞大且包含不必要工具的镜像。
    *   **多阶段构建 (Multi-stage builds)**：利用多个 `FROM` 指令，将构建环境（如包含编译器、SDK 的大镜像）与最终的运行时环境（只包含必要依赖和产物的小镜像）分离。这是减小镜像大小最有效的方法之一。
    *   **合并 `RUN` 指令**：将多个 `RUN` 指令用 `&&` 连接起来，并清理缓存。因为每条 `RUN` 指令都会创建一个新的镜像层，合并它们可以减少层数。
        *   示例：`RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/*`
    *   **最小化安装**：在 `RUN` 指令中只安装必要的软件包，并使用 `--no-install-recommends` 等选项避免安装非必需的依赖。
    *   **以非 root 用户运行**：见下一题。
    *   **不要存储敏感数据**：避免在 Dockerfile 中硬编码密码、密钥等。使用构建时参数 (`ARG`)、环境变量 (`ENV`) 或更好的方式——使用 Docker secrets 或其他 secrets 管理工具。
    *   **利用层缓存**：将不经常变动的指令（如安装依赖）放在前面，将经常变动的指令（如 `COPY` 源代码）放在后面，以充分利用 Docker 的构建缓存。

2.  **为什么不建议在 Docker 容器中以 root 用户身份运行应用程序？如何避免？**

    *   **原因**：
        *   **最小权限原则**：如果容器内的应用以 root 身份运行，一旦应用被攻破，攻击者就获得了容器内的 root 权限。
        *   **容器逃逸风险**：虽然 Docker 有多层安全机制，但如果内核存在漏洞，容器内的 root 用户可能利用这些漏洞获得宿主机的 root 权限，造成严重的安全问题。

    *   **如何避免**：
        1.  **在 Dockerfile 中创建非 root 用户和用户组**：
            `RUN groupadd -r myuser && useradd -r -g myuser myuser`
        2.  **切换到该用户**：
            使用 `USER` 指令切换到新创建的非 root 用户。
            `USER myuser`
        3.  **确保文件权限**：确保应用程序文件和目录对于该非 root 用户是可读/可执行的。可能需要使用 `COPY --chown=myuser:myuser` 或 `RUN chown -R myuser:myuser /app`。

3.  **如何限制容器可使用的内存和CPU资源？**

    在 `docker run` 命令中使用以下参数：

    *   **限制内存**：
        *   `--memory` 或 `-m`：设置内存使用上限。例如 `-m 512m`。
        *   `--memory-swap`：设置内存+交换分区的总上限。

    *   **限制 CPU**：
        *   `--cpus`：设置可使用的 CPU 核心数。例如 `--cpus="1.5"` 表示最多使用 1.5 个 CPU 核心。
        *   `--cpu-shares`：设置 CPU 使用的相对权重（默认为 1024）。当多个容器争用 CPU 资源时，权重越高的容器会获得更多的 CPU 时间。

    在 Docker Compose 中，可以在 `deploy.resources.limits` 下配置这些限制。

4.  **Docker 镜像的漏洞扫描是什么？如何进行？**

    *   **是什么**：镜像漏洞扫描是指使用工具分析 Docker 镜像的各个层，识别其中包含的操作系统包和应用程序依赖是否存在已知的安全漏洞（CVEs）。

    *   **如何进行**：
        *   **`docker scan` 命令**：Docker Desktop 和 Docker Hub 集成了 Snyk 提供的漏洞扫描功能。可以直接使用 `docker scan <image_name>` 命令来扫描本地镜像。
        *   **第三方工具**：
            *   **Trivy**：一个非常流行、简单易用的开源漏洞扫描器。
            *   **Clair**：CoreOS 开发的开源容器镜像安全分析器。
            *   **Grype**：Anchore 开发的开源漏洞扫描工具。
        *   **集成到 CI/CD 流水线**：最佳实践是将漏洞扫描集成到持续集成/持续部署（CI/CD）流程中。在镜像构建完成后、推送到仓库之前自动进行扫描，如果发现高危漏洞，则中断流程。

### 六、场景与故障排查

1.  **如何排查一个无法启动的 Docker 容器？（查看日志、交互式启动等）**

    1.  **查看容器日志**：这是第一步。即使容器快速退出，它的日志也可能记录了错误信息。
        `docker logs <container_id>` (使用 `docker ps -a` 找到失败容器的 ID)
    2.  **检查 `docker inspect`**：查看容器的详细配置和状态，特别是 `State` 部分，可能会有错误信息。
        `docker inspect <container_id>`
    3.  **检查 Dockerfile 和入口点**：确认 `CMD` 或 `ENTRYPOINT` 指令是否正确，执行的命令或脚本是否存在且有执行权限。
    4.  **以交互模式启动进行调试**：覆盖原有的 `ENTRYPOINT` 或 `CMD`，进入容器的 shell 环境进行手动排查。
        `docker run -it --entrypoint /bin/sh <image_name>`
        进入后，可以手动执行启动脚本，检查文件路径、权限、环境变量等是否正确。
    5.  **检查端口冲突**：如果日志显示端口已被占用，使用 `netstat -tuln | grep <port>` 或类似命令检查宿主机端口是否被其他进程占用。

2.  **容器内的应用程序无法连接外部网络（如互联网）或数据库，可能的原因有哪些？如何排查？**

    *   **可能的原因**：
        1.  **DNS 解析问题**：容器内无法解析域名。
        2.  **网络模式问题**：容器使用了 `none` 网络，或者自定义网络配置错误。
        3.  **防火墙/安全组规则**：宿主机或云平台的防火墙阻止了出站或入站连接。
        4.  **Docker 网络问题**：Docker 的 `iptables` 规则出现混乱或损坏。
        5.  **目标服务不可达**：数据库地址错误、端口不通，或者数据库本身没有运行。

    *   **排查步骤**：
        1.  **进入容器**：`docker exec -it <container_id> /bin/sh`
        2.  **测试 DNS**：`ping www.baidu.com`。如果 ping 不通域名但 `ping 8.8.8.8` 可以，说明是 DNS 问题。可以检查 `/etc/resolv.conf` 文件，或在 `docker run` 时使用 `--dns` 参数指定 DNS 服务器。
        3.  **测试网络连通性**：`ping <database_host_ip>` 或使用 `telnet <database_host> <port>` 检查到数据库的 IP 和端口是否通畅。
        4.  **检查宿主机网络**：在宿主机上尝试连接数据库，排除数据库自身的问题。
        5.  **检查防火墙**：检查宿主机的 `iptables` 或 `firewalld` 规则，以及云服务商的安全组设置。
        6.  **重启 Docker 服务**：在某些情况下，`iptables` 规则可能混乱，重启 Docker 服务 (`systemctl restart docker`) 可以重建规则。

3.  **宿主机的磁盘空间很快被占满，可能是什么原因造成的？（镜像、容器、卷、日志等）**

    *   **主要原因**：
        1.  **悬虚镜像 (Dangling Images)**：在构建过程中产生的没有标签的中间层镜像，它们不再被任何有效镜像引用，但仍占用空间。
        2.  **未使用的镜像**：下载或构建了大量镜像，但很多已不再使用。
        3.  **停止的容器**：创建了大量容器，停止后未删除，它们的可写层仍然占用磁盘空间。
        4.  **未使用的卷 (Volumes)**：容器被删除后，其关联的具名数据卷默认不会被删除。
        5.  **容器日志文件过大**：长时间运行的容器，如果日志没有被有效管理（如轮转），可能会产生巨大的日志文件。

4.  **如何清理 Docker 占用的磁盘空间（如删除悬虚镜像、停止的容器等）？**

    使用 `docker system prune` 命令族是最高效的方法。

    *   **清理所有未使用的对象（推荐）**：
        `docker system prune -a --volumes`
        *   `-a`：删除所有未使用的镜像（不仅仅是悬虚镜像）。
        *   `--volumes`：同时删除未被任何容器使用的具名数据卷。
        *   **此命令会删除所有已停止的容器、所有悬虚和未使用的镜像、所有未使用的网络和所有未使用的卷。操作前请务必确认！**

    *   **分步清理**：
        *   **删除所有已停止的容器**：`docker container prune` 或 `docker rm $(docker ps -aq)`
        *   **删除悬虚镜像**：`docker image prune` 或 `docker rmi $(docker images -f "dangling=true" -q)`
        *   **删除所有未使用的镜像**：`docker image prune -a`
        *   **删除未使用的卷**：`docker volume prune` 或 `docker volume rm $(docker volume ls -qf dangling=true)`
        *   **删除未使用的网络**：`docker network prune`

    *   **查看 Docker 磁盘使用情况**：
        `docker system df`

5.  **描述一下你曾经使用 Docker 部署一个完整应用的流程。**

    以部署一个包含 **Web 前端 (React)**、**后端 API (Node.js)** 和 **数据库 (PostgreSQL)** 的应用为例：

    1.  **项目结构规划**：
        *   项目根目录包含 `docker-compose.yml`。
        *   每个服务（`frontend`, `backend`）有自己的子目录，其中包含各自的 `Dockerfile`。

    2.  **编写 Dockerfile**：
        *   **后端 `backend/Dockerfile`**：
            *   使用多阶段构建。
            *   `FROM node:18-alpine as builder`：构建阶段，安装依赖、编译 TypeScript 等。
            *   `FROM node:18-alpine`：生产阶段，仅复制 `node_modules` 和编译后的 `dist` 目录，设置非 root 用户，并用 `CMD` 启动服务。
        *   **前端 `frontend/Dockerfile`**：
            *   同样使用多阶段构建。
            *   构建阶段 (`FROM node:18-alpine as builder`)：安装依赖、运行 `npm run build` 生成静态文件。
            *   生产阶段 (`FROM nginx:stable-alpine`)：将上个阶段生成的静态文件 (`build` 目录) 复制到 Nginx 的 `html` 目录。并复制一个自定义的 `nginx.conf` 配置文件来处理路由和反向代理。

    3.  **编写 `docker-compose.yml`**：
        *   **`version: "3.8"`**
        *   **`services`**:
            *   **`backend`**:
                *   `build: ./backend`
                *   `environment`: 从 `.env` 文件读取数据库连接信息。
                *   `volumes`: 将后端代码目录挂载，方便开发时热重载。
                *   `networks`: 连接到 `app-network`。
            *   **`frontend`**:
                *   `build: ./frontend`
                *   `ports`: 映射 `80:80`，对外提供服务。
                *   `depends_on`: `backend`，确保后端先启动。
            *   **`db`**:
                *   `image: postgres:14-alpine`
                *   `environment`: 设置数据库用户、密码。
                *   `volumes`: 创建一个具名卷 `db-data` 来持久化数据库数据。
                *   `networks`: 连接到 `app-network`。
        *   **`volumes`**:
            *   `db-data:`
        *   **`networks`**:
            *   `app-network:`

    4.  **部署与运行**：
        *   在服务器上，将代码克隆下来。
        *   创建 `.env` 文件并填入生产环境的配置。
        *   运行 `docker-compose up -d --build`。`--build` 确保会根据最新的代码构建镜像。
        *   使用 `docker-compose logs -f` 检查所有服务是否正常启动。
        *   通过服务器的 IP 地址或域名访问应用。

    5.  **维护与更新**：
        *   更新代码后，重新运行 `docker-compose up -d --build` 即可完成服务的滚动更新。
        *   使用 `docker system prune` 定期清理无用的 Docker 对象。
