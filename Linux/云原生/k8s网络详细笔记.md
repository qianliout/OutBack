# Kubernetes 网络深度解析笔记

欢迎来到 Kubernetes 网络的世界。这部分知识是 K8s 中最复杂但也是最核心的部分之一。理解它，你才能真正掌握你的集群。这份笔记将严格按照你制定的大纲，从基础到高级，从使用到原理，为你揭开 K8s 网络的神秘面纱。

## 第一部分：Kubernetes 网络模型基础

这部分是理解后续所有概念的地基。

### 1. Kubernetes 网络模型原则

Kubernetes 的网络模型设计的核心思想是 **“IP-per-Pod” (每个 Pod 一个 IP)**。它刻意选择了一种扁平化的、对应用透明的网络方案，避免了复杂性。为了实现这一点，它规定了所有网络实现都必须遵循以下四条基本原则：

*   **Pod IP 地址的唯一性**：集群中每一个 Pod 都拥有一个全局唯一的 IP 地址。这意味着你不需要关心端口映射（就像 Docker 中那样），简化了应用的配置和迁移。
*   **所有 Pod 之间可以直接通信（无需 NAT）**：任何一个 Pod 都通过对方的 Pod IP 直接访问到另一个 Pod，无论它们是否在同一个节点上。中间的网络设备不能进行网络地址转换（NAT）。
*   **所有 Node 与所有 Pod 之间可以直接通信（无需 NAT）**：任何一个节点（Node）都可以通过 Pod 的 IP 直接访问到集群中的任何一个 Pod。反之亦然。
*   **Pod 看到的自身 IP 和其他 Pod 看到的它的 IP 是同一个**：这消除了网络地址转换带来的复杂性。Pod 内部的应用通过 `ifconfig` 或 `ip addr` 看到的 IP，就是其他任何 Pod 或 Node 用来与它通信的 IP。

**底层原理**：这个模型的设计哲学是为了 **降低心智负担**。它让开发者可以像在传统物理机或虚拟机环境中一样思考网络，服务之间可以直接通过 IP 和端口通信，而不用关心底层的网络拓扑和复杂的端口映射关系。所有实现这些原则的魔法，都交给了下一节要讲的 CNI 插件。

### 2. 容器网络接口 (CNI - Container Network Interface)

**是什么**：CNI 是一套由 Cloud Native Computing Foundation (CNCF) 维护的规范和库，它定义了一个标准的接口，用于在 Linux 容器中配置网络。

*   **CNI 的角色和重要性**：Kubernetes 本身不负责实现网络功能。当 `kubelet`（运行在每个 Node 上的代理）创建或销毁一个 Pod 时，它会调用一个符合 CNI 规范的 **CNI 插件** 来为这个 Pod 配置网络。这种插件化的设计让 Kubernetes 可以灵活地接入各种不同的网络方案。
*   **CNI 规范简介**：CNI 规范非常简单，它只关心两件事：
    1.  `ADD`: 将一个容器添加到网络。插件必须为容器分配 IP，设置路由等。
    2.  `DEL`: 将一个容器从网络中移除。插件必须清理相关的网络资源。
    `kubelet` 通过调用二进制的 CNI 插件并传入 JSON 格式的配置来完成这些操作。
*   **常见的 CNI 插件概览**：
    *   **Flannel**: 最简单、最流行的 CNI 之一。它通常使用 **Overlay Network (覆盖网络)**，如 VXLAN，来创建一个跨越所有节点的虚拟网络层。易于设置，但性能有少量损失。
    *   **Calico**: 以高性能和强大的网络策略著称。它不使用覆盖网络，而是使用 **BGP 协议** 在节点间创建路由，让 Pod IP 可以在底层网络中直接路由。性能接近物理网络，但对底层网络环境有一定要求。
    *   **Cilium**: 一个现代化的 CNI，其核心是 **eBPF (extended Berkeley Packet Filter)** 技术。它直接在 Linux 内核中进行数据包操作，绕过了传统的 `iptables`，提供了极高的性能、强大的安全性和无与伦比的可观察性。
    *   **Weave Net**: 功能全面，开箱即用。它也使用覆盖网络，并且默认提供网络加密，在非受信网络环境中非常有用。

### 3. Pod 间通信

这是检验 CNI 插件是否正确工作的核心场景。

#### **同一节点 (Node) 上的 Pod 间通信**

**底层原理**：这个过程类似于将两台电脑用网线插到同一个交换机上。

1.  **虚拟以太网设备对 (veth pair)**：可以想象成一根虚拟的 "网线"。它总是成对出现，一端连接在 Pod 的网络命名空间 (Network Namespace) 内，表现为 `eth0`；另一端连接在宿主机的根网络命名空间中。
2.  **网桥 (Bridge)**：宿主机上会有一个虚拟网桥（通常叫 `cni0` 或 `docker0`）。可以把它看作一个虚拟的二层交换机。所有连接到该宿主机的 Pod 的 `veth pair` 的另一端都会被“插”到这个网桥上。
3.  **数据包流程分析**：
    *   Pod1 (IP: `10.1.1.2`) 向 Pod2 (IP: `10.1.1.3`) 发送数据包。
    *   数据包从 Pod1 的 `eth0` 出来，通过 `veth pair` 的一端到达宿主机上的另一端。
    *   这一端连接在 `cni0` 网桥上，数据包被转发到网桥。
    *   网桥知道 `10.1.1.3` 这个 MAC 地址连接在 Pod2 的 `veth pair` 接口上，于是将数据包从这个接口发出。
    *   数据包通过 Pod2 的 `veth pair` 到达其内部的 `eth0`，通信完成。
    *   整个过程都在内核中完成，非常高效，本质上是纯二层的交换。



#### **不同节点 (Node) 上的 Pod 间通信**

这是 Kubernetes 网络的核心和难点。

*   **覆盖网络 (Overlay Network) vs. 路由 (Routing) 模式**：
    *   **覆盖网络 (Flannel, Weave)**：它在现有的物理网络（Underlay）之上构建一个虚拟的逻辑网络。当 Node1 上的 Pod 要和 Node2 上的 Pod 通信时，CNI 插件会将原始数据包（源IP: Pod1_IP, 目标IP: Pod2_IP）进行 **封装**，加上一个新的包头（源IP: Node1_IP, 目标IP: Node2_IP），然后通过物理网络发送。Node2 收到后 **解封装**，再转发给 Pod2。
    *   **路由模式 (Calico)**：它不封装数据包。每个节点上的 CNI 代理会作为一个 BGP speaker，向网络中的其他节点或路由器宣告：“我负责 `10.1.1.0/24` 这个网段的路由”。这样，当 Node1 要发包给 `10.1.2.5`（在 Node2 上）时，它的路由表会明确告诉它，下一跳应该是 Node2 的 IP。数据包在网络中是原生路由的。

*   **VXLAN, IP-in-IP 等隧道技术原理**：
    *   **VXLAN (Virtual Extensible LAN)** 是最主流的覆盖网络技术。它将原始的二层以太网帧封装在 UDP 包中进行传输。VXLAN 头中有一个 VNI (VXLAN Network Identifier)，用于隔离不同的虚拟网络。Flannel 就是通过它实现的。
    *   **IP-in-IP** 是一种更简单的隧道技术，直接将原始的 IP 包封装在一个新的 IP 包里。

*   **BGP 协议在 Calico 中的应用简介**：
    *   BGP (Border Gateway Protocol) 是互联网的核心路由协议。Calico 巧妙地将它用在了数据中心内部。每个运行 Calico 的 Node 都是一个 BGP Peer，它们之间建立对等连接，相互学习和通告 Pod 的 IP 路由信息，从而构建了整个集群的 Pod 路由网络。

*   **数据包跨节点流程分析 (以 Flannel/VXLAN 为例)**：
    *   PodA (`10.1.1.2` on Node1) 向 PodB (`10.1.2.3` on Node2) 发送数据包。
    *   数据包离开 PodA，到达 Node1 上的 `cni0` 网桥。
    *   Node1 的路由表指示，`10.1.2.0/24` 网段应该通过一个名为 `flannel.1` 的虚拟设备发送。这个设备就是 VXLAN 的隧道端点 (VTEP)。
    *   `flannel.1` 设备收到原始数据包后，根据 Flannel 的控制平面信息（知道 PodB 在 Node2 上），将其用 VXLAN 协议封装。新的 UDP 包头为：源 IP `Node1_IP`，目标 IP `Node2_IP`。
    *   这个 UDP 包通过 Node1 的物理网卡 `eth0` 发送到物理网络。
    *   Node2 的物理网卡 `eth0` 收到这个 UDP 包。内核发现这是一个 VXLAN 包，将其解封装，得到原始的 Pod 数据包。
    *   解封装后的包被送到 Node2 的 `cni0` 网桥，最后通过 veth pair 送达 PodB。

## 第二部分：核心网络对象 (API Resources)

如果说 CNI 解决了“通”的问题，那么 Service、Ingress 等就解决了“用”的问题。

### 1. Service：Pod 的稳定访问入口

*   **Service 的必要性**：Pod 是“易逝”的。它们可能因为故障、伸缩、更新而销毁重建，每次重建 IP 地址都会改变。直接使用 Pod IP 会导致服务不可用。**Service 提供了一个稳定的、统一的访问入口**，屏蔽了后端 Pod 的动态变化和负载均衡。
*   **虚拟 IP (ClusterIP) 的概念**：`ClusterIP` 是 Service 的默认类型。它是一个 **虚拟 IP**，你无法在任何网络接口上找到它，也 `ping` 不通它。它只存在于每个节点上 `kube-proxy` 所维护的网络规则（如 `iptables` 或 `IPVS`）中，作为一个流量劫持的目标地址。
*   **Service 类型**：
    *   `ClusterIP`: 默认类型。只在集群内部可见。访问 `ClusterIP:Port` 的流量会被负载均衡到后端的 Pods。
    *   `NodePort`: 在 `ClusterIP` 的基础上，在 **集群中的每一个节点** 上都打开一个相同的端口。访问 `AnyNodeIP:NodePort` 的流量，会被转发到该 Service 的 `ClusterIP`，进而转发到 Pod。主要用于集群外部访问。
    *   `LoadBalancer`: 在 `NodePort` 的基础上，向底层云厂商（如 AWS, GCP, Azure）申请一个外部负载均衡器，该负载均衡器指向所有节点的 `NodePort`。这是将服务暴露给公网的标准方式。
    *   `ExternalName`: 一种特殊的 Service，它不做任何代理和转发，而是通过集群 DNS 返回一个 CNAME 记录，将一个服务名映射到另一个外部域名。
*   **`Endpoint` 与 `EndpointSlice` 对象的作用**：Service 通过 Label Selector 找到它应该代理的 Pods。但它如何知道这些 Pods 的具体 IP 地址呢？答案就是 `Endpoint` 或 `EndpointSlice`。当一个 Service 被创建时，Kubernetes 会自动创建一个同名的 `EndpointSlice` 对象，里面包含了所有健康的、与 Service 的 Label Selector 匹配的 Pod 的 IP 和端口列表。`kube-proxy` 正是监听 `EndpointSlice` 的变化来更新其转发规则的。（`EndpointSlice` 是 `Endpoint` 的优化版本，更具扩展性）。
*   **Headless Service (无头服务)**：如果在 Service 定义中设置 `clusterIP: None`，那么这个 Service 就不会被分配 ClusterIP。当在集群内部查询这个 Service 的 DNS 时，返回的将不再是单个虚拟 IP，而是 **所有后端 Pod 的 IP 地址列表**。这对于需要自己做服务发现的分布式应用（如数据库集群 Zookeeper, Kafka）非常有用，特别是与 `StatefulSet` 结合使用时。

### 2. Service 底层实现原理

这一切魔法的核心是运行在每个节点上的 **`kube-proxy`** 组件。

*   **kube-proxy 组件**：
    *   **职责**：`kube-proxy` 是一个守护进程，它监视（watch）API Server 上 `Service` 和 `EndpointSlice` 对象的变化，并将这些变化翻译成节点上的具体网络规则。
    *   **工作模式**：它有几种工作模式，最主要的是 `iptables` 和 `IPVS`。

*   **iptables 模式**：
    *   **DNAT 规则如何实现负载均衡**：当一个 Service (`10.96.0.10:80`) 有 3 个后端 Pod (`10.1.1.2`, `10.1.1.3`, `10.1.1.4`) 时，`kube-proxy` 会创建一系列 `iptables` 规则。
        1.  首先，在 `PREROUTING` 和 `OUTPUT` 链中，它会捕获所有目标地址是 `10.96.0.10:80` 的流量，并将其导向一个自定义的链，如 `KUBE-SERVICES`。
        2.  在 `KUBE-SERVICES` 链中，会有一条规则匹配 `10.96.0.10:80`，并跳转到另一个专属于这个 Service 的链，如 `KUBE-SVC-XYZ`。
        3.  在 `KUBE-SVC-XYZ` 链中，会使用 `statistic` 模块，以一定的概率（例如，1/3 的概率）跳转到三个不同的链 `KUBE-SEP-ABC`, `KUBE-SEP-DEF`, `KUBE-SEP-GHI`，每个链代表一个 Pod。
        4.  在每个 Pod 对应的链（如 `KUBE-SEP-ABC`）中，会有一条 **DNAT (Destination Network Address Translation)** 规则，将数据包的目标 IP 和端口修改为对应 Pod 的 IP 和端口（如 `10.1.1.2:8080`）。
        这样，通过概率选择和 DNAT，就实现了负载均衡。
    *   **优缺点分析**：
        *   **优点**：非常成熟、稳定，几乎所有 Linux 发行版都支持。
        *   **缺点**：当 Service 和 Pod 数量巨大（成千上万）时，`iptables` 规则会变得非常多，规则匹配是一个线性查找过程，会导致性能下降。

*   **IPVS (IP Virtual Server) 模式**：
    *   **IPVS 的工作原理**：IPVS 是 Linux 内核的一部分，专门用于高性能的四层负载均衡。它使用哈希表来存储规则，查找效率是 O(1)，远高于 `iptables` 的 O(n)。
    *   `kube-proxy` 在 IPVS 模式下，会为每个 Service 创建一个虚拟服务器，并为每个后端 Pod 创建一个真实服务器。当流量到达 Service 的 ClusterIP 时，IPVS 会根据预设的负载均衡算法（如轮询、最少连接）直接选择一个后端 Pod 并转发流量。
    *   **相比 iptables 的优势**：性能更高、延迟更低，尤其是在大规模集群中。

*   **eBPF 模式 (Cilium 等 CNI)**：
    *   **eBPF 技术简介**：eBPF 允许在不修改内核代码的情况下，将自定义的、安全的程序注入到内核的各个钩子点（如网络事件）。
    *   **如何实现**：像 Cilium 这样的 CNI 插件可以使用 eBPF 程序，在数据包进入网络接口时就进行处理。它可以完全绕过 `kube-proxy` 和 `iptables/IPVS`，直接在内核中完成 Service 的地址转换和负载均衡，提供了极致的性能。

### 3. 集群 DNS

服务发现的基石。

*   **CoreDNS 的作用和架构**：CoreDNS 是一个灵活、可扩展的 DNS 服务器，在现代 K8s 集群中作为标准的 DNS 服务。它以 Pod 的形式运行在集群中，并被暴露为一个 ClusterIP 类型的 Service。CoreDNS 监视 API Server，自动为集群中创建的每个 Service 生成 DNS 记录。
*   **Pod 如何通过 DNS 解析 Service 名**：当 Pod 内的应用尝试连接 `my-svc` 时，它实际上是向 `my-svc.default.svc.cluster.local` (假设在 default namespace) 发起 DNS 查询。
*   **`/etc/resolv.conf` 在 Pod 中的配置和意义**：`kubelet` 会为每个 Pod 动态生成这个文件。它通常包含：
    *   `nameserver <CoreDNS_Service_ClusterIP>`: 告诉 Pod 应该向哪个 DNS 服务器查询。
    *   `search <namespace>.svc.cluster.local svc.cluster.local cluster.local`: 定义 DNS 搜索域。这使得你可以在同一个 namespace 内直接使用 `my-svc`，而不是完整的 FQDN。
    *   `options ndots:5`: 当查询的域名中的点 `.` 少于 5 个时，会依次尝试 `search` 列表中的后缀。
*   **DNS 记录**：
    *   **A/AAAA 记录**: 将 `service-name.namespace.svc.cluster.local` 解析为 Service 的 ClusterIP (IPv4/IPv6)。
    *   **SRV 记录**: 用于发现 Service 的具名端口。例如，查询 `_http._tcp.my-svc.my-ns.svc.cluster.local` 可以获得 `http` 端口的端口号、协议等信息。

### 4. Ingress：集群流量的入口网关

*   **Ingress 与 Service 的区别 (七层 vs. 四层)**：
    *   `Service` (NodePort, LoadBalancer) 工作在 **TCP/IP 协议栈的第四层（传输层）**。它只关心 IP 和端口，进行流量转发。
    *   `Ingress` 工作在 **第七层（应用层）**，主要是 HTTP/HTTPS。它能理解 URL 路径、域名、请求头等，从而实现更复杂的路由规则。
*   **Ingress Controller 的角色**：`Ingress` 资源本身只是一个 **规则定义**（“我希望 `foo.com/bar` 的流量导向 `bar-service`”）。真正实现这些规则的是 **Ingress Controller**。Ingress Controller 是一个运行在集群中的反向代理服务器（如 Nginx, Traefik, HAProxy），它监听 `Ingress` 资源的变化，并动态更新自己的配置来实现路由。你需要单独在集群中部署一个 Ingress Controller。
*   **Ingress 规则**：
    *   **Host-based 路由**: 基于域名（HTTP Host header）进行路由。例如，`foo.com` 的流量给 Service A，`bar.com` 的流量给 Service B。
    *   **Path-based 路由**: 基于 URL 路径进行路由。例如，`example.com/foo` 给 Service A，`example.com/bar` 给 Service B。
*   **TLS/SSL 证书管理**：Ingress 是处理 HTTPS 流量的理想位置。你可以将 TLS 证书和私钥存储在 Kubernetes 的 `Secret` 对象中，并在 Ingress 规则里引用它，Ingress Controller 会自动配置 HTTPS。这被称为 **TLS 终止**。

### 5. Egress：集群流量的出口

*   **Pod 访问集群外部服务的流程**：当 Pod 内的应用需要访问互联网上的服务（如 `www.google.com`）时，数据包从 Pod 发出，经过 veth-pair 和 cni0 网桥到达宿主机。
*   **SNAT (源地址转换) 的发生时机**：由于 Pod IP 是集群内部的私有 IP，无法在公网上路由，因此在数据包离开宿主机之前，宿主机的内核会对其进行 **SNAT**，将数据包的源 IP 从 `Pod_IP` 修改为 `Node_IP`。这样，外部服务看到请求是来自 Node，而不是 Pod。
*   **Egress Gateway 的概念和应用场景**：在某些场景下，你需要让集群所有（或部分）的出站流量都从一个或少数几个固定的公网 IP 出去（例如，对方服务设置了 IP 白名单）。这时就可以部署一个 **Egress Gateway**。其原理是，通过配置路由规则，强制将所有出站流量都先路由到一个或一组特定的 Pod（即 Egress Gateway Pods），再由这些 Pods 进行 SNAT 后发往外部网络。

## 第三部分：网络策略与安全

### 1. NetworkPolicy

默认情况下，Kubernetes 集群中的网络是“全通”的，任何 Pod 都可以和任何其他 Pod 通信。NetworkPolicy 提供了 **Pod 级别的防火墙** 功能，允许你定义谁可以访问谁。

*   **网络策略的意义**：实现 **零信任网络**。你可以创建精细化的规则，比如“只有 frontend Pods 才能访问 backend Pods 的 8080 端口”，从而增强安全性。
*   `podSelector` 和 `namespaceSelector`:
    *   `podSelector`: 选择一组 Pod 来应用策略。如果为空，则选择 Namespace 下的所有 Pod。
    *   `namespaceSelector`: 基于 Namespace 的标签来选择允许或拒绝的流量来源/目标。
*   **Ingress (入站) 和 Egress (出站) 规则**:
    *   `ingress`: 定义哪些来源的流量可以进入被选中的 Pod。
    *   `egress`: 定义被选中的 Pod 可以访问哪些目标。
*   **默认拒绝 (Default Deny) 策略**：一旦为一个 Pod 应用了任何 `ingress` 策略，所有未被明确允许的入站流量都会被 **拒绝**。同理，`egress` 也是如此。这是实现安全隔离的关键。
*   **NetworkPolicy 的实现原理**：NetworkPolicy 资源本身也只是一个定义。**它的实现完全依赖于 CNI 插件**。像 Calico, Cilium, Weave Net 这样的 CNI 插件会读取 NetworkPolicy 对象，并将其转换成底层的网络规则（如 `iptables`, `eBPF`）来真正地执行策略。如果你用的 CNI 插件不支持 NetworkPolicy（如 Flannel 的早期版本），那么创建的 NetworkPolicy 资源将不会起任何作用。

## 第四部分：高级主题

### 1. IPv4/IPv6 双栈 (Dual-Stack)

随着 IPv4 地址的枯竭，IPv6 变得越来越重要。Kubernetes 提供了双栈支持。

*   **配置和优势**：管理员可以为 Pod 和 Service 同时分配 IPv4 和 IPv6 地址。这使得 Pod 既可以与只支持 IPv4 的老系统通信，也可以与支持 IPv6 的新系统进行原生通信，有利于平滑过渡。
*   **如何工作**：集群需要被配置为双栈模式。Pod 会从 CNI 插件获取到两个地址。Service 会被分配两个 ClusterIP（一个 v4，一个 v6）。DNS 会为 Service 返回 A 和 AAAA 两条记录。

### 2. 服务网格 (Service Mesh)

*   **服务网格解决了什么问题**：Kubernetes 解决了服务部署和网络连通性的问题，但对于服务间通信的 **治理**（如熔断、重试、超时、金丝雀发布）、**可观察性**（获取延迟、成功率等黄金指标）和 **安全**（端到端的 mTLS 加密）则力不从心。服务网格正是为了解决这些问题而生。
*   **基本概念：Sidecar 代理模式**：服务网格通过向每个业务 Pod 中 **自动注入一个代理（Sidecar）**，如 Envoy。Pod 内的应用不再直接与其他服务通信，而是将所有流量都发送给这个 Sidecar 代理。所有服务间的通信都被代理接管，从而可以在代理层实现上述高级功能，且对应用程序完全透明。
*   **主流项目**：
    *   **Istio**: 功能最强大、最复杂的服务网格，提供了极其丰富的流量治理、安全和可观察性能力。
    *   **Linkerd**: 以轻量、简单和高性能著称，专注于提供核心的可观察性、可靠性和安全性，更易于上手。

### 3. eBPF 与云原生网络

*   **eBPF 对 K8s 网络的革命性影响**：eBPF 允许在内核中安全地运行自定义代码，这为网络带来了前所未有的可能性。它可以实现高性能的负载均衡、复杂的网络策略、精细的流量监控，而无需修改内核或依赖 `iptables` 等传统工具。
*   **Cilium CNI 的核心优势**：Cilium 是一个完全基于 eBPF 的 CNI。它的优势包括：
    *   **性能**：通过绕过 `iptables` 和上层网络协议栈，提供了接近物理机的网络性能。
    *   **可观察性**：可以提供基于 API 层面（如 HTTP 请求）的可见性，知道是哪个 Pod 在调用哪个 API，而不仅仅是 IP 和端口。
    *   **安全**：可以基于服务身份（而不仅是 IP 地址）来实施网络策略。

## 第五部分：网络诊断与排错

这是实践中最重要的技能。

### 1. 常用诊断工具

*   `kubectl get/describe <resource> <name>`: 查看资源的状态、事件和配置。`describe svc my-svc` 可以看到 `Endpoints` 列表，`describe pod my-pod` 可以看到 Pod 的事件和 IP。
*   `kubectl logs <pod-name>`: 查看 CoreDNS, CNI 插件, Ingress Controller 等组件的日志，是排错的第一手资料。
*   `kubectl exec -it <pod-name> -- /bin/sh`: **这是最重要的排错命令之一**。它允许你进入 Pod 内部，从 Pod 的视角来测试网络。
    *   `ping <ip>`: 测试三层连通性。
    *   `curl -v <url>` 或 `nc -vz <host> <port>`: 测试七层或四层连通性。
*   `nslookup <service-name>` 和 `dig <service-name>`: 在 Pod 内部执行，用来排查 DNS 解析问题。

### 2. Linux 网络工具集

当 `kubectl` 不足以解决问题时，你需要登录到 Node 上，使用这些底层工具。

*   `ip addr` 和 `ip route`: 查看节点和 Pod 的 veth-pair 的 IP 地址、路由表等。
*   `iptables-save` 或 `ipvsadm -Ln`: 查看 `kube-proxy` 生成的规则是否正确。
*   `tcpdump` 和 `wireshark`: 终极武器。可以在 veth 接口、网桥、物理网卡等不同位置抓包，分析数据包的流向、是否被 SNAT、是否被封装等。例如，在 Node2 上 `tcpdump -ni eth0 udp port 4789` 可以查看是否有 VXLAN 流量从 Node1 过来。

### 3. 常见网络问题分析

*   **DNS 解析失败**:
    1.  进入 Pod，`nslookup kubernetes.default` 看是否能解析。
    2.  检查 CoreDNS Pod 是否正常运行，看它的日志。
    3.  检查是否有 NetworkPolicy 阻止了你的 Pod 访问 CoreDNS 的 53 端口。
*   **Service 无法访问**:
    1.  `describe svc <service-name>`，检查 `Endpoints` 列表是否有 IP。如果没有，说明 Service 的 `selector` 和 Pod 的 `labels` 不匹配，或者 Pod 不健康。
    2.  如果 Endpoints 正常，登录到 Node 上，检查 `iptables` 或 `ipvs` 规则是否正确生成。
    3.  检查是否有 NetworkPolicy 阻止了访问。
*   **跨节点通信故障**:
    1.  检查 CNI 插件的 Pod (如 `calico-node`, `kube-flannel-ds`) 在两个节点上是否都正常运行，看日志。
    2.  在两个节点上 `ip route` 查看路由是否正确。Calico 模式下看是否有到对方 Pod CIDR 的路由；Flannel 模式下看是否有到 `flannel.1` 设备的路由。
    3.  检查云厂商的安全组或物理防火墙是否阻止了节点间的通信（特别是 CNI 需要的端口，如 VXLAN 的 UDP 4789）。
    4.  使用 `tcpdump` 在源和目的节点上同时抓包，看包发出去了没有，到达了没有。

---

恭喜你！你已经完成了一次 Kubernetes 网络的深度旅行。这份笔记内容很多，建议你结合实践，在自己的测试集群中亲自验证每个流程，这样才能真正地将知识内化。祝学习愉快！