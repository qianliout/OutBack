# Kubernetes 外部流量接入全链路剖析：从公网 IP 到业务 Pod

本文档系统性地梳理了外部流量如何从公网 DNS 解析开始，经过云厂商 NAT 网关，最终精准打入 Kubernetes 集群内部特定业务 Pod 的完整链路。

本案例基于真实的生产环境架构：未直接使用昂贵的云原生负载均衡器 (ELB/LoadBalancer)，而是通过 **EIP + NAT 网关 (DNAT) + Ingress (HostNetwork)** 的模式实现流量接入。

---

## 一、 整体架构概览

在我们的场景中，一条外部请求到达前端 Pod，经历了以下完整的关键节点：

1.  **用户浏览器 / DNS**：发起请求并解析域名为公网 EIP。
2.  **云厂商 NAT 网关**：接收公网流量，通过 DNAT 转换为内网节点 IP。
3.  **K8s Node 节点 (HostNetwork)**：集群 Master 或 Worker 节点，通过特定端口接收内网流量。
4.  **Ingress Controller (如 Nginx)**：七层反向代理，基于 HTTP 请求头 (`Host`) 进行路由拆解。
5.  **K8s Service / Endpoints**：服务发现机制，锁定业务 Pod 真实 IP。
6.  **业务 Pod**：最终处理请求的容器。

---

## 二、 核心链路拆解与原理解析

### 1. 公网入口：DNS 解析与弹性公网 IP (EIP)
*   **现象**：多个不同的业务域名（例如 `console.tensorsecurity.cn`, `console-miaoyun.tensorsecurity.cn` 等）在 DNS（如腾讯云 DNSPod）中的 A 记录，全部指向同一个华为云 EIP（例如 `116.63.188.121`）。
*   **原理**：这是多租户或多服务复用公网 IP 的标准做法，旨在节省公网 IP 资源。在四层（TCP/IP）网络看来，这些请求的目的地都是完全一致的。

### 2. 云网络层：NAT 网关与 DNAT 转换
*   **现象**：在华为云控制台中，该 EIP 并没有绑定任何 ELB（弹性负载均衡实例），而是绑定在了一个名为 `nat-17221` 的 NAT 网关上。
*   **原理**：
    *   NAT 网关配置了 **DNAT (Destination NAT) 规则**。
    *   例如规则：`116.63.188.121:443 -> 172.21.2.229:443 (TCP)`。
    *   当外部流量打到 EIP 时，NAT 网关会在网络层（IP层）修改数据包的目的地址，将其从公网 IP 替换为 K8s 集群内某台 Node 节点的内网 IP（`172.21.2.229`）。

### 3. K8s 集群入口：HostNetwork 模式的 Ingress Controller
*   **现象**：查看集群中的 `ingress-controller` Pod 时，发现它的 IP 就是宿主机的内网 IP `172.21.2.229`。
    ```bash
    kube-system  ingress-controller-ingress-nginx-controller-xxx  1/1  Running  172.21.2.229  ecs-prod-master01
    ```
*   **原理**：
    *   通常 Ingress Controller Pod 会配置 `hostNetwork: true`。
    *   这意味着该 Pod 突破了 K8s 的容器网络（如 Flannel/Calico 提供的 Pod IP），直接绑定并占用了宿主机（`ecs-prod-master01`）的 80 和 443 端口。
    *   因此，从 NAT 网关转发过来的流量，直接进入了宿主机的端口，被 Ingress Controller 进程（如 Nginx）接管。

### 4. 七层路由分发：基于 Host 头的虚拟主机技术 (核心魔法)
*   **疑问**：既然所有域名都指向同一个 IP，进入了同一个 Nginx，流量是如何被精准区分到不同 Pod 的？
*   **现象**：执行 `kubectl get ingress -A`，可以看到大量不同域名的 Ingress 资源，它们都共享 80/443 端口。
*   **原理（七层路由）**：
    *   在四层（传输层），这些请求毫无区别。但在七层（应用层），浏览器会在 HTTP 请求头中携带 **`Host`** 字段（例如 `Host: console-miaoyun.tensorsecurity.cn`）。
    *   Ingress Controller 本质是一个七层代理服务器。它会拆解 HTTP 数据包，读取 `Host` 字段。
    *   K8s 中的 Ingress 资源，实际上会被动态转化为 Nginx 的 `server_name` 配置：
        ```nginx
        server {
            listen 80;
            server_name console-miaoyun.tensorsecurity.cn;
            location / { proxy_pass http://<对应Pod的IP>; }
        }
        ```
    *   Nginx 通过匹配 HTTP 头中的 `Host` 值与 `server_name`，从而决定将流量转发给哪个后端的 `proxy_pass`。

### 5. 服务发现与最终触达：Service 与 Endpoints
*   **现象**：Ingress 规则将流量指向了一个具体的 Service。
*   **原理**：
    *   Ingress Controller 并不会将流量先发给 Service 的 ClusterIP（避免额外的 Kube-Proxy iptables/IPVS 转发开销）。
    *   相反，它会监听 K8s API，动态获取该 Service 背后对应的 **Endpoints 列表**（即真实 Pod 的 IP 地址列表）。
    *   Nginx 直接将流量代理（反向代理）到这些真实的 Pod IP 上，完成闭环。

---

## 三、 实战排障指南

当你发现某个域名（如 `console-miaoyun.tensorsecurity.cn`）无法访问时，请按照以下链路正向排查：

1.  **查 DNS**：`ping console-miaoyun.tensorsecurity.cn`，确认是否解析到了预期的 EIP（116.63.188.121）。
2.  **查 NAT 规则**：在云控制台确认 NAT 网关的 DNAT 规则是否存在，且映射的内网 IP（172.21.2.229）和端口（80/443）正确。
3.  **查 Ingress Controller 状态**：
    *   登录目标节点（`172.21.2.229`），检查 80/443 端口是否被监听。
    *   查看 Controller Pod 日志：`kubectl logs -n kube-system <ingress-controller-pod-name>`，看是否有错误或拒绝连接的信息。
4.  **查 Ingress 路由规则**：
    *   `kubectl get ingress -A | grep console-miaoyun`，找到对应的 Ingress。
    *   `kubectl describe ingress <ingress名称> -n <命名空间>`，确认 `Backend` 指向的 Service 名称是否正确。
5.  **查 Service 与 Endpoints**：
    *   `kubectl describe svc <service名称> -n <命名空间>`，检查 `Endpoints` 字段是否有实际的 IP 地址。如果为空，说明背后的 Pod 挂了或者 Label Selector 没匹配上。
6.  **查 业务 Pod**：
    *   `kubectl get pods -n <命名空间> -l <service的selector>`，查看 Pod 是否处于 `Running` 和 `Ready` 状态。
    *   查看业务 Pod 日志排查代码级错误：`kubectl logs -n <命名空间> <pod名称>`。

---

> 总结：理解了从四层 NAT 转换到七层 Host 路由拆解的过程，就能在复杂的云原生网络中迅速定位故障节点，这套架构也是目前中小规模集群极具性价比的流量接入方案。
