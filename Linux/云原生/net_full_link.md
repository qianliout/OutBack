# Kubernetes 学习笔记：从浏览器到 Pod 的全链路深度解析

## 引言：我们的学习之旅

### 学习目标
本笔记旨在通过追踪一个标准的 Web 请求在 Kubernetes (K8s) 集群中的完整生命周期，帮助您深入理解 K8s 的核心网络概念和工作流。我们将从用户的浏览器开始，一直深入到运行应用程序的 Pod 内部。

### 核心场景
用户在浏览器中输入 `https://www.hello.com` 并按下回车。这个请求最终被 K8s 集群中运行的 `frontend` 应用 Pod 处理。我们将详细剖析从 DNS 解析到 Pod 响应的每一个环节。

### 知识脉络概览
我们的学习路径将遵循请求的实际流向，层层递进：
1.  **外部流量入口**：DNS, Ingress Controller, Ingress 资源。
2.  **内部服务发现与路由**：Service, EndpointSlice, kube-proxy。
3.  **应用负载**：Pod, Container。
4.  **安全**：HTTPS 证书管理。
5.  **可观测性**：全链路故障排查。

---

## 第一部分：流量入口 - Ingress 与 Ingress Controller

当用户请求 `https://www.hello.com` 时，第一站是 K8s 集群的入口。Ingress 机制是处理外部访问的核心。

### 1.1 浏览器如何找到 K8s 集群？

首先，用户的浏览器必须知道 `www.hello.com` 对应的 IP 地址。

- **DNS 解析**：您需要在您的 DNS 提供商处创建一条记录（通常是 `A` 记录或 `CNAME` 记录），将域名 `www.hello.com` 指向 **Ingress Controller Service 的外部 IP 地址 (External IP)**。

  这个 External IP 是 Ingress Controller 为了接收外部流量而暴露的公网 IP。通常，它由云服务商的负载均衡器（LoadBalancer）提供。

  > **官方文档**: [DNS for Services and Pods](https://kubernetes.io/docs/concepts/services-networking/dns-pod-service/)

### 1.2 Ingress Controller：规则的执行者

仅仅有 DNS 指向还不够，K8s 集群需要知道如何处理发往这个 IP 的请求。这就是 Ingress Controller 的职责。

- **核心作用**：Ingress Controller 是一个运行在集群中的应用程序（通常是一个或多个 Pod），它负责实现 Ingress 资源中定义的路由规则。它是一个反向代理和负载均衡器。

- **工作原理** (以 NGINX Ingress Controller 为例):
    1.  **Watch K8s API**：Ingress Controller 持续监听 (watch) K8s API Server 中关于 Ingress、Service、EndpointSlice 和 Secret 资源的创建、更新和删除事件。
    2.  **动态生成配置**：当检测到 Ingress 资源发生变化时，它会根据所有 Ingress 规则，动态生成一个 NGINX 配置文件 (`nginx.conf`)。
    3.  **加载配置**：它将新生成的配置热加载到其管理的 NGINX 实例中，使其立即生效，而无需重启 NGINX 服务。

  > **官方文档**: [Ingress Controllers](https://kubernetes.io/docs/concepts/services-networking/ingress-controllers/)

### 1.3 Ingress 与 NGINX 的协作

`Ingress` 资源本身只是一个声明式的配置，它告诉 Ingress Controller 你期望的路由规则是什么。

这是一个典型的 Ingress 资源示例：

```yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: hello-ingress
  annotations:
    nginx.ingress.kubernetes.io/rewrite-target: /
spec:
  rules:
  - host: "www.hello.com"
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: frontend-service
            port:
              number: 80
```

这段 YAML 会被 NGINX Ingress Controller 翻译成类似下面的 NGINX 配置：

```nginx
server {
    server_name www.hello.com;

    location / {
        set $proxy_upstream_name "default-frontend-service-80";
        proxy_pass http://<IP-of-frontend-service>;
        # ... 其他由 Ingress Controller 自动生成的配置
    }
}
```

- **Annotation 的力量**：Ingress 资源的标准字段有限，但通过 `metadata.annotations`，我们可以极大地扩展其功能，以控制 NGINX 的具体行为。例如：
    - `nginx.ingress.kubernetes.io/ssl-redirect`: 强制将 HTTP 请求重定向到 HTTPS。
    - `nginx.ingress.kubernetes.io/cors-enable`: 启用跨域资源共享 (CORS)。
    - `nginx.ingress.kubernetes.io/proxy-body-size`: 设置客户端请求体的最大允许大小。

  > **官方文档**: [NGINX Ingress Controller Annotations](https://kubernetes.github.io/ingress-nginx/user-guide/nginx-configuration/annotations/)

---

## 第二部分：服务发现与负载均衡 - Service 的角色

Ingress Controller 已经知道要将 `www.hello.com` 的请求转发给 `frontend-service`。但 `frontend-service` 是什么？它又如何找到最终的 Pod？

### 2.1 Service：连接内外的桥梁

Service 是 K8s 中实现服务发现和负载均衡的核心抽象。它为一组功能相同的 Pod 提供了一个统一、稳定的访问入口。

- **稳定的访问点**：Pod 是“短暂的”，它们可能因为节点故障、扩缩容等原因被销毁和重建，其 IP 地址会随之改变。Service 提供了一个虚拟 IP (ClusterIP)，这个 IP 在 Service 的生命周期内是固定的。Ingress Controller 和其他内部服务都应该通过 Service 的 ClusterIP 或服务名称来访问 Pod，而不是直接访问 Pod IP。

### 2.2 Service 如何管理一组 Pod？

Service 通过 **标签选择器 (Label Selector)** 来动态地发现并关联它应该代理的 Pod。

- **Labels & Selectors**：
    1.  **为 Pod 打标签**：在你的 `frontend` Pod (或其控制器，如 Deployment) 的 `metadata` 中定义 `labels`，例如 `app: frontend`。
    2.  **Service 定义选择器**：在 `frontend-service` 的 `spec` 中定义 `selector`，使其匹配 Pod 的标签，例如 `selector: { app: frontend }`。

- **EndpointSlice**：K8s 会自动创建一个名为 `EndpointSlice` 的对象。它包含了一个由 Service 的选择器匹配到的所有健康 Pod 的 IP 地址和端口列表。当 Pod 创建、删除或状态改变时，EndpointSlice Controller 会实时更新这个列表。

  你可以通过 `kubectl describe service frontend-service` 查看到 `Endpoints` 字段，它就来自 EndpointSlice。

  > **官方文档**: [Service](https://kubernetes.io/docs/concepts/services-networking/service/) | [EndpointSlices](https://kubernetes.io/docs/concepts/services-networking/endpoint-slices/)

### 2.3 从 Service 到 Pod 的最后一跳

请求已经到达了 Service 的虚拟 IP (ClusterIP)，但这个 IP 并不与任何网络设备绑定，它无法被 ping 通。那么流量是如何最终到达某个具体 Pod 的呢？答案是 `kube-proxy`。

- **kube-proxy 的角色**：`kube-proxy` 是一个运行在集群中每个节点上的守护进程。它的核心任务就是将发往 Service ClusterIP 的流量，通过某种机制转发到该 Service 对应的某个后端 Pod IP。

- **实现模式**：
    1.  **iptables (默认)**：`kube-proxy` 会在每个节点上创建一系列 `iptables` 规则。这些规则会捕获发往 Service ClusterIP 的流量，然后通过 DNAT (Destination Network Address Translation) 将其目标地址修改为某个随机选择的后端 Pod IP。
    2.  **IPVS (IP Virtual Server)**：在较大规模的集群中，IPVS 模式性能更优。它使用内核中的 IPVS 负载均衡功能，创建虚拟服务器来转发流量。

  无论哪种模式，最终效果都是：当 Ingress Controller 将请求发送到 `frontend-service` 的 ClusterIP 时，该请求在节点上被 `kube-proxy` 设置的规则拦截，并被负载均衡地转发到了一个健康的 `frontend` Pod 的实际 IP 地址上。

  > **官方文档**: [kube-proxy](https://kubernetes.io/docs/reference/command-line-tools-reference/kube-proxy/)

### 2.4 Service 的类型与选择

K8s 提供了多种 Service 类型，以适应不同的暴露需求。它们的底层实现和适用场景有本质区别。

> **官方文档**: [Publishing Services (Service Types)](https://kubernetes.io/docs/concepts/services-networking/service/#publishing-services-service-types)

#### 1. ClusterIP (默认)

-   **本质**：只在集群内部暴露一个虚拟 IP (`ClusterIP`)。这是 Service 的默认类型。
-   **底层原理**：`kube-proxy` 在每个节点上配置 `iptables` 或 `ipvs` 规则，将发往这个 `ClusterIP` 的流量转发到后端的 Pod。这个 IP 地址只能在集群内部访问。
-   **使用场景**：**集群内部服务间通信**。例如，`frontend` 服务需要调用 `backend-api` 服务，`backend-api` 就应该使用 `ClusterIP` 类型的 Service。

#### 2. NodePort

-   **本质**：在 `ClusterIP` 的基础上，额外在集群中**每个节点**上都暴露一个相同的静态端口 (`NodePort`)。
-   **底层原理**：`kube-proxy` 会配置规则，将发往 ` <NodeIP>:<NodePort>` 的流量，最终转发到 Service 对应的 Pod。`NodePort` 类型的 Service 会自动创建一个 `ClusterIP` Service。
-   **使用场景**：
    -   **开发和测试环境**：快速暴露服务进行测试，而无需配置复杂的 Ingress 或负载均衡器。
    -   **当没有外部负载均衡器时**：在裸金属（Bare-Metal）集群或没有云厂商负载均衡器支持的环境中，`NodePort` 是将服务暴露给外部的一种方式。通常需要配合外部的负载均衡设备（如 F5, HAProxy）来使用。

#### 3. LoadBalancer

-   **本质**：在 `NodePort` 的基础上，进一步利用云服务商（如 AWS, GCP, Azure）提供的**外部负载均衡器**来暴露服务。
-   **底层原理**：当创建 `LoadBalancer` 类型的 Service 时，K8s 会向云厂商的 API 发出请求，创建一个外部负载均衡器。该负载均衡器会获得一个公网 IP，并将流量路由到所有节点的 `NodePort` 上。`LoadBalancer` 类型的 Service 会自动创建一个 `NodePort` Service 和一个 `ClusterIP` Service。
-   **使用场景**：**向公网暴露服务的标准方式**。当用户需要一个稳定的公网 IP 来访问应用时，这是首选。我们的 `Ingress Controller` 本身就是通过一个 `LoadBalancer` 类型的 Service 来接收外部流量的。

#### 4. ExternalName

-   **本质**：它不进行任何代理或转发，而是为 Service 创建一个 DNS `CNAME` 记录。
-   **底层原理**：当集群内的 Pod 查询这个 Service 的 DNS 名称时，`kube-dns` 或 `CoreDNS` 会直接返回 `spec.externalName` 字段中定义的外部域名。
-   **使用场景**：**在集群内部为外部服务创建一个别名**。例如，你的应用需要访问一个外部数据库 `rds.amazonaws.com`。你可以创建一个 `ExternalName` 类型的 Service，名为 `database`，`externalName` 指向 `rds.amazonaws.com`。这样，你的应用代码中就可以硬编码 `database` 作为地址，而无需关心实际的外部地址，方便未来迁移和配置管理。

---

## 第三部分：安全与证书管理

为了实现 `https://www.hello.com` 的访问，我们需要为 Ingress 配置 TLS 证书。

### 3.1 HTTPS 工作原理简介

HTTPS = HTTP + TLS。TLS (Transport Layer Security) 协议通过证书和密钥实现身份验证、数据加密和完整性保护。

### 3.2 在 Ingress 层面配置 HTTPS 证书

这是最常见和推荐的方式，称为 **TLS 终止 (TLS Termination)**。外部流量是加密的，但到达 Ingress Controller 后被解密，Ingress Controller 与后端 Pod 之间是普通的 HTTP 流量。

1.  **创建 Secret**：首先，你需要将你的 TLS 证书和私钥存储在一个 K8s Secret 中。这个 Secret 的类型必须是 `kubernetes.io/tls`。

    ```bash
    kubectl create secret tls hello-tls --cert=path/to/tls.crt --key=path/to/tls.key
    ```

2.  **修改 Ingress 资源**：在 Ingress 的 `spec` 中添加 `tls` 字段，引用刚刚创建的 Secret。

    ```yaml
    apiVersion: networking.k8s.io/v1
    kind: Ingress
    metadata:
      name: hello-ingress
    spec:
      tls:
      - hosts:
        - www.hello.com
        secretName: hello-tls # 引用 Secret
      rules:
      - host: "www.hello.com"
        http:
          # ... paths 配置
    ```

  > **官方文档**: [Ingress TLS](https://kubernetes.io/docs/concepts/services-networking/ingress/#tls)

### 3.3 自动化证书管理：cert-manager

手动管理证书（特别是续签）非常繁琐。`cert-manager` 是一个流行的 K8s 插件，可以与 Let's Encrypt 等 ACME CA 集成，自动完成证书的申请、续签和注入。

### 3.4 其他可以配置证书的位置

- **Service Mesh (如 Istio/Linkerd)**：在服务网格中，可以在网格的入口网关 (Gateway) 上配置证书，实现更高级的流量管理和安全策略。
- **Pod 内部**：你也可以不在 Ingress 处终止 TLS，而是将加密流量直接透传到 Pod，由 Pod 内的应用程序自己处理 TLS 握手和解密。这称为 **TLS Passthrough**。这种方式提供了端到端的加密，但管理更复杂，且 Ingress Controller 无法检查七层流量内容。

---

## 第四部分：全链路故障排查 (Troubleshooting)

当 `https://www.hello.com` 访问失败时，遵循“由外向内，层层排查”的原则。

### 1. 客户端与 DNS
- **问题**：浏览器显示“无法访问此网站”或“DNS_PROBE_FINISHED_NXDOMAIN”。
- **排查**：
    - `dig www.hello.com` 或 `nslookup www.hello.com`：检查 DNS 解析是否正确返回了 Ingress Controller 的 External IP。
    - `curl -v https://www.hello.com`：查看详细的连接过程，检查 TLS 握手是否成功。如果 IP 不对，是 DNS 问题；如果连接超时，可能是网络防火墙问题。

### 2. Ingress 环节
- **问题**：返回 404 (Not Found) 或 503 (Service Temporarily Unavailable)。
- **排查**：
    - `kubectl get ingress`: 确认 Ingress 资源存在且 `HOSTS` 和 `ADDRESS` 字段正确。
    - `kubectl describe ingress hello-ingress`: 这是最重要的命令。检查 `Rules` 是否正确，`Backend` 是否指向了正确的 Service 和端口。特别关注 `Events` 部分，它会显示 Ingress Controller 处理此 Ingress 时的错误信息。
    - `kubectl logs <ingress-controller-pod-name> -n <namespace>`: 查看 Ingress Controller 的实时日志，这是定位问题的金矿。你可以看到它是否收到了请求，以及为什么转发失败。

### 3. Service 环节
- **问题**：Ingress 日志显示无法连接到上游 (upstream)。
- **排查**：
    - `kubectl describe service frontend-service`: 检查 `Selector` 是否与你的 Pod `Labels` 匹配。最重要的是看 `Endpoints` 字段，如果这里是空的 (`<none>`)，说明 Service 没有找到任何健康的 Pod。
    - `kubectl get endpointslices -l kubernetes.io/service-name=frontend-service`: 更详细地查看 EndpointSlice，确认 Pod IP 列表是否正确。

### 4. Pod 环节
- **问题**：Service 的 Endpoints 列表为空，或者返回 502 (Bad Gateway)。
- **排查**：
    - `kubectl get pods -l app=frontend`: 检查 Pod 是否处于 `Running` 状态。如果不是，使用 `describe` 查看原因。
    - `kubectl describe pod <pod-name>`: 检查 `Events`，常见的错误有：
        - `ImagePullBackOff`: 镜像拉取失败。
        - `CrashLoopBackOff`: 容器启动后立即崩溃，反复重启。
        - `FailedScheduling`: 节点资源不足或不满足调度条件。
        - `Unhealthy`: Pod 的健康检查 (Liveness/Readiness Probe) 失败。
    - `kubectl logs <pod-name>`: 查看应用容器的标准输出日志，定位应用程序本身的错误。
    - `kubectl exec -it <pod-name> -- /bin/sh`: 进入容器内部，尝试手动访问服务端口 (`curl localhost:80`)，检查应用是否在正常监听。

### 5. 网络策略 (Network Policy)
- **问题**：所有环节看起来都正常，但流量就是不通。
- **排查**：
    - `kubectl get networkpolicy`: 检查是否有网络策略存在。如果有，仔细阅读其规则，确认它是否允许从 Ingress Controller Pod 到 `frontend` Pod 的流量。

通过以上步骤，您应该能够系统地定位并解决从浏览器到 Pod 整个链路中的绝大多数问题。
