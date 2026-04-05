在 Kubernetes 中，`ClusterIP` 和 `NodePort` 是两种常见的 Service 类型，它们的主要区别在于访问方式和适用场景：

***

### **1. ClusterIP**

*   **作用**：默认的 Service 类型，为集群内部提供访问服务的虚拟 IP。
*   **特点**：
    *   仅在集群内部（Pod 或节点）可访问，**外部无法直接访问**。
    *   通过一个稳定的虚拟 IP（ClusterIP）和端口暴露服务。
    *   通常用于集群内服务间的通信（如微服务之间的调用）。
*   **示例配置**：
    ```yaml
    apiVersion: v1
    kind: Service
    metadata:
      name: my-service
    spec:
      type: ClusterIP  # 可省略（默认值）
      selector:
        app: my-app
      ports:
        - port: 80       # Service 的端口
          targetPort: 80 # Pod 的端口
    ```
*   **访问方式**：
    *   集群内通过 `my-service:80` 或 `<ClusterIP>:80` 访问。

***

### **2. NodePort**

*   **作用**：在 ClusterIP 的基础上，通过每个节点的固定端口（NodePort）暴露服务，允许外部访问。
*   **特点**：
    *   在集群所有节点的 **同一端口**（默认范围 30000-32767）上开放服务。
    *   外部用户可以通过任意节点的 `IP:NodePort` 访问服务。
    *   底层会**自动创建 ClusterIP**，集群内仍可通过 ClusterIP 访问。
*   **示例配置**：
    ```yaml
    apiVersion: v1
    kind: Service
    metadata:
      name: my-nodeport-service
    spec:
      type: NodePort
      selector:
        app: my-app
      ports:
        - port: 80        # Service 的端口（通过 ClusterIP 访问时使用）
          targetPort: 80   # Pod 的端口
          nodePort: 31000 # 手动指定节点端口（可选，不指定则随机分配）
    ```
*   **访问方式**：
    *   集群外：`<任意节点IP>:31000`
    *   集群内：`my-nodeport-service:80` 或 `<ClusterIP>:80`

***

### **核心区别**

| 特性        | ClusterIP            | NodePort               |
| --------- | -------------------- | ---------------------- |
| **访问范围**  | 仅集群内部                | 集群外部可通过节点 IP 访问        |
| **IP/端口** | 虚拟 IP（ClusterIP）+ 端口 | 节点 IP + 固定端口（NodePort） |
| **用途**    | 内部服务通信（如微服务）         | 开发测试、临时外部访问            |
| **性能**    | 高效（不经过节点网络）          | 多一层 NAT 转换，性能略低        |

***

### **补充说明**

*   **NodePort 的局限性**：
    *   生产环境通常配合 `LoadBalancer`（云厂商）或 `Ingress`（七层代理）使用，而非直接暴露 NodePort。
    *   节点 IP 可能变化（如扩缩容），需结合外部负载均衡器。
*   **其他类型**：
    *   `LoadBalancer`：在 NodePort 基础上，通过云厂商的负载均衡器暴露服务。
    *   `ExternalName`：通过 DNS CNAME 映射到外部服务。

根据需求选择类型：优先用 `ClusterIP`（内部访问），需要外部访问时再考虑 `NodePort` 或更高级的 `Ingress`。
