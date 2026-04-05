# Kubernetes容器编排

## 目录

- [3.1 clusterIP和nodePort](#31-clusterIP和nodePort)
- [3.2 readiness_liveness_startup](#32-readiness_liveness_startup)
- [3.3 通过文件创建_configMap](#33-通过文件创建_configMap)
- [3.4 ubuntu安装k3d](#34-ubuntu安装k3d)

---

## 3.1 clusterIP和nodePort

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

---

## 3.2 readiness_liveness_startup

# readiness liveness startup
- apiVersion: apps/v1
- kind: Deployment
- metadata:
- name: web-demo
- namespace: dev
- spec:
- selector:
- matchLabels:
- app: web-demo
- replicas: 1
- template:
- metadata:
- labels:
- app: web-demo
- spec:
- containers:
- name: web-demo
- image: hub.mooc.com/kubernetes/web:v1
- ports:
- containerPort: 8080
- livenessProbe:
- tcpSocket:
- port: 8080
- initialDelaySeconds: 20 # 容器启动多久后执行检查
- periodSeconds: 10
- failureThreshold: 2
- successThreshold: 1
- timeoutSeconds: 5
- readinessProbe:
- tcpSocket:
- port: 8080
- initialDelaySeconds: 20 # 容器启动多久后执行检查
- periodSeconds: 10
- failureThreshold: 2
- successThreshold: 1
- timeoutSeconds: 5

---

## 几个参数的含义initialDelaySeconds: 20 # 容器启动多久后执行检查

periodSeconds: 10 # 每10秒检查一次failureThreshold: 2 # 两次检查出错就认为检查不通过
successThreshold: 1 # 一次检查成功就认为检查成功timeoutSeconds: 5 # 执行tcp检查时的超时时间，防止执行命令一真卡住
### kubelet 使用存活探测器（livenessProbe）来知道什么时候要重启容器。 例如，存活探测器可以捕捉
到死锁（应用程序在运行，但是无法继续执行后面的步骤）。 这样的情况下重启容器有助于让应用程序在有问题的情况下更可用。
### kubelet 使用就绪探测器（readinessProbe）可以知道容器什么时候准备好了并可以开始接受请求流
量， 当一个 Pod 内的所有容器都准备好了，才能把这个 Pod 看作就绪了。 这种信号的一个用途就是控制哪个 Pod 作为 Service 的后端。 在 Pod 还没有准备好的时候，会从 Service 的负载均衡器中被剔
除的。
### kubelet 使用启动探测器（startupProbe）可以知道应用程序容器什么时候启动了。 如果配置了这类探
测器，就可以控制容器在启动成功后再进行存活性和就绪检查， 确保这些存活、就绪探测器不会影响应用程序的启动。 这可以用于对慢启动容器进行存活性检测，避免它们在启动运行之前就被杀掉。

---

---

## 3.3 通过文件创建_configMap

# 通过文件创建 configMap
这样创建出来的 configMap 相当于有个文件:nginx-default-config.conf ，所以这里的文档名应该写对
#!/bin/bash
```bash
kubectl create configmap default-nginx-conf \
```
--from-file=nginx-default-config.conf \
--namespace=tensorsec

---

---

## 3.4 ubuntu安装k3d

### 1,安装docker
### 2,安装k3d
### 因为腾讯云不能访问外网，所以只能以安装包的方式
获取下载地址下载之后 chmod +x   获取执行权限
### 安装kubectl
```bash
curl -LO "$(curl -L -s
```
"Install Docker Engine on Ubuntu | Docker Documentation
Releases · k3d-io/k3d (github.com)https://dl.k8s.io/release/
https://dl.k8s.io/release/stable.txt)/bin/linux/amd64/kubectl

---

---
