# Helm 完整学习笔记

## 第一部分：Helm 基础篇

### 1.1 Helm 简介与核心概念

#### 什么是Helm
Helm是Kubernetes的包管理器。它允许开发者和运维人员更轻松地打包、配置、部署和管理Kubernetes应用。

#### Helm与Kubernetes的关系
Kubernetes本身通过YAML文件来管理应用资源，但当应用复杂、微服务众多时，手动管理大量的YAML文件变得非常困难。Helm通过将一组相关的K8s资源打包成一个可管理的单元（Chart），极大地简化了这个过程。

#### Helm的三大核心概念
- **Chart**: Helm的打包格式，它包含了创建Kubernetes应用实例所需的所有资源定义、配置信息和依赖关系。可以将其理解为`yum`或`apt`的软件包。
- **Repository**: Chart的存储库。Helm客户端可以从不同的仓库中查找和下载Chart。
- **Release**: 在Kubernetes集群上运行的Chart的实例。每一次安装Chart都会创建一个新的Release。

#### Helm架构与工作原理 (Helm 3)
Helm 3移除了Tiller（服务器端组件），架构变得更加简单和安全。

- **Helm CLI**: 用户通过Helm命令行工具与Kubernetes API服务器直接交互。
- **工作流程**:
    1.  用户执行`helm install`命令。
    2.  Helm CLI读取Chart，并结合用户提供的`values.yaml`和命令行参数，渲染出最终的Kubernetes YAML清单。
    3.  Helm CLI将生成的YAML清单发送给Kubernetes API服务器。
    4.  Kubernetes根据清单创建或更新相应的资源。
    5.  Helm将Release信息存储在Kubernetes的Secret资源中，以便跟踪和管理。

#### 为什么需要Helm
- **管理复杂性**: 将复杂应用的所有K8s资源打包成一个Chart，简化管理。
- **可重用性**: 一次创建Chart，可以在任何Kubernetes集群中多次部署。
- **标准化部署**: 确保开发、测试和生产环境的部署流程一致。
- **版本控制**: 对Release进行版本管理，轻松实现升级和回滚。

### 1.2 Helm 安装与配置

#### Helm版本选择（v2 vs v3）
强烈建议使用 **Helm 3**。Helm 2的架构包含一个名为Tiller的集群内组件，存在安全风险和权限管理问题。Helm 3已完全移除Tiller，更加安全、简洁。

#### 安装Helm CLI
**macOS (使用Homebrew):**
```bash
brew install helm
```

**Linux (使用脚本):**
```bash
curl -fsSL -o get_helm.sh https://raw.githubusercontent.com/helm/helm/main/scripts/get-helm-3
chmod 700 get_helm.sh
./get_helm.sh
```

#### 配置Helm（初始化、仓库配置）
Helm 3无需初始化。安装后即可使用。

**添加常用的Chart仓库:**
```bash
# 添加官方的Bitnami仓库，包含大量高质量的Chart
helm repo add bitnami https://charts.bitnami.com/bitnami

# 更新仓库信息
helm repo update
```

#### 验证安装
```bash
helm version
# 应该输出类似: version.BuildInfo{Version:"v3.x.x", ...}

helm search repo bitnami/nginx
# 应该能搜索到Nginx Chart
```

#### 环境准备：Minikube配置
为了本地学习和测试，推荐使用Minikube。

**安装Minikube:**
```bash
# macOS
brew install minikube

# 启动一个本地K8s集群
minikube start --driver=docker
```

### 1.3 第一个Helm应用

#### 使用Helm部署现有Chart
以部署Nginx为例：

```bash
# 安装Nginx Chart，并为Release命名为my-nginx
helm install my-nginx bitnami/nginx
```

#### 查看Release状态
```bash
# 列出所有Release
helm list

# 查看my-nginx的部署状态
helm status my-nginx
```
输出会提示你如何访问Nginx服务。

#### Release版本管理
每次升级都会创建一个新的版本号（Revision）。

```bash
# 查看my-nginx的所有历史版本
helm history my-nginx
```

#### 升级与回滚
假设我们要更改Nginx的Service类型为`LoadBalancer`。

**1. 查看可配置项:**
```bash
helm show values bitnami/nginx > values.yaml
```
编辑`values.yaml`文件，找到`service.type`并修改为`LoadBalancer`。

**2. 升级Release:**
```bash
helm upgrade my-nginx bitnami/nginx --values values.yaml
```

**3. 回滚到上一个版本:**
如果升级出现问题，可以轻松回滚。
```bash
# 回滚到版本1
helm rollback my-nginx 1
```

#### 卸载应用
```bash
helm uninstall my-nginx
```

---

## 第二部分：Chart 开发篇

### 2.1 Chart 结构详解

#### Chart目录结构
使用`helm create`命令可以快速创建一个标准的Chart目录。
```bash
helm create my-chart
```
生成的结构如下：
```
my-chart/
├── Chart.yaml          # Chart元数据
├── values.yaml         # 默认配置文件
├── charts/             # 存放依赖的子Chart
├── templates/          # K8s资源模板目录
│   ├── NOTES.txt       # 安装后显示的提示信息
│   ├── _helpers.tpl    # 可复用的模板片段
│   ├── deployment.yaml
│   ├── service.yaml
│   └── ...
└── .helmignore         # 忽略不需要打包的文件
```

#### 关键文件说明
- **Chart.yaml**: 定义了Chart的名称、版本、描述等信息。
  ```yaml
  apiVersion: v2
  name: my-chart
  description: A Helm chart for Kubernetes
  version: 0.1.0
  appVersion: "1.16.0" # 应用的版本
  ```
- **values.yaml**: 为模板提供默认值，是Chart配置的核心。
- **templates/**: 存放所有Kubernetes资源模板文件。Helm会渲染这个目录下的所有`.yaml`和`.tpl`文件。

#### Chart依赖管理
在`Chart.yaml`中定义依赖项：
```yaml
dependencies:
- name: mariadb
  version: "10.5.2"
  repository: "https://charts.bitnami.com/bitnami"
  condition: mariadb.enabled # 仅当values.yaml中mariadb.enabled为true时才安装
```
下载并更新依赖：
```bash
helm dependency update my-chart
```
依赖会被下载到`charts/`目录下。

### 2.2 Go Template 基础

Helm使用Go Template语言来渲染模板。

#### 模板语法入门
- **占位符**: `{{ .Values.key }}` 用于引用`values.yaml`中的值。
- **注释**: `{{- /* this is a comment */ -}}`
- **移除空白**: `-`可以加在`{{`或`}}`的旁边，用于移除渲染时产生的多余空行。例如`{{- ... -}}`。

#### 变量、对象与引用
- **` . `**: 代表当前作用域的对象。在顶层，` . `代表`values.yaml`。
- **` .Values `**: `values.yaml`中的所有内容。
- **` .Release `**: 关于Release的信息，如`.Release.Name`。
- **` .Chart `**: `Chart.yaml`中的内容。

#### 流程控制
- **if/else**:
  ```go-template
  {{- if .Values.service.enabled -}}
  # service content here
  {{- end -}}
  ```
- **with**: 修改`.`的作用域。
  ```go-template
  {{- with .Values.service }}
  # 这里 . 代表 .Values.service
  type: {{ .type }}
  port: {{ .port }}
  {{- end }}
  ```
- **range**: 遍历列表。
  ```go-template
  ports:
  {{- range .Values.ports }}
  - port: {{ . }}
    protocol: TCP
  {{- end }}
  ```

#### 管道与函数
Helm提供了大量内置函数，可以通过管道符`|`调用。
```go-template
name: {{ .Release.Name | trunc 63 | trimSuffix "-" }}
# quote: 加引号
# default: 提供默认值
# upper: 转大写
```

#### 命名模板
在`_helpers.tpl`中定义可复用的模板片段。

**定义:**
```go-template
{{/*
Define a common label block
*/}}
{{- define "my-chart.labels" -}}
app.kubernetes.io/name: {{ .Chart.Name }}
app.kubernetes.io/instance: {{ .Release.Name }}
{{- end -}}
```

**使用:**
```go-template
metadata:
  labels:
    {{- include "my-chart.labels" . | nindent 4 }}
```

### 2.3 编写K8s资源模板
实战：创建一个完整的应用Chart，包含Deployment, Service, ConfigMap。

`templates/deployment.yaml`:
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: {{ .Release.Name }}-deployment
spec:
  replicas: {{ .Values.replicaCount }}
  template:
    spec:
      containers:
        - name: {{ .Chart.Name }}
          image: "{{ .Values.image.repository }}:{{ .Values.image.tag }}"
```

`templates/service.yaml`:
```yaml
apiVersion: v1
kind: Service
metadata:
  name: {{ .Release.Name }}-service
spec:
  type: {{ .Values.service.type }}
  ports:
    - port: {{ .Values.service.port }}
  selector:
    app.kubernetes.io/name: {{ .Chart.Name }}
```

### 2.4 Values 文件管理

#### values.yaml详解
这是Chart的"API"，用户通过修改它来配置应用。

`values.yaml`:
```yaml
replicaCount: 1

image:
  repository: nginx
  tag: stable

service:
  type: ClusterIP
  port: 80
```

#### 多环境配置
可以为不同环境创建不同的values文件。

- `values-prod.yaml`
- `values-dev.yaml`

部署时指定：
```bash
# 开发环境
helm install dev-release my-chart -f values-dev.yaml

# 生产环境
helm install prod-release my-chart -f values-prod.yaml
```

#### Values覆盖机制
优先级从低到高：
1. `values.yaml` (Chart内部)
2. 父Chart的`values.yaml`传递给子Chart
3. `-f`或`--values`指定的values文件
4. `--set`命令行参数

#### 使用--set参数
临时覆盖单个值，非常适合CI/CD。
```bash
helm install my-release my-chart --set replicaCount=3 --set image.tag=latest
```

---

## 第三部分：Helm 高级篇

### 3.1 Hooks 机制
Hooks允许你在Release生命周期的特定时间点执行操作，例如在安装前备份数据库，或在升级后运行数据迁移。

#### Hooks类型与执行时机
- `pre-install`: 安装前
- `post-install`: 安装后
- `pre-delete`: 删除前
- `post-delete`: 删除后
- `pre-upgrade`: 升级前
- `post-upgrade`: 升级后
- `pre-rollback`: 回滚前
- `post-rollback`: 回滚后

在K8s资源中通过`annotations`定义Hook：
```yaml
apiVersion: batch/v1
kind: Job
metadata:
  name: "db-migration"
  annotations:
    # 这是一个Hook，在升级后执行
    "helm.sh/hook": "post-upgrade"
    # Hook权重，数字越小越先执行
    "helm.sh/hook-weight": "1"
    # 删除策略：在完成后删除
    "helm.sh/hook-delete-policy": "hook-succeeded"
```

### 3.2 测试与验证

#### Helm Tests
Helm提供了一种测试机制，用于验证部署是否成功。测试本身也是一个K8s资源（通常是Pod），定义在`templates/`目录下，文件名前缀为`test-`。

`templates/test-connection.yaml`:
```yaml
apiVersion: v1
kind: Pod
metadata:
  name: "{{ .Release.Name }}-test-connection"
  annotations:
    "helm.sh/hook": test
spec:
  containers:
    - name: wget
      image: busybox
      command: ['wget']
      args: ['{{ .Release.Name }}-service:{{ .Values.service.port }}']
  restartPolicy: Never
```

运行测试：
```bash
helm test my-release
```

#### Lint检查
在打包或安装前，检查Chart的语法和最佳实践。
```bash
helm lint my-chart
```

---

## 第四部分：CI/CD 集成篇

### 4.1 Helm 与 CI/CD

#### 在CI流程中使用Helm
- **打包**: `helm package my-chart`
- **Lint**: `helm lint my-chart`
- **测试**: `helm test my-release`
- **部署**: `helm upgrade --install my-release my-chart --namespace my-ns -f values.yaml`

#### GitOps与Helm
GitOps是一种现代化的CD实践，它以Git仓库作为唯一可信源。
- **Argo CD**: 持续监控Git仓库中的Helm Chart或K8s清单，并自动同步到集群。
- **Flux**: 另一个流行的GitOps工具，同样可以与Helm无缝集成。

### 4.2 实战：构建完整的CI/CD流程 (GitHub Actions)

`.github/workflows/deploy.yml`:
```yaml
name: Deploy to K8s

on:
  push:
    branches:
      - main

jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v2

    - name: Set up K8s context
      uses: azure/k8s-set-context@v1
      with:
        method: kubeconfig
        kubeconfig: ${{ secrets.KUBE_CONFIG }} # K8s配置存在Secrets中

    - name: Deploy with Helm
      run: |
        helm upgrade --install my-app ./my-chart \
          --namespace production \
          --set image.tag=${{ github.sha }}
```

---

## 第五部分：故障排查与最佳实践

### 5.1 常见问题排查
- **`helm install`失败**:
  - `helm status my-release`: 查看详细状态。
  - `kubectl get events -n my-ns`: 查看Kubernetes集群事件。
  - `kubectl logs <pod-name>`: 查看Pod日志。
- **模板渲染错误**:
  - `helm template my-chart`: 在本地渲染模板，不部署，用于调试。
  - `helm install --dry-run --debug`: 模拟安装并打印出所有渲染的YAML。

### 5.2 最佳实践
- **Chart设计**: 保持Chart的单一职责，避免一个Chart过于庞大。
- **Values文件**: 结构清晰，提供合理的默认值和文档注释。
- **模板可读性**: 善用`_helpers.tpl`和`with`语句简化模板。
- **版本管理**: 遵循语义化版本（SemVer）。
- **安全**:
  - 不要在Chart中硬编码Secret。使用`values.yaml`传递，或通过外部Secret管理工具。
  - 使用Helm 3，避免Tiller。
  - 配置RBAC权限。

---

## 附录

### A. 常用命令速查表
- `helm create [NAME]`: 创建一个新的Chart。
- `helm package [CHART_PATH]`: 打包一个Chart。
- `helm install [RELEASE] [CHART]`: 安装一个Chart。
- `helm uninstall [RELEASE]`: 卸载一个Release。
- `helm list`: 列出所有Release。
- `helm status [RELEASE]`: 显示一个Release的状态。
- `helm upgrade [RELEASE] [CHART]`: 升级一个Release。
- `helm rollback [RELEASE] [REVISION]`: 回滚一个Release。
- `helm history [RELEASE]`: 查看Release的历史版本。
- `helm repo add [NAME] [URL]`: 添加一个Chart仓库。
- `helm repo update`: 更新仓库信息。
- `helm search repo [KEYWORD]`: 搜索仓库中的Chart。
- `helm lint [PATH]`: 检查Chart的语法。
- `helm test [RELEASE]`: 运行Release的测试。

### B. Go Template 语法速查
- `{{ .Values.some.value | default "default-value" }}
- `{{- if and .Values.foo .Values.bar }}`
- `{{- range $index, $value := .Values.myList }}`
- `{{- include "my-chart.labels" . | nindent 4 }}`
- `{{- required "A valid foo is required!" .Values.foo }}`
- `{{- toYaml .Values.someObject }}`
