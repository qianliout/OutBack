---
title: K8s 安装 MySQL 1主1从方案设计
tags:
  - k8s
  - kubernetes
  - mysql
  - helm
  - replication
aliases:
  - K8s MySQL Helm 主从方案
  - MySQL Helm replication 方案
---

# K8s 安装 MySQL 1主1从方案设计

## 0. 先说明这篇文档的定位

这篇文档记录的是当时的设计思路。

当时我的目标是优先采用：

1. `Helm`
2. `replication`
3. 成熟 MySQL Chart

但是后续真实安装过程中，因为 `Bitnami MySQL` 镜像源在当前环境里不可稳定使用，最终实操并没有完全按这篇设计文档落地，而是切换到了：

1. 官方 `mysql:8.4`
2. `StatefulSet + Service + PVC + ConfigMap + Secret`
3. 手工搭建 `1 主 1 从`

真实安装全过程、失败路线和最终修复记录，见：

1. [[k8s安装mysql1主1从-实操安装手册]]

## 1. 这次方案的结论

这次不再采用手工编写 `StatefulSet + ConfigMap + Secret + Service + 主从初始化脚本` 的方案，而是改成：

1. 使用 `Helm`
2. 使用成熟的 `MySQL Helm Chart`
3. 使用 `replication` 模式部署 `1 主 1 从`

当前阶段的目标不是研究每一个底层 YAML 细节，而是先用一条更快、更稳、更容易维护的路径，把 MySQL 主从服务跑在 K8s 中。

## 2. 为什么这次改成 Helm + replication

当前这套方案更适合你现在的阶段，原因如下：

1. 你已经有可用的 K8s 集群。
2. 你现在的重点是先把 MySQL 主从服务部署起来。
3. 你希望兼顾速度、可维护性和后续继续学习的空间。
4. `Helm Chart` 已经把很多重复的 YAML 和初始化逻辑封装好了。

如果继续手工写整套主从 YAML，虽然学习价值更高，但会明显拖慢落地速度。

如果一上来就上 `Operator`，又会引入更多控制器、CRD 和排障复杂度。

所以当前阶段最合适的折中方案就是：

1. 不手搓全部资源。
2. 不直接上 Operator。
3. 使用 Helm 的 `replication` 模式先落地。

## 3. 当前环境检查结果

我已经通过 `ssh k8s-master-01` 在集群里做了基础检查。

当前确认结果如下：

1. 两个节点都已经 `Ready`
2. `master` 和 `worker` 都正常加入集群
3. 当前没有查到 `StorageClass`
4. 当前 `master` 上没有看到可直接使用的 `helm version` 输出

本次实际检查结果如下：

```text
NAME            STATUS   ROLES           AGE   VERSION    INTERNAL-IP       CONTAINER-RUNTIME
k8s-master-01   Ready    control-plane   9h    v1.33.13   192.168.139.167   containerd://2.2.1
k8s-worker-01   Ready    <none>          9h    v1.33.13   192.168.139.177   containerd://2.2.1
```

当前 `StorageClass` 检查结果：

```text
No resources found
```

这意味着一件非常关键的事：

1. 现在还不能直接安装 MySQL
2. 必须先补齐存储类
3. 否则 Helm 创建出来的 PVC 会一直 `Pending`

## 4. 当前需求怎么落地最合适

如果你的目标是：

1. 快速安装
2. 后续可维护
3. 结构是 `1 主 1 从`
4. 可以继续演进

那当前最合适的路线是：

1. 先安装 `Helm`
2. 先补一个默认 `StorageClass`
3. 再使用支持复制模式的 MySQL Chart 安装
4. 让 Chart 自动帮我们创建主库、副本、PVC 和 Service

这条路线相对手工方案的优势：

1. 安装更快
2. 参数集中
3. 升级更方便
4. 删除和重建更规范
5. 更适合后面整理成标准化实操文档

## 5. 推荐的 Chart 思路

当前更推荐采用“成熟社区 Chart + replication 模式”的方案。

这里的核心要求不是 Chart 名字本身，而是它必须支持下面这些能力：

1. 支持 `architecture=replication`
2. 支持主从副本数量配置
3. 支持持久化卷
4. 支持主库和从库 Service
5. 支持通过 Helm values 管理密码、资源限制和存储大小

这类 Chart 通常会帮我们处理这些事情：

1. 创建主实例
2. 创建从实例
3. 创建 PVC
4. 创建主从访问 Service
5. 初始化复制关系

## 6. 这套方案和手工 StatefulSet 方案的区别

这次不是否定手工 `StatefulSet` 方案，而是换一个更适合当前目标的落地路径。

两者区别可以简单理解为：

1. 手工方案更偏“学习原理”
2. Helm 方案更偏“快速落地”

手工方案更适合：

1. 想彻底搞懂主从初始化过程
2. 想理解每个 YAML 资源的职责
3. 想做高度定制化主从结构

Helm 方案更适合：

1. 想先尽快跑起来
2. 想减少重复 YAML 编写
3. 想后续更方便升级和回滚

## 7. 为什么现在不优先上 Operator

在生产环境里，有些团队会使用 `Operator` 管理 MySQL。

但对你当前这个环境，我仍然不建议一开始就走 `Operator` 路线。

原因如下：

1. 你当前是本地双节点实验型集群。
2. 你现在优先目标是把 MySQL 主从跑起来。
3. `Operator` 会引入额外 CRD、Controller 和更复杂的排障路径。
4. `Operator` 更适合在你已经理解基本部署逻辑后再引入。

所以当前更合理的顺序是：

1. 先 Helm
2. 再实操
3. 再验证主从
4. 最后再考虑是否需要 Operator

## 8. 预期架构

这次 Helm + replication 方案的预期架构如下：

1. 一个 MySQL 主库 Pod
2. 一个 MySQL 从库 Pod
3. 每个 Pod 都有独立 PVC
4. 主从通过集群内 DNS 互相发现
5. 业务通过不同 Service 区分读写访问

逻辑上可以理解为：

1. 主库负责写入和 binlog 产生
2. 从库负责复制主库数据
3. 主库 Service 用于写流量
4. 从库 Service 用于读流量

## 9. 这套方案真正依赖的前置条件

在执行 Helm 安装前，必须满足下面几个前提。

### 9.1 集群节点健康

至少要满足：

1. 所有目标节点 `Ready`
2. `kube-system` 核心组件正常
3. CNI 正常

你当前这部分已经满足。

### 9.2 必须有可用的 StorageClass

这是当前最关键的阻塞项。

如果没有 `StorageClass`，那 Helm 虽然能把资源创建出来，但：

1. PVC 会 `Pending`
2. Pod 会卡住
3. MySQL 不会真正启动成功

所以正式安装 MySQL 之前，必须先解决：

```bash
kubectl get storageclass
```

当前你的结果是：

```text
No resources found
```

### 9.3 Helm 工具可用

这次方案依赖 Helm，因此需要先满足：

1. `helm` 命令已安装
2. `helm repo` 可用
3. 能正常拉取 Chart

当前检查没有看到有效的 `helm version` 输出，因此这一步也需要先补。

### 9.4 镜像可拉取

你前面安装 Calico 时已经遇到过镜像拉取超时问题。

所以 MySQL 方案里也要提前考虑：

1. Chart 引用的 MySQL 镜像是否可拉取
2. 如果镜像源慢，是否需要预拉
3. 是否需要后续替换镜像仓库

## 10. 当前阶段的最优目标

这篇方案文档里，我建议先把目标限定为下面这些：

1. 使用 Helm 成功部署 `1 主 1 从`
2. 主库可写
3. 从库可同步
4. 数据走 PVC 持久化
5. 可以通过 Service 区分主从访问

当前阶段先不追求下面这些能力：

1. 自动主从切换
2. 自动故障转移
3. 自动备份恢复
4. 多副本横向扩展
5. Operator 化管理

## 11. 当前方案的边界

即使 Helm 已经帮我们快速装好了 `1 主 1 从`，也不意味着这就是完整的生产级数据库高可用方案。

它的边界主要在这里：

1. 主库挂了，不会自动提升从库
2. 从库延迟仍要单独监控
3. 数据安全仍然依赖存储和备份策略
4. 重建 Pod 后，仍要验证复制状态是否健康
5. 双节点实验环境不等于真正生产高可用环境

所以你要把当前阶段的目标理解为：

1. 先把 MySQL 以更标准、更快捷的方式运行在 K8s 中
2. 先把主从结构、持久化和访问方式跑通
3. 后续再逐步增强备份、监控和故障切换能力

## 12. 这次方案里最容易踩的坑

### 12.1 没有 StorageClass 就直接安装

这是当前最现实的问题。

如果直接执行 Helm 安装，大概率会出现：

1. PVC 一直 `Pending`
2. Pod 一直 `Pending` 或 `ContainerCreating`

### 12.2 Helm 已装，但镜像拉不下来

这在你当前环境里不能忽略。

MySQL Chart 如果依赖的镜像仓库访问慢，也可能出现：

1. `ImagePullBackOff`
2. `ErrImagePull`

### 12.3 把主从访问混用

后面业务使用时不能把所有连接都接到同一个入口。

否则就可能出现：

1. 写请求误打到只读实例
2. 读写分离逻辑混乱

### 12.4 误以为 Helm 方案等于生产级高可用

Helm 只是让安装和管理更方便。

它解决的是：

1. 更快部署
2. 更规范管理

它没有自动解决：

1. 故障转移策略
2. 备份恢复体系
3. 性能压测与容量规划

## 13. 后续文档应该怎么拆

既然这次已经把路线改成 `Helm + replication`，后续文档结构也要跟着调整。

### 13.1 第一篇：当前这篇方案设计

重点解决：

1. 为什么选 Helm
2. 为什么不用手工 YAML
3. 为什么当前不先上 Operator

### 13.2 第二篇：环境准备

重点包括：

1. 安装 Helm
2. 准备 Namespace
3. 检查或补齐 StorageClass
4. 检查镜像拉取能力

### 13.3 第三篇：Helm 安装 MySQL 1主1从实操

重点包括：

1. 添加 Helm 仓库
2. 编写 values 文件
3. 开启 replication 模式
4. 配置主从副本数
5. 安装和验证

### 13.4 第四篇：验证与排障

重点包括：

1. Pod 状态检查
2. PVC 检查
3. Service 检查
4. 主从复制检查
5. 镜像拉取问题排查

## 14. 当前结论

结合你当前的环境，我认为最合理的路线已经变成：

1. 先补齐 `StorageClass`
2. 先安装 `Helm`
3. 再用 `Helm + replication` 模式安装 MySQL
4. 让 Chart 帮我们完成主从部署、PVC 和 Service 创建
5. 再做主从同步验证和后续优化

这条路线比手工写整套主从 YAML 更适合你当前阶段，因为它同时兼顾：

1. 安装速度
2. 实操成功率
3. 后续维护便利性

## 15. 下一步建议

现在最合适的下一步不是直接开始写 MySQL 安装命令，而是先补下面两件事：

1. 在集群里补一个可用的 `StorageClass`
2. 在 `master` 节点安装 `Helm`

等这两个前提满足后，再继续写下一篇文档：

1. `K8s 使用 Helm 安装 MySQL 1主1从环境准备`

然后再写：

1. `K8s 使用 Helm 安装 MySQL 1主1从实操步骤`

## 16. 关联笔记

当前 K8s 集群来源于：

1. [[mac使用orbstack搭建k8s集群]]
2. [[mac使用orbstack搭建k8s集群-实操安装手册]]
