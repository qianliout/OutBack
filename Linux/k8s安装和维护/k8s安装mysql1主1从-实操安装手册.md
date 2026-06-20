---
title: K8s 安装 MySQL 1主1从实操安装手册
tags:
  - k8s
  - kubernetes
  - mysql
  - replication
  - statefulset
aliases:
  - K8s MySQL 1主1从实操
  - MySQL 主从实操安装手册
---

# K8s 安装 MySQL 1主1从实操安装手册

## 1. 这次实操的最终结论

这次实际安装没有完全按最初设计文档里的 `Helm + replication` 路线落地。

真实原因不是方案本身有问题，而是当前实验环境里遇到了镜像源和运行时层面的连续阻塞，主要包括：

1. `Bitnami MySQL` 镜像在当前网络环境下不可稳定拉取
2. `docker.m.daocloud.io/bitnami/mysql` 返回 `403 Forbidden`
3. `docker.io/bitnami/mysql` 在当前环境多次超时
4. `local-path-provisioner` 和 helper pod 也出现过镜像拉取和 sandbox 问题

所以这次最终落地的实际路径改成了：

1. 保留 K8s 双节点集群
2. 保留 `local-path` 作为默认存储
3. 放弃继续硬顶 `Helm + Bitnami MySQL`
4. 改用官方 `mysql:8.4`
5. 使用 `StatefulSet + Service + PVC + ConfigMap + Secret` 方式搭建 `1 主 1 从`
6. 最终已经验证主从复制正常工作

## 2. 当前最终状态

最终状态如下：

1. `mysql-primary-0` 已经 `Running`
2. `mysql-secondary-0` 已经 `Running`
3. 两个 PVC 都已经 `Bound`
4. `SHOW REPLICA STATUS` 显示：
5. `Replica_IO_Running: Yes`
6. `Replica_SQL_Running: Yes`
7. `Seconds_Behind_Source: 0`
8. 主库写入测试数据后，从库已经成功读到相同数据

最终检查命令：

```bash
ssh k8s-master-01
kubectl -n mysql get pods,svc,pvc -o wide
```

最终检查结果：

```text
NAME                    READY   STATUS    RESTARTS   AGE
pod/mysql-primary-0     1/1     Running   1          16m
pod/mysql-secondary-0   1/1     Running   0          115s

NAME                                           STATUS   STORAGECLASS
persistentvolumeclaim/data-mysql-primary-0     Bound    local-path
persistentvolumeclaim/data-mysql-secondary-0   Bound    local-path
```

## 3. 环境信息

本次实操基于下面这套环境：

1. 本地主机：`macOS`
2. 宿主平台：`MacBook M1`
3. 虚拟化环境：`OrbStack`
4. K8s 节点：
5. `k8s-master-01`
6. `k8s-worker-01`
7. 容器运行时：`containerd`
8. 存储类：`local-path`
9. MySQL 镜像：`docker.io/library/mysql:8.4`

## 4. 原始计划

最开始的计划不是手工写 MySQL 主从 YAML，而是：

1. 使用 `Helm`
2. 使用成熟 `MySQL Chart`
3. 使用 `replication` 模式创建 `1 主 1 从`

当时这样设计的原因很明确：

1. 更快
2. 更标准
3. 后续升级更方便
4. 也更接近日常 K8s 中间件安装习惯

对应的设计说明见：

1. [[k8s安装mysql1主1从方案设计]]

## 5. 前置准备过程

### 5.1 检查集群

先登录 `master` 节点：

```bash
ssh k8s-master-01
```

检查节点状态：

```bash
kubectl get nodes -o wide
kubectl get pods -A -o wide
```

目标是确认：

1. 两个节点都 `Ready`
2. `kube-system` 核心组件正常
3. 网络插件正常

### 5.2 检查 Helm

先检查 `master` 节点上是否已有 Helm：

```bash
helm version
```

一开始没有可直接使用的 Helm。

因为访问 GitHub 安装脚本不稳定，所以没有走在线脚本方式，而是改成直接下载 Helm 二进制安装。

### 5.3 检查 StorageClass

检查命令：

```bash
kubectl get storageclass
```

当时结果是：

```text
No resources found
```

这意味着 Helm 就算能装 Chart，PVC 也会一直 `Pending`，所以必须先解决存储类。

### 5.4 安装 local-path

后续在集群里补了 `local-path-provisioner`，并让它成为默认存储类。

中间遇到的问题很多，主要是：

1. `rancher/local-path-provisioner` 镜像拉取失败
2. helper pod 使用的 `busybox` 镜像不稳定
3. 期间还出现过 `FailedCreatePodSandBox`
4. `cgroupsPath` 格式错误

最终通过：

1. 预拉镜像
2. 重打标签
3. 调整 helper pod 镜像
4. 重启 `containerd`
5. 重启 `kubelet`

才把 `local-path` 恢复到可用状态。

## 6. Helm 路线的实际尝试过程

### 6.1 安装 Helm

因为 GitHub raw 脚本访问不稳定，所以最后使用二进制安装 Helm。

### 6.2 获取 MySQL Chart

一开始尝试的是 Bitnami MySQL Chart。

为了绕开 Docker 官方源问题，尝试过：

1. `oci://docker.m.daocloud.io/bitnamicharts/mysql`
2. `docker.io`
3. 其他代理源

### 6.3 编写 values 文件

在 `k8s-master-01` 上创建过：

1. `~/mysql-replication-values.yaml`
2. `~/.mysql-replication-secrets.env`

values 文件里已经启用了：

1. `architecture: replication`
2. `secondary.replicaCount: 1`
3. `local-path` 存储类
4. 主库固定到 `k8s-master-01`
5. 从库固定到 `k8s-worker-01`

### 6.4 Helm 路线失败原因

虽然 Chart 元数据和 values 基本能拿到，但真正阻塞点是镜像。

关键失败点如下：

1. `docker.m.daocloud.io/bitnami/mysql:*` 返回 `403 Forbidden`
2. `docker.io/bitnami/mysql:*` 在当前网络环境下多次超时
3. `allowInsecureImages: true` 只能绕过 Chart 的镜像校验
4. 不能解决镜像实际不可拉取的问题

所以这一步最终判断为：

1. 不是 Helm 不行
2. 不是 replication 方案有问题
3. 是当前实验环境无法稳定获取 Bitnami MySQL 镜像

于是正式决定切换路线。

## 7. 改用官方 mysql:8.4 的原因

切换到官方镜像的核心原因只有一个：

1. 官方 `mysql:8.4` 在当前环境最终可以稳定拉取

相比继续卡在 Bitnami 上，改用官方镜像更符合这次实操目标：

1. 先把服务跑起来
2. 先把主从关系打通
3. 先把全过程记录完整

## 8. 第一次官方镜像方案

第一次官方镜像方案的核心资源如下：

1. `Namespace mysql`
2. `Secret mysql-auth`
3. `ConfigMap mysql-primary-config`
4. `ConfigMap mysql-secondary-config`
5. 主库 `StatefulSet`
6. 从库 `StatefulSet`
7. 主从 `Service`
8. 两个 `PVC`

第一次方案里的关键思路是：

1. 主库写 binlog
2. 从库通过 `postStart` 脚本自动执行 `CHANGE REPLICATION SOURCE TO`
3. 主库和从库都启用 GTID

## 9. 第一次官方镜像方案踩到的问题

### 9.1 `default_authentication_plugin` 不兼容

主库和从库最开始都因为下面这行配置失败：

```cnf
default_authentication_plugin=caching_sha2_password
```

在 `mysql:8.4` 中，这行会导致：

```text
unknown variable 'default_authentication_plugin=caching_sha2_password'
```

处理方式：

1. 从主库和从库的 `my.cnf` 中删除这行
2. 重新部署 MySQL 资源

### 9.2 变量展开导致密码丢失

第一次生成 YAML 的过程中，遇到了 here-doc 和 shell 转义问题。

典型表现如下：

1. `init.sql` 里的复制密码丢了
2. `postStart` 脚本里的复制密码也丢了
3. 最终虽然 YAML 创建成功，但复制用户密码实际上没有被正确写入

处理方式：

1. 不再继续用不稳定的远端 here-doc 拼复杂脚本
2. 改成本地先生成修复文件
3. 再通过 `scp` 下发到 `master`

### 9.3 从库初始化阶段被 `read_only` 阻塞

从库最初把下面两项直接写进了 `my.cnf`：

```cnf
read_only=ON
super_read_only=ON
```

这会导致官方 MySQL 镜像在初始化临时实例时无法写系统表，日志里报：

```text
The MySQL server is running with the --super-read-only option so it cannot execute this statement
```

处理方式：

1. 从从库 `my.cnf` 中移除 `read_only`
2. 从从库 `my.cnf` 中移除 `super_read_only`
3. 等复制关系建立成功后，再在 SQL 中执行：

```sql
SET GLOBAL read_only=ON;
SET GLOBAL super_read_only=ON;
```

### 9.4 主库复制用户没有真正创建成功

虽然主库已经能启动，但主库初始化脚本里也受过一次转义污染。

结果是：

1. `replicator` 用户没有被正确初始化
2. 从库日志报：

```text
Access denied for user 'replicator'@'192.168.36.217'
```

处理方式：

1. 手工登录主库容器
2. 执行 SQL 显式创建复制用户
3. 重新授权
4. 刷新权限

实际补救 SQL：

```sql
CREATE USER IF NOT EXISTS 'replicator'@'%' IDENTIFIED BY '<replication-password>';
ALTER USER 'replicator'@'%' IDENTIFIED BY '<replication-password>';
GRANT REPLICATION SLAVE, REPLICATION CLIENT ON *.* TO 'replicator'@'%';
FLUSH PRIVILEGES;
```

### 9.5 从库访问主库 Service 的地址需要改成 FQDN

从库最初配置的是：

```sql
SOURCE_HOST='mysql-primary'
```

后续在从库里检查 `SHOW REPLICA STATUS\G` 时看到：

```text
Last_IO_Error: Unknown MySQL server host 'mysql-primary' (-2)
```

处理方式：

把主库地址改成集群内完整域名：

```sql
SOURCE_HOST='mysql-primary.mysql.svc.cluster.local'
```

## 10. 最终修复步骤

### 10.1 删除失败安装

因为前一轮安装已经留下了错误初始化的数据目录，所以没有在原地硬修，而是先删除失败资源后重建。

实际操作思路：

1. 删除 `mysql` namespace
2. 重新创建 `mysql` namespace
3. 重新创建 `mysql-auth` secret
4. 重新 apply MySQL 资源

### 10.2 重建主库和从库

重新部署后，先让主库正常启动。

确认主库启动命令：

```bash
kubectl -n mysql logs mysql-primary-0 --tail=100
kubectl -n mysql get pod mysql-primary-0 -o wide
```

确认点：

1. 主库必须先 `Running`
2. 主库必须能正常接受本地 root 连接

### 10.3 单独修复从库

从库因为初始化和复制逻辑更复杂，后续做了单独修复：

1. 更新从库 `ConfigMap`
2. 更新从库 `StatefulSet`
3. 删除旧 Pod
4. 删除旧 PVC
5. 重新让 `StatefulSet` 创建新 Pod

### 10.4 手工补建复制账号

在主库中执行：

```bash
kubectl -n mysql exec -i mysql-primary-0 -- sh -lc 'mysql -uroot -p"$MYSQL_ROOT_PASSWORD"'
```

然后执行：

```sql
CREATE USER IF NOT EXISTS 'replicator'@'%' IDENTIFIED BY '<replication-password>';
ALTER USER 'replicator'@'%' IDENTIFIED BY '<replication-password>';
GRANT REPLICATION SLAVE, REPLICATION CLIENT ON *.* TO 'replicator'@'%';
FLUSH PRIVILEGES;
```

### 10.5 手工修正从库 Source 地址

在从库中执行：

```sql
STOP REPLICA;
CHANGE REPLICATION SOURCE TO
  SOURCE_HOST='mysql-primary.mysql.svc.cluster.local',
  SOURCE_PORT=3306,
  SOURCE_USER='replicator',
  SOURCE_PASSWORD='<replication-password>',
  SOURCE_AUTO_POSITION=1,
  GET_SOURCE_PUBLIC_KEY=1;
START REPLICA;
```

## 11. 最终验证过程

### 11.1 检查主从 Pod

```bash
kubectl -n mysql get pods -o wide
```

### 11.2 检查复制状态

```bash
kubectl -n mysql exec mysql-secondary-0 -- sh -lc 'mysql -uroot -p"$MYSQL_ROOT_PASSWORD" -e "SHOW REPLICA STATUS\\G"'
```

关键结果：

```text
Replica_IO_Running: Yes
Replica_SQL_Running: Yes
Seconds_Behind_Source: 0
Source_Host: mysql-primary.mysql.svc.cluster.local
```

### 11.3 主库写入测试

在主库中执行：

```sql
CREATE DATABASE IF NOT EXISTS appdb;
CREATE TABLE IF NOT EXISTS appdb.replication_check (
  id INT PRIMARY KEY AUTO_INCREMENT,
  val VARCHAR(64) NOT NULL,
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
INSERT INTO appdb.replication_check(val) VALUES ('check-20260620-1');
SELECT id, val, created_at FROM appdb.replication_check ORDER BY id DESC LIMIT 3;
```

### 11.4 从库读取验证

在从库中执行：

```sql
SELECT id, val, created_at FROM appdb.replication_check ORDER BY id DESC LIMIT 3;
```

本次最终验证结果：

```text
id  val               created_at
2   check-20260620-1  2026-06-20 05:30:09
1   check-20260620-1  2026-06-20 05:29:12
```

说明：

1. 主库写入成功
2. 从库已经收到并执行复制事件
3. 主从复制链路已经打通

## 12. 这次实操里最重要的经验

这次最有价值的经验不是“命令本身”，而是下面这些判断：

1. 设计方案和真实落地方案可以不同
2. 遇到镜像源阻塞时，不要一直卡在原方案上
3. 本地双节点实验环境里，能先跑通比坚持某个工具链更重要
4. 官方镜像的初始化逻辑和手工 `my.cnf` 配置会互相影响
5. `read_only` 这类参数不能想当然直接提前打开
6. 带变量的 here-doc 很容易把 SQL 和密码弄坏
7. 需要区分“Pod Running”和“复制真正健康”是两回事

## 13. 当前还不算解决的部分

这次虽然主从已经跑通，但还不等于生产级高可用。

当前还没有做的事情包括：

1. 自动故障切换
2. 自动备份
3. 监控告警
4. 慢查询分析
5. 半同步复制
6. Operator 化管理

## 14. 后续建议

下一步更建议你按下面顺序继续：

1. 先把当前这套主从结构完全理解清楚
2. 再把最终工作中的 YAML 整理成一份稳定版本
3. 再补备份方案
4. 再补监控
5. 最后再回头看 `Operator`

## 15. 关联笔记

相关笔记如下：

1. [[k8s安装mysql1主1从方案设计]]
2. [[mac使用orbstack搭建k8s集群]]
3. [[mac使用orbstack搭建k8s集群-实操安装手册]]
