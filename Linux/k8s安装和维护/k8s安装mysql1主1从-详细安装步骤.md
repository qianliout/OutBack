---
title: K8s 安装 MySQL 1主1从详细安装步骤
tags:
  - k8s
  - kubernetes
  - mysql
  - replication
  - statefulset
aliases:
  - K8s MySQL 1主1从详细步骤
  - MySQL 主从一步一步安装
---

# K8s 安装 MySQL 1主1从详细安装步骤

## 1. 这篇文档的目标

这篇文档不是方案讨论，也不是过程复盘，而是给你一份可以直接照着执行的安装手册。

你按这篇文档做，目标是完成下面几件事：

1. 在已经可用的 K8s 集群里安装 MySQL `1 主 1 从`
2. 主库固定运行在 `k8s-master-01`
3. 从库固定运行在 `k8s-worker-01`
4. 使用 `local-path` 作为存储类
5. 主从复制自动建立
6. 最后能验证主库写入、从库同步成功

## 2. 前提条件

开始前，请先确认下面这些前提已经满足：

1. 你已经有一个可用的 K8s 集群
2. 你可以执行 `ssh k8s-master-01`
3. 集群里已经有可用的 `StorageClass`
4. 当前默认存储类是 `local-path`
5. 两个节点名字分别是：
6. `k8s-master-01`
7. `k8s-worker-01`

先登录 master：

```bash
ssh k8s-master-01
```

执行下面这些检查命令：

```bash
kubectl get nodes -o wide
kubectl get storageclass
kubectl get pods -A -o wide
```

你至少要看到下面这些结果：

1. `k8s-master-01` 是 `Ready`
2. `k8s-worker-01` 是 `Ready`
3. `local-path` 存在
4. `kube-system` 里的核心 Pod 基本都是 `Running`

如果 `local-path` 还没有准备好，不要继续往下装 MySQL，否则 PVC 会一直 `Pending`。

## 3. 安装思路

这次不使用 Helm，而是使用一套稳定可控的方式：

1. `Namespace`
2. `Secret`
3. `ConfigMap`
4. `Service`
5. `StatefulSet`
6. `PVC`

这样做的原因很简单：

1. 当前环境里官方 `mysql:8.4` 镜像可用
2. 这套方式更容易完全记录和复现
3. 你后续自己动手时，每个步骤都能看清楚

## 4. 第一步：准备工作目录

在 `master` 节点上创建一个工作目录：

```bash
mkdir -p ~/mysql-replication-install
cd ~/mysql-replication-install
pwd
```

建议后续所有文件都放在这个目录里，方便你以后重装或排查。

## 5. 第二步：生成密码并保存

先生成三组密码：

1. root 密码
2. 业务用户密码
3. 复制用户密码

执行：

```bash
cat > ~/.mysql-replication-secrets.env <<EOF
MYSQL_ROOT_PASSWORD=$(openssl rand -hex 12)
MYSQL_APP_PASSWORD=$(openssl rand -hex 12)
MYSQL_REPLICATION_PASSWORD=$(openssl rand -hex 12)
EOF
```

查看文件内容：

```bash
cat ~/.mysql-replication-secrets.env
```

加载这些环境变量：

```bash
source ~/.mysql-replication-secrets.env
```

确认变量已经生效：

```bash
echo "$MYSQL_ROOT_PASSWORD"
echo "$MYSQL_APP_PASSWORD"
echo "$MYSQL_REPLICATION_PASSWORD"
```

注意：

1. 这三个密码后面都会被用到
2. 这个文件不要随便删除
3. 后面验证主从时还要继续用

## 6. 第三步：如果之前装过，先清理旧资源

如果这是第一次安装，可以跳过这一步。

如果你之前已经装过失败版本，建议先清理干净，避免旧 PVC 里的坏数据影响新的安装。

执行：

```bash
kubectl delete namespace mysql --ignore-not-found=true
kubectl wait --for=delete namespace/mysql --timeout=180s || true
```

确认命名空间已经删掉：

```bash
kubectl get ns
```

如果输出里已经没有 `mysql`，说明旧资源已经清理完成。

## 7. 第四步：创建命名空间

执行：

```bash
kubectl create namespace mysql
kubectl get ns mysql
```

预期结果：

```text
NAME    STATUS   AGE
mysql   Active   ...
```

## 8. 第五步：创建 MySQL 密码 Secret

执行：

```bash
source ~/.mysql-replication-secrets.env

kubectl -n mysql create secret generic mysql-auth \
  --from-literal=mysql-root-password="$MYSQL_ROOT_PASSWORD" \
  --from-literal=mysql-app-password="$MYSQL_APP_PASSWORD" \
  --from-literal=mysql-replication-password="$MYSQL_REPLICATION_PASSWORD"
```

检查 Secret：

```bash
kubectl -n mysql get secret mysql-auth
kubectl -n mysql describe secret mysql-auth
```

你应该能看到：

1. `mysql-auth` 已创建
2. `DATA` 数量为 `3`

## 9. 第六步：写 MySQL 主从资源文件

在 `master` 节点上创建安装文件：

```bash
cat > ~/mysql-replication-install/mysql-replication.yaml <<'EOF'
apiVersion: v1
kind: ConfigMap
metadata:
  name: mysql-primary-config
  namespace: mysql
data:
  my.cnf: |
    [mysqld]
    server-id=1
    log_bin=mysql-bin
    binlog_format=ROW
    gtid_mode=ON
    enforce_gtid_consistency=ON
  init-primary.sh: |
    #!/bin/sh
    set -eu
    mysql -uroot -p"${MYSQL_ROOT_PASSWORD}" <<SQL
    CREATE USER IF NOT EXISTS 'replicator'@'%' IDENTIFIED BY '${MYSQL_REPLICATION_PASSWORD}';
    ALTER USER 'replicator'@'%' IDENTIFIED BY '${MYSQL_REPLICATION_PASSWORD}';
    GRANT REPLICATION SLAVE, REPLICATION CLIENT ON *.* TO 'replicator'@'%';
    FLUSH PRIVILEGES;
    SQL
---
apiVersion: v1
kind: ConfigMap
metadata:
  name: mysql-secondary-config
  namespace: mysql
data:
  my.cnf: |
    [mysqld]
    server-id=2
    relay_log=relay-bin
    gtid_mode=ON
    enforce_gtid_consistency=ON
---
apiVersion: v1
kind: Service
metadata:
  name: mysql-primary-headless
  namespace: mysql
spec:
  clusterIP: None
  selector:
    app: mysql-primary
  ports:
    - name: mysql
      port: 3306
      targetPort: 3306
---
apiVersion: v1
kind: Service
metadata:
  name: mysql-primary
  namespace: mysql
spec:
  selector:
    app: mysql-primary
  ports:
    - name: mysql
      port: 3306
      targetPort: 3306
---
apiVersion: v1
kind: Service
metadata:
  name: mysql-secondary-headless
  namespace: mysql
spec:
  clusterIP: None
  selector:
    app: mysql-secondary
  ports:
    - name: mysql
      port: 3306
      targetPort: 3306
---
apiVersion: v1
kind: Service
metadata:
  name: mysql-secondary
  namespace: mysql
spec:
  selector:
    app: mysql-secondary
  ports:
    - name: mysql
      port: 3306
      targetPort: 3306
---
apiVersion: apps/v1
kind: StatefulSet
metadata:
  name: mysql-primary
  namespace: mysql
spec:
  serviceName: mysql-primary-headless
  replicas: 1
  selector:
    matchLabels:
      app: mysql-primary
  template:
    metadata:
      labels:
        app: mysql-primary
    spec:
      nodeSelector:
        kubernetes.io/hostname: k8s-master-01
      tolerations:
        - key: node-role.kubernetes.io/control-plane
          operator: Exists
          effect: NoSchedule
      containers:
        - name: mysql
          image: docker.io/library/mysql:8.4
          imagePullPolicy: IfNotPresent
          ports:
            - containerPort: 3306
              name: mysql
          env:
            - name: MYSQL_ROOT_PASSWORD
              valueFrom:
                secretKeyRef:
                  name: mysql-auth
                  key: mysql-root-password
            - name: MYSQL_DATABASE
              value: appdb
            - name: MYSQL_USER
              value: appuser
            - name: MYSQL_PASSWORD
              valueFrom:
                secretKeyRef:
                  name: mysql-auth
                  key: mysql-app-password
            - name: MYSQL_REPLICATION_PASSWORD
              valueFrom:
                secretKeyRef:
                  name: mysql-auth
                  key: mysql-replication-password
            - name: MYSQL_ROOT_HOST
              value: "%"
          volumeMounts:
            - name: data
              mountPath: /var/lib/mysql
            - name: primary-config
              mountPath: /etc/mysql/conf.d/my.cnf
              subPath: my.cnf
            - name: primary-init
              mountPath: /docker-entrypoint-initdb.d/init-primary.sh
              subPath: init-primary.sh
      volumes:
        - name: primary-config
          configMap:
            name: mysql-primary-config
            items:
              - key: my.cnf
                path: my.cnf
        - name: primary-init
          configMap:
            name: mysql-primary-config
            defaultMode: 0755
            items:
              - key: init-primary.sh
                path: init-primary.sh
  volumeClaimTemplates:
    - metadata:
        name: data
      spec:
        accessModes: ["ReadWriteOnce"]
        storageClassName: local-path
        resources:
          requests:
            storage: 8Gi
---
apiVersion: apps/v1
kind: StatefulSet
metadata:
  name: mysql-secondary
  namespace: mysql
spec:
  serviceName: mysql-secondary-headless
  replicas: 1
  selector:
    matchLabels:
      app: mysql-secondary
  template:
    metadata:
      labels:
        app: mysql-secondary
    spec:
      nodeSelector:
        kubernetes.io/hostname: k8s-worker-01
      containers:
        - name: mysql
          image: docker.io/library/mysql:8.4
          imagePullPolicy: IfNotPresent
          ports:
            - containerPort: 3306
              name: mysql
          env:
            - name: MYSQL_ROOT_PASSWORD
              valueFrom:
                secretKeyRef:
                  name: mysql-auth
                  key: mysql-root-password
            - name: MYSQL_REPLICATION_PASSWORD
              valueFrom:
                secretKeyRef:
                  name: mysql-auth
                  key: mysql-replication-password
          lifecycle:
            postStart:
              exec:
                command:
                  - /bin/sh
                  - -c
                  - |
                    set -eu
                    until mysqladmin ping -h 127.0.0.1 -uroot -p"$MYSQL_ROOT_PASSWORD" --silent; do
                      sleep 5
                    done
                    if mysql -uroot -p"$MYSQL_ROOT_PASSWORD" -e "SHOW REPLICA STATUS\\G" | grep -q "Source_Host:"; then
                      mysql -uroot -p"$MYSQL_ROOT_PASSWORD" -e "SET GLOBAL read_only=ON; SET GLOBAL super_read_only=ON;"
                      exit 0
                    fi
                    until mysql -h mysql-primary.mysql.svc.cluster.local -uroot -p"$MYSQL_ROOT_PASSWORD" -e "SELECT 1" >/dev/null 2>&1; do
                      sleep 5
                    done
                    mysql -uroot -p"$MYSQL_ROOT_PASSWORD" <<SQL
                    STOP REPLICA;
                    CHANGE REPLICATION SOURCE TO
                      SOURCE_HOST='mysql-primary.mysql.svc.cluster.local',
                      SOURCE_PORT=3306,
                      SOURCE_USER='replicator',
                      SOURCE_PASSWORD='$MYSQL_REPLICATION_PASSWORD',
                      SOURCE_AUTO_POSITION=1,
                      GET_SOURCE_PUBLIC_KEY=1;
                    START REPLICA;
                    SET GLOBAL read_only=ON;
                    SET GLOBAL super_read_only=ON;
                    SQL
          volumeMounts:
            - name: data
              mountPath: /var/lib/mysql
            - name: secondary-config
              mountPath: /etc/mysql/conf.d/my.cnf
              subPath: my.cnf
      volumes:
        - name: secondary-config
          configMap:
            name: mysql-secondary-config
            items:
              - key: my.cnf
                path: my.cnf
  volumeClaimTemplates:
    - metadata:
        name: data
      spec:
        accessModes: ["ReadWriteOnce"]
        storageClassName: local-path
        resources:
          requests:
            storage: 8Gi
EOF
```

写完后先看一眼文件：

```bash
sed -n '1,260p' ~/mysql-replication-install/mysql-replication.yaml
```

重点确认下面几点：

1. 主库节点是 `k8s-master-01`
2. 从库节点是 `k8s-worker-01`
3. 存储类是 `local-path`
4. 从库 `SOURCE_HOST` 是 `mysql-primary.mysql.svc.cluster.local`

## 10. 第七步：应用资源文件

执行：

```bash
kubectl apply -f ~/mysql-replication-install/mysql-replication.yaml
```

应用后立刻检查：

```bash
kubectl -n mysql get all,pvc
```

你会看到：

1. `mysql-primary-0`
2. `mysql-secondary-0`
3. 两个 PVC

刚开始可能是：

1. `Pending`
2. `ContainerCreating`
3. `Init`

这是正常的，先不要着急。

## 11. 第八步：等待 PVC 绑定

执行：

```bash
kubectl -n mysql wait --for=jsonpath='{.status.phase}'=Bound pvc/data-mysql-primary-0 --timeout=180s
kubectl -n mysql wait --for=jsonpath='{.status.phase}'=Bound pvc/data-mysql-secondary-0 --timeout=180s
kubectl -n mysql get pvc
```

你最终应该看到：

```text
STATUS   VOLUME   CAPACITY   ACCESS MODES   STORAGECLASS
Bound    ...      8Gi        RWO            local-path
```

如果 PVC 一直 `Pending`，说明存储类没有准备好，不要继续排 MySQL，先回去检查 `local-path`。

## 12. 第九步：等待主库启动

先观察主库：

```bash
kubectl -n mysql get pod mysql-primary-0 -w
```

看到主库进入 `Running` 后，按 `Ctrl + C` 停掉观察。

再看主库日志：

```bash
kubectl -n mysql logs mysql-primary-0 --tail=100
```

看到类似下面内容，说明主库已经起来了：

```text
mysqld: ready for connections
```

再执行：

```bash
kubectl -n mysql get pod mysql-primary-0 -o wide
```

确认状态是：

```text
1/1 Running
```

## 13. 第十步：等待从库启动

执行：

```bash
kubectl -n mysql get pod mysql-secondary-0 -w
```

看到从库进入 `Running` 后，按 `Ctrl + C` 停掉观察。

再看从库日志：

```bash
kubectl -n mysql logs mysql-secondary-0 --tail=200
```

你需要重点关注下面几类信息：

1. 初始化数据库成功
2. 启动临时实例成功
3. `CHANGE REPLICATION SOURCE TO` 成功执行
4. 没有明显的 `Access denied`
5. 没有明显的 `Unknown MySQL server host`

## 14. 第十一步：检查 Pod、Service、PVC 总状态

执行：

```bash
kubectl -n mysql get pods,svc,pvc -o wide
```

正常状态应当是：

1. `mysql-primary-0` 为 `Running`
2. `mysql-secondary-0` 为 `Running`
3. `data-mysql-primary-0` 为 `Bound`
4. `data-mysql-secondary-0` 为 `Bound`
5. `mysql-primary` service 已存在
6. `mysql-secondary` service 已存在

## 15. 第十二步：检查从库复制状态

执行：

```bash
kubectl -n mysql exec mysql-secondary-0 -- sh -lc 'mysql -uroot -p"$MYSQL_ROOT_PASSWORD" -e "SHOW REPLICA STATUS\\G"'
```

关键看下面几项：

```text
Replica_IO_Running: Yes
Replica_SQL_Running: Yes
Seconds_Behind_Source: 0
Source_Host: mysql-primary.mysql.svc.cluster.local
```

只要这几项对了，说明主从复制已经通了。

## 16. 第十三步：做一次真实写入验证

先进入主库执行 SQL：

```bash
kubectl -n mysql exec -i mysql-primary-0 -- sh -lc 'mysql -uroot -p"$MYSQL_ROOT_PASSWORD"' <<'EOF'
CREATE DATABASE IF NOT EXISTS appdb;
CREATE TABLE IF NOT EXISTS appdb.replication_check (
  id INT PRIMARY KEY AUTO_INCREMENT,
  val VARCHAR(64) NOT NULL,
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
INSERT INTO appdb.replication_check(val) VALUES ('check-001');
SELECT id, val, created_at FROM appdb.replication_check ORDER BY id DESC LIMIT 3;
EOF
```

正常情况下，你会在主库看到刚插入的数据。

## 17. 第十四步：在从库读取验证

执行：

```bash
kubectl -n mysql exec -i mysql-secondary-0 -- sh -lc 'mysql -uroot -p"$MYSQL_ROOT_PASSWORD"' <<'EOF'
SHOW REPLICA STATUS\G
SELECT id, val, created_at FROM appdb.replication_check ORDER BY id DESC LIMIT 3;
EOF
```

只要从库里能读到刚才主库插入的 `check-001`，就说明这套安装已经真正成功。

## 18. 第十五步：保存最终检查结果

建议你把最终状态再保存一份，方便以后回看：

```bash
kubectl -n mysql get pods,svc,pvc -o wide
kubectl -n mysql exec mysql-secondary-0 -- sh -lc 'mysql -uroot -p"$MYSQL_ROOT_PASSWORD" -e "SHOW REPLICA STATUS\\G"'
```

## 19. 安装完成后怎么使用

到这里为止，MySQL 已经不是“装好了但不知道怎么连”，而是已经可以正式使用了。

你现在要先记住两个 Service：

1. `mysql-primary`
2. `mysql-secondary`

它们的职责非常明确：

1. `mysql-primary`：主库入口，负责写
2. `mysql-secondary`：从库入口，负责读

先确认这两个 Service：

```bash
kubectl -n mysql get svc
```

你应该能看到：

```text
mysql-primary
mysql-primary-headless
mysql-secondary
mysql-secondary-headless
```

### 19.1 在集群内部怎么访问

如果你的业务 Pod 和 MySQL 在同一个 K8s 集群里，那么最常见的访问方式就是直接使用 K8s Service DNS。

主库地址：

```text
mysql-primary.mysql.svc.cluster.local:3306
```

从库地址：

```text
mysql-secondary.mysql.svc.cluster.local:3306
```

如果业务 Pod 和 MySQL 在同一个 namespace，也可以直接写短地址：

```text
mysql-primary:3306
mysql-secondary:3306
```

### 19.2 应用应该怎么区分读写

这是最重要的使用规则。

应用不要把所有流量都打到同一个 Service。

正确方式是：

1. 写请求连 `mysql-primary`
2. 读请求连 `mysql-secondary`

可以这样理解：

1. `INSERT`
2. `UPDATE`
3. `DELETE`
4. `DDL`

这些都应该去主库。

而：

1. 查询列表
2. 查询详情
3. 报表类查询
4. 只读分析查询

这些更适合去从库。

### 19.3 当前集群里的账号怎么用

这次安装里已经准备好了三类密码：

1. root 密码
2. 业务用户 `appuser` 密码
3. 复制用户 `replicator` 密码

它们都保存在：

```bash
~/.mysql-replication-secrets.env
```

查看方式：

```bash
cat ~/.mysql-replication-secrets.env
```

重新加载方式：

```bash
source ~/.mysql-replication-secrets.env
```

业务连接最常用的是：

1. 用户名：`appuser`
2. 数据库：`appdb`

### 19.4 在主库里执行 SQL

如果你只是想先手工操作数据库，最简单的方式是直接进入主库 Pod 执行 mysql 客户端。

执行：

```bash
kubectl -n mysql exec -it mysql-primary-0 -- sh
```

进入容器后执行：

```bash
mysql -uroot -p"$MYSQL_ROOT_PASSWORD"
```

如果你不想先进 shell，也可以一条命令直接进 MySQL：

```bash
kubectl -n mysql exec -it mysql-primary-0 -- sh -lc 'mysql -uroot -p"$MYSQL_ROOT_PASSWORD"'
```

### 19.5 在从库里执行 SQL

执行：

```bash
kubectl -n mysql exec -it mysql-secondary-0 -- sh -lc 'mysql -uroot -p"$MYSQL_ROOT_PASSWORD"'
```

进入后你可以执行：

```sql
SHOW REPLICA STATUS\G
SELECT * FROM appdb.replication_check;
```

### 19.6 不进入 Pod，直接一条命令执行 SQL

如果你只是想快速执行一条 SQL，建议直接这样做。

在主库上执行：

```bash
kubectl -n mysql exec mysql-primary-0 -- sh -lc 'mysql -uroot -p"$MYSQL_ROOT_PASSWORD" -e "SHOW DATABASES;"'
```

在从库上执行：

```bash
kubectl -n mysql exec mysql-secondary-0 -- sh -lc 'mysql -uroot -p"$MYSQL_ROOT_PASSWORD" -e "SHOW REPLICA STATUS\\G"'
```

### 19.7 用业务账号连接主库

如果你想用真正的业务账号验证，而不是 root，可以这样：

```bash
kubectl -n mysql exec mysql-primary-0 -- sh -lc 'mysql -uappuser -p"$MYSQL_PASSWORD" appdb -e "SHOW TABLES;"'
```

不过这个命令依赖主库容器里已经有 `MYSQL_PASSWORD` 环境变量。

更稳妥的方式是先从 secrets 文件里拿到业务密码：

```bash
source ~/.mysql-replication-secrets.env
kubectl -n mysql exec mysql-primary-0 -- sh -lc "mysql -uappuser -p'$MYSQL_APP_PASSWORD' appdb -e 'SHOW TABLES;'"
```

### 19.8 用一个临时客户端 Pod 连接 Service

如果你想模拟“业务 Pod 访问 MySQL Service”，最实用的办法是起一个临时客户端 Pod。

执行：

```bash
kubectl -n mysql run mysql-client --rm -it \
  --image=docker.io/library/mysql:8.4 \
  --restart=Never \
  -- bash
```

进入这个临时 Pod 后，你可以连接主库：

```bash
mysql -hmysql-primary -uappuser -p
```

也可以连接从库：

```bash
mysql -hmysql-secondary -uappuser -p
```

这样更接近真实业务使用场景，因为它不是直接进入数据库容器，而是通过 Service 访问。

### 19.9 在本地电脑上怎么连接

如果你想从自己的 Mac 本地连接 K8s 里的 MySQL，最简单的方法是端口转发。

先转发主库：

```bash
ssh k8s-master-01
kubectl -n mysql port-forward svc/mysql-primary 3307:3306
```

然后在你本机另开一个终端，用 MySQL 客户端连接：

```bash
mysql -h127.0.0.1 -P3307 -uappuser -p appdb
```

如果你想连接从库，再开一个端口转发：

```bash
kubectl -n mysql port-forward svc/mysql-secondary 3308:3306
```

然后本机连接：

```bash
mysql -h127.0.0.1 -P3308 -uappuser -p appdb
```

### 19.10 Navicat / DataGrip / TablePlus 怎么连

如果你使用图形化工具，推荐也通过 `port-forward` 来连。

主库连接参数：

1. Host：`127.0.0.1`
2. Port：`3307`
3. User：`appuser`
4. Password：`~/.mysql-replication-secrets.env` 里的 `MYSQL_APP_PASSWORD`
5. Database：`appdb`

从库连接参数：

1. Host：`127.0.0.1`
2. Port：`3308`
3. User：`appuser`
4. Password：`~/.mysql-replication-secrets.env` 里的 `MYSQL_APP_PASSWORD`
5. Database：`appdb`

### 19.11 最常见的使用动作

下面是你后面最常用的一些命令。

查看所有数据库：

```bash
kubectl -n mysql exec mysql-primary-0 -- sh -lc 'mysql -uroot -p"$MYSQL_ROOT_PASSWORD" -e "SHOW DATABASES;"'
```

查看某个库里的表：

```bash
kubectl -n mysql exec mysql-primary-0 -- sh -lc 'mysql -uroot -p"$MYSQL_ROOT_PASSWORD" -e "USE appdb; SHOW TABLES;"'
```

查看主库上的用户：

```bash
kubectl -n mysql exec mysql-primary-0 -- sh -lc 'mysql -uroot -p"$MYSQL_ROOT_PASSWORD" -e "SELECT user,host FROM mysql.user;"'
```

查看从库复制状态：

```bash
kubectl -n mysql exec mysql-secondary-0 -- sh -lc 'mysql -uroot -p"$MYSQL_ROOT_PASSWORD" -e "SHOW REPLICA STATUS\\G"'
```

### 19.12 最实用的读写验证方式

你后面如果想快速确认“现在主从还正常”，最简单就是下面这组命令。

先往主库插入一条数据：

```bash
kubectl -n mysql exec -i mysql-primary-0 -- sh -lc 'mysql -uroot -p"$MYSQL_ROOT_PASSWORD" appdb' <<'EOF'
INSERT INTO replication_check(val) VALUES ('manual-check');
SELECT id, val, created_at FROM replication_check ORDER BY id DESC LIMIT 5;
EOF
```

再去从库查询：

```bash
kubectl -n mysql exec -i mysql-secondary-0 -- sh -lc 'mysql -uroot -p"$MYSQL_ROOT_PASSWORD" appdb' <<'EOF'
SELECT id, val, created_at FROM replication_check ORDER BY id DESC LIMIT 5;
EOF
```

如果从库里能查到 `manual-check`，说明主从复制还是正常的。

### 19.13 使用时你要遵守的几个规则

后面真正开始用时，建议你始终遵守这几条：

1. 写操作只走 `mysql-primary`
2. 读操作优先走 `mysql-secondary`
3. 不要在从库上做业务写入
4. 改用户权限、建库建表、改结构，优先在主库操作
5. 每次怀疑异常时，先查从库 `SHOW REPLICA STATUS\G`

### 19.14 最后你应该记住的三个入口

如果你只想记最关键的三个入口，那就是：

1. 主库 Service：`mysql-primary`
2. 从库 Service：`mysql-secondary`
3. 密码文件：`~/.mysql-replication-secrets.env`

## 20. 如果你想重新安装

如果你后面想彻底重装一次，按下面顺序做：

```bash
kubectl delete namespace mysql
kubectl wait --for=delete namespace/mysql --timeout=180s || true
kubectl create namespace mysql
source ~/.mysql-replication-secrets.env
kubectl -n mysql create secret generic mysql-auth \
  --from-literal=mysql-root-password="$MYSQL_ROOT_PASSWORD" \
  --from-literal=mysql-app-password="$MYSQL_APP_PASSWORD" \
  --from-literal=mysql-replication-password="$MYSQL_REPLICATION_PASSWORD"
kubectl apply -f ~/mysql-replication-install/mysql-replication.yaml
```

## 21. 常见错误和处理方式

### 21.1 PVC 一直 Pending

检查：

```bash
kubectl get storageclass
kubectl -n mysql describe pvc data-mysql-primary-0
kubectl -n mysql describe pvc data-mysql-secondary-0
```

通常原因：

1. `local-path` 不存在
2. `local-path-provisioner` 没跑起来

### 21.2 主库 CrashLoopBackOff

检查：

```bash
kubectl -n mysql logs mysql-primary-0 --previous
kubectl -n mysql describe pod mysql-primary-0
```

如果你在 `my.cnf` 里加了下面这行：

```cnf
default_authentication_plugin=caching_sha2_password
```

请删掉它。

### 21.3 从库 PostStartHookError

检查：

```bash
kubectl -n mysql logs mysql-secondary-0
kubectl -n mysql describe pod mysql-secondary-0
```

重点看下面几个方向：

1. 主库是否已经先启动成功
2. `replicator` 用户是否创建成功
3. `SOURCE_HOST` 是否写成了完整 FQDN
4. 是否错误地把 `read_only=ON` 和 `super_read_only=ON` 写进了 `my.cnf`

### 21.4 从库报 Access denied for user replicator

在主库里重新执行：

```bash
kubectl -n mysql exec -i mysql-primary-0 -- sh -lc 'mysql -uroot -p"$MYSQL_ROOT_PASSWORD"' <<EOF
CREATE USER IF NOT EXISTS 'replicator'@'%' IDENTIFIED BY '${MYSQL_REPLICATION_PASSWORD}';
ALTER USER 'replicator'@'%' IDENTIFIED BY '${MYSQL_REPLICATION_PASSWORD}';
GRANT REPLICATION SLAVE, REPLICATION CLIENT ON *.* TO 'replicator'@'%';
FLUSH PRIVILEGES;
EOF
```

然后重启从库：

```bash
kubectl -n mysql delete pod mysql-secondary-0
```

### 21.5 从库报 Unknown MySQL server host

说明从库找不到主库地址。

请检查你 YAML 里是否写成：

```sql
SOURCE_HOST='mysql-primary.mysql.svc.cluster.local'
```

不要只写成：

```sql
SOURCE_HOST='mysql-primary'
```

## 22. 最终你应该记住的几个文件

这次安装最重要的两个文件是：

1. `~/.mysql-replication-secrets.env`
2. `~/mysql-replication-install/mysql-replication.yaml`

只要这两个文件还在，你后面基本就可以按这篇文档重新部署。

## 23. 关联文档

如果你想看设计思路和真实排坑过程，可以再看下面两篇：

1. [[k8s安装mysql1主1从方案设计]]
2. [[k8s安装mysql1主1从-实操安装手册]]
