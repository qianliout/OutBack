# Docker容器技术

## 目录

- [2.1 docker及docker-compose的安装和配置_centos](#21-docker及docker-compose的安装和配置_centos)
- [2.2 docker-registry安装](#22-docker-registry安装)
- [2.3 harbor安装及配置_centos](#23-harbor安装及配置_centos)
- [2.4 docker_编译Go_image](#24-docker_编译Go_image)
- [2.5 docker安装常用软件](#25-docker安装常用软件)
- [2.6 docker-es搭建集群](#26-docker-es搭建集群)
- [2.7 docker-mongo复制集群](#27-docker-mongo复制集群)
- [2.8 docker-redis集群](#28-docker-redis集群)
- [2.9 rabbitMQ docker集群搭建](#29-rabbitMQ-docker集群搭建)
- [2.10 停止_删除所有的docker容器和镜像](#210-停止_删除所有的docker容器和镜像)
- [2.11 关于我下载了orbstack导致本地docker无法在终端显示的问题](#211-关于我下载了orbstack导致本地docker无法在终端显示的问题)

---

## 2.1 docker及docker-compose的安装和配置_centos

## 安装概述

本文档介绍在CentOS系统上安装Docker和Docker Compose的完整步骤。

**官方文档参考**: https://docs.docker.com/engine/install/centos/

## 安装步骤

### 1. 卸载旧版本

首先卸载系统中可能存在的旧版本Docker：

```bash
sudo yum remove docker \
    docker-client \
    docker-client-latest \
    docker-common \
    docker-latest \
    docker-latest-logrotate \
    docker-logrotate \
    docker-engine
```

### 2. 设置Docker仓库

安装必要的工具包：

```bash
sudo yum install -y yum-utils
```

添加Docker官方仓库：

```bash
sudo yum-config-manager \
    --add-repo \
    https://download.docker.com/linux/centos/docker-ce.repo
```

### 3. 安装Docker Engine

更新yum包索引并安装Docker：

```bash
sudo yum -y update
sudo yum install docker-ce docker-ce-cli containerd.io
```

---
### 当然，还有一些其他配置，可以参考文档
### 安装docker-compose
### 安装docker-compose之前得先安装docker
```bash
Note: If the command docker-compose fails after installation, check your path. You can also create
```
a symbolic link to /usr/bin or any other directory in your path.
For example:
- // 起
```bash
systemctl start docker
```
- // 停
```bash
system stop docker
```
- // 查看状态
```bash
systemctl status docker
```
- // 开机自己启动
```bash
systemctl enable docker
```
- // 重新启动
```bash
sudo systemctl restart docker
```
# 如果启动失败，可能需要执行以下命令
```bash
sudo systemctl daemon-reload
```
```bash
https://docs.docker.com/engine/install/centos/
```
```bash
sudo curl -L "https://github.com/docker/compose/releases/download/2.20.0/docker-
```
```bash
compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
```
```bash
sudo chmod +x /usr/local/bin/docker-compose
```
```bash
sudo ln -s /usr/local/bin/docker-compose /usr/bin/docker-compose
```

---
### docker-registry的安装
### 我的目地是搭建一个只用用户名和密码登录的本地开发环境，和docker官网上的安装方法有差别，记
录如下
### 第一步，进入安装目录
第二步,生成认证文件（testuser是用户名，testpassword是密码，可以自己指定）
### 第三步，安装
### docker官网上的安装
```bash
https://docs.docker.com/registry/deploying/
```
```bash
cd /root/app/docker-regestry
```
- mkdir auth
```bash
docker run \
```
--entrypoint htpasswd \
- httpd:2 -Bbn testuser testpassword > auth/htpasswd
```bash
docker run -d \
```
-p 5000:5000 \
--restart=always \
--name registry \
-v "$(pwd)"/auth:/auth \
-e "REGISTRY_AUTH=htpasswd" \
-e "REGISTRY_AUTH_HTPASSWD_REALM=Registry Realm" \
-e REGISTRY_AUTH_HTPASSWD_PATH=/auth/htpasswd \
- registry:2

---
因为我们不需要https认证，所以去除几行。
然后就可以登录了
```bash
查看docker-rejestry的镜像占用存储大小
```
```bash
docker image时去除none
```
### 在安装K8s时docker 默认不能以root用户运行，执行以下命令让其他用户可以执行docker
```bash
docker run -d \
```
-p 5000:5000 \
--restart=always \
--name registry \
-v "$(pwd)"/auth:/auth \
-e "REGISTRY_AUTH=htpasswd" \
-e "REGISTRY_AUTH_HTPASSWD_REALM=Registry Realm" \
-e REGISTRY_AUTH_HTPASSWD_PATH=/auth/htpasswd \
-v "$(pwd)"/certs:/certs \
-e REGISTRY_HTTP_TLS_CERTIFICATE=/certs/domain.crt \
-e REGISTRY_HTTP_TLS_KEY=/certs/domain.key \
- registry:2
```bash
docker login 192.168.199.104:5000
```
```bash
docker login -u testuser -p testpassword  192.168.199.104:5000
```
- du -sch /val/lib/rejestry
```bash
docker images --filter "dangling=false"
```

---
```bash
sudo usermod -aG docker $USER && newgrp docker
```

---

---

## 2.2 docker-registry安装

# docker registry安装
### 本地安装一个不带密码的很简单，直接运行就行
```bash
docker run -d -p 5000:5000 --restart=always --name registry registry:2
```
### 安装一个带用户名和密码的
Create a password file with one entry for the user testuser, with password testpassword:
1，创建密码
```bash

## 2，启动docker-registry

```
3,登录mkdir auth
```bash
docker run \
```
--entrypoint htpasswd \
- httpd:2 -Bbn testuser testpassword > auth/htpasswd
```bash
docker run -d \
```
-p 5000:5000 \
--restart=always \
--name liuqianli-registry \
-v "$(pwd)"/auth:/auth \
-e "REGISTRY_AUTH=htpasswd" \
-e "REGISTRY_AUTH_HTPASSWD_REALM=Registry Realm" \
-e REGISTRY_AUTH_HTPASSWD_PATH=/auth/htpasswd \
-v "$(pwd)"/certs:/certs \
-e REGISTRY_HTTP_TLS_CERTIFICATE=/certs/domain.crt \
-e REGISTRY_HTTP_TLS_KEY=/certs/domain.key \
- registry:2
```bash
docker login myregistrydomain.com:5000
```

---

---

## 2.3 harbor安装及配置_centos

# harbor安装及配置 centos
### ubuntu是一样的，只是把命令改了就行，安装文档详情：
### 一、介绍
1. Harbor，是一个英文单词，意思是港湾，港湾是干什么的呢，就是停放货物的，而货物呢，是装在
```bash
集装箱中的，说到集装箱，就不得不提到Docker容器，因为docker容器的技术正是借鉴了集装箱的原
```
理。所以，Harbor正是一个用于存储Docker镜像的企业级Registry服务。
2. Registry是Dcoker官方的一个私有仓库镜像，可以将本地的镜像打标签进行标记然后push到以
### Registry起的容器的私有仓库中。企业可以根据自己的需求，使用Dokcerfile生成自己的镜像，并推到
私有仓库中，这样可以大大提高拉取镜像的效率。
二、Harbor核心组件解释Proxy：他是一个nginx的前端代理，代理Harbor的registry,UI, token等服务。
db：负责储存用户权限、审计日志、Dockerimage分组信息等数据。
UI：提供图形化界面，帮助用户管理registry上的镜像, 并对用户进行授权。
jobsevice：jobsevice是负责镜像复制工作的，他和registry通信，从一个registry pull镜像然后push到另一个registry，并记录job_log。
### Adminserver：是系统的配置管理中心附带检查存储用量，ui和jobserver启动时候回需要加载
### adminserver的配置。
Registry：镜像仓库，负责存储镜像文件。

## Log：为了帮助监控Harbor运行，负责收集其他组件的log，供日后进行分析。

三：Harbor和Registry的比较Harbor和Registry都是Docker的镜像仓库，但是Harbor作为更多企业的选择，是因为相比较于
Regisrty来说，它具有很多的优势。

## 1、提供分层传输机制，优化网络传输

### Docker镜像是是分层的，而如果每次传输都使用全量文件(所以用FTP的方式并不适合)，显然不经济。
### 必须提供识别分层传输的机制，以层的UUID为标识，确定传输的对象。
### 2、提供WEB界面，优化用户体验
只用镜像的名字来进行上传下载显然很不方便，需要有一个用户界面可以支持登陆、搜索功能，包括区分公有、私有镜像。
3、支持水平扩展集群当有用户对镜像的上传下载操作集中在某服务器，需要对相应的访问压力作分解。
https://goharbor.io/docs/latest/install-config

---
4、良好的安全机制企业中的开发团队有很多不同的职位，对于不同的职位人员，分配不同的权限，具有更好的安全性。
5、Harbor提供了基于角色的访问控制机制，并通过项目来对镜像进行组织和访问权限的控制。
kubernetes中通过namespace来对资源进行隔离，在企业级应用场景中，通过将两者进行结合可以有
### 效将kubernetes使用的镜像资源进行管理和访问控制，增强镜像使用的安全性。尤其是在多租户场景
下，可以通过租户、namespace和项目相结合的方式来实现对多租户镜像资源的管理和访问控制。

## 四：Harbor简单部署

### 官方最小配置
* 2个cpu
* 4g内存
* 40g硬盘，因为是存储镜像的所以推荐硬盘大点
### 安装前配置
### 在真正安装之前，我们还要安装下面这几个组件
* docker
* docker-compose
* python
* openssl
### 一，安装docker及docker-compose（见另一篇笔记）
二，下载Harbor离线包https://github.com/vmware/harbor/releases
### 安装有两种方式，一种是off-line ，一种是on-line，即离线和在线安装，离线安装需要下载的安装包较
### 大，在线安装下载的安装包很小，课题根据自己的情况选择，
### 下载在线安装包
- cd ~/app
```bash
wget https://storage.googleapis.com/harbor-releases/release-2.2.0/harbor-online-installer-v2.2.0.tgz
```

---
### 安装Harbor
### 解压安装包
### 编辑配置文件
- tar zxvf harbor-online-installer-v2.2.0.tgz -C /usr/src/
- cd /usr/src/harbor

---
之后直接执行cd /usr/src/harbor
- vim /usr/src/harbor/harbor.yml
- // 修改hostname为本机ip地址(如果要在本地外面访问，这一步是必须的)
- hostname: 192.168.10.122
### // 以下的其它参数，可以根据实际情况修改（一般不用改动），去除多余的注释和空行，配置文件如下：
root@ubuntu:/usr/src/harbor# cat harbor.yml |grep -v "^#"|grep -v "  #"|grep -v"^$"
- hostname: 192.168.10.122
- http:
- port: 80
- harbor_admin_password: Harbor12345
- database:
- password: root123
- data_volume: /data
- clair:
- updaters_interval: 12
- http_proxy:
- https_proxy:
- no_proxy: 127.0.0.1,localhost,core,registry
- jobservice:
- max_job_workers: 10
- chart:
- absolute_url: disabled
- log:
- level: info
- rotate_count: 50
- rotate_size: 200M
- location: /var/log/harbor
- _version: 2.2.0
- cd /usr/src/harbor
- ./install.sh

---
### 出现以下提示，表示安装成功
✔ ----Harbor has been installed and started successfully.----Now you should be able to visit the admin portal at http://192.168.10.122.
For more details, please visit https://github.com/goharbor/harbor .
### 之后如果你想修改某一个配置比如你要把域名修改一下,我是这么做的先修改上面harbor.cfg这个文件,
之后重新执行install.sh这个文件就可以了访问页面
### 使用默认的账号:admin,密码:Harbor12345登录验证

## 增加开机启动

### 安装过程中如果提示 ：The protocol is https but attribute ssl_cert is not set
### 临时解决办法：修改harbor.yml ， 注释掉https相关 ，这样本机连接就不使用https

## 生产环境中的解决办法：

五、测试推送镜像
### 登录到另外一台服务器，确保docker已经安装好了。
### 本地配置仓库地址
由于默认的harbor，已经创建了一个公开项目 libary。
- http://192.168.10.122
- vi /etc/rc.local
- 最后一行添加
#harbor start
```bash
cd /usr/src/harbor && docker-compose up -d
```
https://goharbor.io/docs/2.2.0/install-config/configure-https/

---
接着重启服务推送镜像
登录仓库，admin密码还是Harbor12345
```bash
root@k8s-node1:~# docker login 192.168.10.122
```
Username: adminPassword:
```bash
WARNING! Your password will be stored unencrypted in /root/.docker/config.json.
```
Configure a credential helper to remove this warning. See
```bash
https://docs.docker.com/engine/reference/commandline/login/#credentials-store
```
Login Succeeded现在下载一个镜像alpine
```bash
docker pull alpine
```
给镜像打tag
```bash
docker tag alpine:latest 192.168.10.122/library/alpine
```
推送镜像
```bash
root@k8s-node1:~# docker push 192.168.10.122/library/alpine
```
The push refers to repository [192.168.10.122/library/alpine]1bfeebd65323: Pushed
latest: digest:
sha256:57334c50959f26ce1ee025d08f136c2292c128f84e7b229d1b0da5dac89e9866 size: 528
```bash
vim /etc/docker/daemon.json
```
- // 增加或更改内容如下：这里的ip就是我们搭建的harbor的IP
- {"insecure-registries": ["192.168.10.122"]}
```bash
systemctl restart docker
```

---

## pull一次仓库上镜像的pulls就会加一次,可以在web界面上看的增加开机启动

vi /etc/rc.local最后一行添加
#harbor start
```bash
cd /usr/src/harbor && docker-compose up -d
```
### 安装过程中如果提示 ：The protocol is https but attribute ssl_cert is not set

## 解决办法：修改harbor.yml ， 注释掉https相关注意事项

```bash
公开的项目，推送镜像时，docker必须要登录。
```
下载镜像时，无须登录。
```bash
非公开项目，下载和推送，必须docker登录才行。
```

---

---

## 2.4 docker_编译Go_image

# docker 编译Go image

## 方法一，在容器中编译go代码这种方式有以下几个缺点：

```bash
http://wolferhua.anycps.com/2019/05/16/docker-006/
```
# 指定基础镜像, 必须存在于Dockerfile 第一行
- FROM daocloud.io/golang
# 作者信息
- MAINTAINER wolferhua <wolferhua@qq.com>
# 添加文件到镜像
- ADD . /usr/src/app
# 切换目录，相当于cd xxx
- WORKDIR /usr/src/app
# 执行编译g
- RUN go build -o app main.go
# 切换目录
- WORKDIR /app
# 将编译后的移动到 /app 目录
- RUN mv /usr/src/app/app .
# 开放端口，在运行时，并不会正在的开放端口。还是需要-p
### # 这里只是标明需要使用的端口
### EXPOSE 8080
# 镜像启动时执行的命令
- CMD ["./app"]
# 构建命令
#docker build -t goweb .

---
1，编译的image过大2，如果有包依赖还得下载包依赖，这一步容易出错，所以一般不用这种方式
第二种方式，在外部编译好Go的二进制包，然后添加到image中因为在基础镜像中没有其他内容，所以要执行以下命令，使生成的main静态链接所有的库
几个基础镜像的比较：
### scratch：这个镜像是一个空的镜像，没有任何内容。且不能执行任何命令 不般不使用
busybox:是一个集成了一百多个最常用Linux命令和工具（如cat、echo、grep、mount、telnet等）的精简工具箱，它只有几MB的大小，很方便进行各种快速验证，被誉为“Linux系统的瑞士军刀”。
BusyBox可运行于多款POSIX环境的操作系统中，如Linux（包括Android）、Hurd、FreeBSD等Alpine操作系统是一个面向安全的轻型Linux发行版。它不同于通常的Linux发行版，Alpine采用了musl libc和BusyBox以减小
系统的体积和运行时资源消耗，但功能上比BusyBox又完善得多。在保持瘦身的同时，Alpine还提供了自己的包管理工具
### apk，可以通过https://pkgs.alpinelinux.org/packages查询包信息，也可以通过apk命令直接查询和安装各种软件。
# 指定基础镜像, 必须存在于Dockerfile 第一行
### # 使用基础镜像
- FROM alpine:latest
# 作者信息
- MAINTAINER wolferhua <wolferhua@qq.com>
# 切换目录，相当于cd xxx
- WORKDIR /app/
# 添加文件到镜像
- ADD /main /app/
# 执行命令
- CMD ["/app/main"]
- CGO_ENABLED=0 GOOS=linux go build -v -a -installsuffix cgo -o main .

---
alpine Docker镜像也继承了Alpine Linux发行版的这些优势。相比于其他Docker镜像，它的容量非常小，仅仅只有5MB左右
```bash
（Ubuntu系列镜像接近200MB），且拥有非常友好的包管理机制。官方镜像来自docker-alpine项目。
```
### 所以在docker image中如果没有其他特别要求，推荐使用：alpine

---

---

## 2.5 docker安装常用软件

## 安装mysql

*   查看所有镜像：docker images
*   删除对应的镜像：docker rmi id
*   删除原来的容器：docker rm -f id
*   把本地对应的数据删除了，
*   以上4部是可选步骤，如果是新装，可以不写，然后执行以下命令。

```bash
docker run -d  --restart=always \
```
```bash
\--name liuqiang-mysql -p 3306:3306 -v \
```
```bash
\~/environment/docker/mysql/data:/var/lib/mysql \
```
```bash
\-e MYSQL\_ROOT\_PASSWORD=root mysql\
```
\--collation-server=utf8mb4\_unicode\_ci

```bash
参考：<https://hub.docker.com/_/mysql>
```
```bash

## 注意，推荐使用5.6作为测试，如果不写版本号的话，就是mysql8,会有各种各样的麻烦。

```

```

```bash
docker run -d  --restart=always \
```
--name mysql -p 3306:3306 \
-v  /Users/liuqianli/work/docker/mysql/data:/var/lib/mysql \
-e MYSQL_ROOT_PASSWORD=root mysql:8.4.5 \
--collation-server=utf8mb4_unicode_ci
```

## 安装es

```bash
docker run -d --restart=always \
```
--name es-liuqiang  \
-p 9200:9200 \
-p 9300:9300 \
-e "discovery.type=single-node" \
```bash
-v ~/work/docker/elasticsearch/data/node1:/usr/share/elasticsearch/data \
```
```bash
-v ~/word/docker/elasticsearch/logs/node1:/usr/share/elasticsearch/logs \
```
-e ES_JAVA_OPTS="-Xms256m -Xmx256m" \
elasticsearch:8.18.2

*   \-e ES\_JAVA\_OPTS=“-Xms256” 设置初始内存大小256M
*   \-e ES\_JAVA\_OPTS=“-Xmx256m” 设置最大内存使用256M
*   注意，本地没有文件夹会自动创建

```bash

## 使用docker-compose.yaml的方式

```

services:
elasticsearch:
```bash
image: docker.elastic.co/elasticsearch/elasticsearch:8.12.0
```
container_name: elasticsearchenvironment:
- discovery.type=single-node
- ES_JAVA_OPTS="-Xms256m -Xmx256m"
- xpack.security.enabled=false

volumes:
- ./data:/usr/share/elasticsearch/data  # 数据目录
- ./logs:/usr/share/elasticsearch/logs # 日志目录
ports:
- "9200:9200"
- "9300:9300"
networks:
- elastic-net

kibana:
```bash
image: docker.elastic.co/kibana/kibana:8.12.0
```
container_name: kibanadepends_on:
- elasticsearch
ports:
- "5601:5601"
environment:
- ELASTICSEARCH_HOSTS=http://elasticsearch:9200
networks:
- elastic-net

networks:
elastic-net:
driver: bridge

## 安装rabbitmq

```bash
docker run --restart=always \
```
--hostname liuqiang-rabbitmq \
--name rabbitmq-liuqiang \
-e RABBITMQ_DEFAULT_USER=guest \
-e RABBITMQ_DEFAULT_PASS=guest \
-p 15672:15672  \
-p 5672:5672  \
-p 25672:25672 \
-p 61613:61613 \
-p 1883:1883  \
```bash
-v ~/environment/docker/rabbitmq/home:/var/lib/rabbitmq  \
```
```bash
-v ~/environment/docker/rabbitmq/config:/etc/rabbitmq/rabbitmq.config \
```
-d rabbitmq:3-management

*   \--hostname：指定容器主机名称
*   \--name:指定容器名称
*   \-p:将mq端口号映射到本地
*   \-e 运行时设置用户和密码
*   15672：控制台端口号
*   5672：应用访问端口号
*   /var/lib/rabbitmq  是rabbitmq的home dir里面有数据文件目录
*   /etc/rabbitmq/rabbitmq.config是配置文件目录(我自己搭建时发现这个设置无效，目录是空的)
*   \-management 表示开启管控台插件

## 安装单体redis

```bash
docker run --restart=always \
```
```bash
-v ~/Public/docker/redis/data/docker/redis/data:/data \
```
--name liuqiang-redis \
-p 6379:6379 -d redis redis-server --appendonly yes

\--appendonly yes 表示：start with persistent storage如果开启持久化就可以指定义持久化的挂载卷

## 上述方法安装的 redis 是没有用户名和密码的

## 加入 redis 配置

```bash
docker run --restart=always \
```
```bash
-v ~/Public/docker/redis/data/docker/redis/data:/data \
```
--name liuqiang-redis \
-e REDIS_PASSWORD=redis \
-e REDIS_USER=redis \
-p 6379:6379 -d redis redis-server --appendonly yes

## 安装mongo

```bash
docker run --restart=always \
```
```bash
--name liuqiang-mongo \
```
```bash
-v ~/Public/docker/redis/data/docker/mongo/data/db:/data/db \
```
-p 27017:27017 \
-p 27018:27018 \
-p 27019:27019 \
-e MONGO_INITDB_ROOT_USERNAME=admin \
-e MONGO_INITDB_ROOT_PASSWORD=admin \
```bash
-d mongo
```

## 安装Jaeger

```bash
docker run -d --name jaeger \
```
-e COLLECTOR_ZIPKIN_HTTP_PORT=9411 \
-p 5775:5775/udp \
-p 6831:6831/udp \
-p 6832:6832/udp \
-p 5778:5778 \
-p 16686:16686 \
-p 14268:14268 \
-p 9411:9411 jaegertracing/all-in-one:1.6

## 安装hystrix-dashboard

```bash
docker run --name hystrix-dashboard -d -p 8081:9002 mlabouardy/hystrix-dashboard
```

## 安装etcd

```bash
docker run \
```
-d \
-p 2379:2379 \
-p 2380:2380 \
-p 4001:4001 \
-p 7001:7001 \
```bash
-v /root/dockerdata/etcd01/data:/data \
```
--name liuqiang-etcd \
elcolio/etcd:latest \
-name liuqiang-etcd \
-discovery=https://discovery.etcd.io/liuqiang \
-advertise-client-urls http://192.168.1.99:4001 \
-initial-advertise-peer-urls http://192.168.1.99:7001

## 安装postgres

```bash
docker run -d \
```
—name postgres \
-e POSTGRES_PASSWORD=root \
-e POSTGRES_USER=postgres \
-e PGDATA=/var/lib/postgresql/data/pgdata \
```bash
-v /Users/liuqianli/environment/docker/postgres/data:/var/lib/postgresql/data \
```
-p 5432:5432  postgres:latest

## 安装redis-sentinel

*   新建两个目录

```bash
cd /Users/liuqianli/environment/docker/redis
```
mkdir redismkdir redis-sentinel

*   redis目录下的操作

```bash
vim docker-compose.yml
```

然后写入如下内容

version: '2'services:
master:
image: redis       ## 镜像container_name: redis-master
command: redis-server --requirepass 123456ports:
- "6379:6379"
slave1:
image: redis                ## 镜像container_name: redis-slave-1
ports:
- "6380:6379"           ## 暴露端口
command: redis-server --slaveof redis-master 6379 --requirepass 123456 --masterauth 123456depends_on:
- master
slave2:
image: redis                ## 镜像container_name: redis-slave-2
ports:
- "6381:6379"           ## 暴露端口
command: redis-server --slaveof redis-master 6379 --requirepass 123456 --masterauth 123456depends_on:
- master

*   docker-compomse启动

```bash
docker-compose up -d
```

```bash
redis-sentinel目录下的操作
```

```bash
vim docker-compose.yml
```

然后写入如下内容

```bash
vim docker-compose.yml
```

然后写入如下内容

version: '2'services:
sentinel1:
image: redis       ## 镜像container_name: redis-sentinel-1
ports:
- "26379:26379"
command: redis-sentinel /usr/local/etc/redis/sentinel.confvolumes:
- "./sentinel1.conf:/usr/local/etc/redis/sentinel.conf"
sentinel2:
image: redis                ## 镜像container_name: redis-sentinel-2
ports:
- "26380:26379"
command: redis-sentinel /usr/local/etc/redis/sentinel.confvolumes:
- "./sentinel2.conf:/usr/local/etc/redis/sentinel.conf"
sentinel3:
image: redis                ## 镜像container_name: redis-sentinel-3
ports:
- "26381:26379"
command: redis-sentinel /usr/local/etc/redis/sentinel.confvolumes:
- ./sentinel3.conf:/usr/local/etc/redis/sentinel.conf

然后再创建三个配置文件（sentinel1.conf，sentinel2.conf，sentinel3.conf）三个文件的内容如下

port 26379dir /tmp
#172.18.0.3填写自己的主节点ip
sentinel monitor mymaster 172.0.0.1 6379 2sentinel auth-pass mymaster 123456
sentinel down-after-milliseconds mymaster 30000sentinel parallel-syncs mymaster 1
sentinel failover-timeout mymaster 10000sentinel deny-scripts-reconfig yes

*   port表示端口号

---

## 2.6 docker-es搭建集群

# 安装docker

```bash
sudo sh /Users/wolferhua/workspace/vagrants/test/docker/docker/docker.sh && sudo usermod -a -G docker $USER && exit
```

# 获取镜像

```bash
docker pull 192.168.2.29:5000/elasticsearch/elasticsearch:7.6
```

# 创建目录

```bash
sudo mkdir -p /data/es/data
sudo chown -Rf 1000:1000 /data/es/data
sudo chmod -Rf 0755 /data/es/data

```

# 修改机器限制

```bash
sudo su
sysctl -w vm.max_map_count=262144
echo 'vm.max_map_count=262144' >>/etc/sysctl.conf
exit
```

# 运行

```bash
# /usr/share/elasticsearch/data

```bash
docker run \
```
-p 9200:9200 \
-p 9300:9300 \
--name es \
-e node.master=true \
-e node.data=true \
-e node.max_local_storage_nodes=3 \
-e node.name=`hostname` \
-e cluster.name=loyo \
-e network.host=0.0.0.0 \
-e network.publish_host=$(ip route|grep -v default|grep `ip route |grep default |awk '{print $5}'`|awk '{print $NF}') \
--add-host es1:192.168.31.36 \
--add-host es2:192.168.31.37 \
--add-host es3:192.168.31.38 \
-e discovery.seed_hosts=es1,es2,es3 \
-e cluster.initial_master_nodes=es1,es2,es3 \
-e "ES_JAVA_OPTS=-Xms512m -Xmx512m" \
-v /data/es/data:/usr/share/elasticsearch/data \
--restart=always \
-d 192.168.2.29:5000/elasticsearch/elasticsearch:7.6

# docker run \
# -p 9200:9200 \
# -p 9300:9300 \
# --name es \
# -e node.master=true \
# -e node.data=true \
# -e node.max_local_storage_nodes=3 \
# -e node.name=`hostname` \
# -e cluster.name=loyo \
# -e network.host=0.0.0.0 \
# -e network.publish_host=192.168.31.37 \
# --add-host es1:192.168.31.36 \
# --add-host es2:192.168.31.37 \
# --add-host es3:192.168.31.38 \
# -e discovery.seed_hosts=es1,es2,es3 \
# -e cluster.initial_master_nodes=es1,es2,es3 \
# -e "ES_JAVA_OPTS=-Xms512m -Xmx512m" \
# -v /data/es/data:/usr/share/elasticsearch/data \
# --restart=always \
# -d 192.168.2.29:5000/elasticsearch/elasticsearch:7.6

# docker run \
# -p 9200:9200 \
# -p 9300:9300 \
# --name es \
# -e node.master=true \
# -e node.data=true \
# -e node.max_local_storage_nodes=3 \
# -e node.name=`hostname` \
# -e cluster.name=loyo \
# -e network.host=0.0.0.0 \
# -e network.publish_host=192.168.31.38 \
# --add-host es1:192.168.31.36 \
# --add-host es2:192.168.31.37 \
# --add-host es3:192.168.31.38 \
# -e discovery.seed_hosts=es1,es2,es3 \
# -e cluster.initial_master_nodes=es1,es2,es3 \
# -e "ES_JAVA_OPTS=-Xms512m -Xmx512m" \
# -v /data/es/data:/usr/share/elasticsearch/data \
# --restart=always \
# -d 192.168.2.29:5000/elasticsearch/elasticsearch:7.6

```

```bash
docker rm -f es
sudo rm -rf /data/es/data

```

---

## 2.7 docker-mongo复制集群

# 安装docker

```bash
sudo sh /Users/wolferhua/workspace/vagrants/test/docker/docker/docker.sh && sudo usermod -a -G docker $USER && exit
```

# 获取镜像

```bash
docker pull 192.168.2.29:5000/mongo:4.loyo
```

# 创建目录

```bash
sudo mkdir -p /data/mongo/data
sudo chown -Rf 999:999 /data/mongo/data
sudo chmod -Rf 0755 /data/mongo/data
```

# 运行镜像(主节点 -- 第一次运行)

```bash
docker run \
-it --rm \
-p 27017:27017 \
-e MONGO_INITDB_ROOT_USERNAME=loyo -e MONGO_INITDB_ROOT_PASSWORD=loyo2020 \
--name mongod \
-v /data/mongo/data:/data/db \
192.168.2.29:5000/mongo:4.loyo mongod

## 执行完成后ctrl+c退出。
```

# 主从节点执行

```bash

```bash
docker run \
```
-p 27017:27017 \
--name mongod \
-v /data/mongo/data:/data/db \
--restart=always \
-d 192.168.2.29:5000/mongo:4.loyo

```bash
docker run \
```
-p 27018:27017 \
--name mongod_arbiter \
--restart=always \
-d 192.168.2.29:5000/mongo:4.loyo
```

# 主节点执行

```bash
cat > init.js  <<CODE
ret = rs.initiate( {
_id : "loyo",
members: [
{ _id: 0, host: "192.168.2.22:27017" },
{ _id: 1, host: "192.168.2.22:27018" ,arbiterOnly: true },
{ _id: 2, host: "192.168.2.23:27017" },
{ _id: 3, host: "192.168.2.23:27018" ,arbiterOnly: true }
]
});
print(JSON.stringify(ret));
print("success");
CODE

```bash
docker cp init.js mongod:/loyo/init.js
docker exec -it   mongod mongo -u loyo -p "loyo2020" --authenticationDatabase admin admin /loyo/init.js
```

# 登陆控制台

```bash
docker exec -it   mongod mongo -u loyo -p "loyo2020" --authenticationDatabase admin admin
```

```

# 卸载

```bash
docker rm -f mongod
sudo rm -rf /data/mongo/data

```

---

## 2.8 docker-redis集群

```bash

## \#安装docker

```
`tar -zxvf docker.tar.gz&&cd docker && sh docker.sh`

\#准备redis把6379 6380 6381 3个文件夹复制到 /home/loyo目录下

分别在192.168.2.30，192.168.2.31上执行如下内容,一共启动6个节点的redis：

```
docker run -v /home/loyo/6379/redis-master.conf:/usr/local/etc/redis/redis.conf --name redis6379  -p 6379:6379 -p 16379:16379 --net=host --restart=always -d redis redis-server /usr/local/etc/redis/redis.conf

```bash
docker run -v /home/loyo/6380/redis-master.conf:/usr/local/etc/redis/redis.conf --name redis6380  -p 6380:6380 -p 16380:16380 --net=host --restart=always -d redis redis-server /usr/local/etc/redis/redis.conf
```

```bash
docker run -v /home/loyo/6381/redis-master.conf:/usr/local/etc/redis/redis.conf --name redis6381  -p 6381:6381 -p 16381:16381 --net=host --restart=always -d redis redis-server /usr/local/etc/redis/redis.conf
```

```

\#构建集群

```
docker exec -it  redis6379 /bin/bash

```bash
redis-cli --cluster create 192.168.2.30:6379  192.168.2.30:6380  192.168.2.30:6381  192.168.2.31:6379  192.168.2.31:6380  192.168.2.31:6381  --cluster-replicas 1
```

```

---

## 2.9 rabbitMQ docker集群搭建

## 一构建虚拟机

环境：ubuntu18.04

```ruby
Vagrant.configure("2") do |config|
# 使用华哥的系统
config.vm.box = "wolferhua/bionic64"
config.vm.hostname = "rabbit01"
# 另一台机子是：ip:"192.168.199.103"其他没变动
config.vm.network "private_network",ip:"192.168.199.103",auto_config: true
config.vm.synced_folder "/Users/liuqiang/environment/virtualBox/sharddata/", "/home/vagrant/data/", :nfs => true

config.vm.provider "virtualbox" do |vb|
# Display the VirtualBox GUI when booting the machine
vb.gui = false
#   # Customize the amount of memory on the VM:
vb.memory = "2048"
vb.cpus = 2
vb.name = "rabbit01"
end
config.vm.provision "shell", inline: <<-SHELL
SCRIPT
# default router
app_config.vm.provision "shell", run: "always", inline: "sudo route del default"
app_config.vm.provision "shell", run: "always", inline: "sudo route add default gw 192.168.199.1"
SHELL
end
```

进入虚拟机，切换成root用户

rabbit02上执行

```shell
docker run -d --hostname rabbit02 --name rabbit02 -e RABBITMQ_ERLANG_COOKIE='loyocrm30rabbitcookie' -e RABBITMQ_NODENAME=rabbit02 -e RABBITMQ_DEFAULT_USER=loyo -e RABBITMQ_DEFAULT_PASS=loyo -e RABBITMQ_DEFAULT_VHOST=loyocrm3 -p 5672:5672 -p 15672:15672 -p 4369:4369 -p 25672:25672 -p 15671:15671 -p 5671:5671  --add-host=rabbit02:192.168.199.104 --add-host=rabbit01:192.168.199.103 -v /data/rabbitmq:/var/lib/rabbitmq rabbitmq:3.7-management
```

rabbit01上执行

```shell
docker run -d --hostname rabbit01 --name rabbit01 -e RABBITMQ_ERLANG_COOKIE='loyocrm30rabbitcookie' -e RABBITMQ_NODENAME=rabbit01 -e RABBITMQ_DEFAULT_USER=loyo -e RABBITMQ_DEFAULT_PASS=loyo -e RABBITMQ_DEFAULT_VHOST=loyocrm3 -p 5672:5672 -p 15672:15672 -p 4369:4369 -p 25672:25672 -p 15671:15671 -p 5671:5671 --add-host=rabbit02:192.168.199.104 --add-host=rabbit01:192.168.199.103 -v /data/rabbitmq:/var/lib/rabbitmq rabbitmq:3.7-management
```

## 这样两台机器都启动成功

然后把rabbit02加入到集群(在rabbit02上执行)

```bash
docker exec -it rabbit02 bash
```
rabbitmqctl stop_apprabbitmqctl join_cluster rabbit01@rabbit01
rabbitmqctl start_app

有几个容易出错的点，

*   在宿主机里更改hosts文件，在docker内部不生效，又因为是在不同的宿主机中，所以不能用netword的方式对docker进行通信，所以可以用 --add-host 方式，这种方式相当于在docker里更改host文件。（这一点坑了好久）
*   rabbitmqctl join\_cluster rabbit01\@rabbit01 这条命令@前面是节点名字，也就是上面docker run 命令中RABBITMQ\_NODENAME所起的名字，@后面是hosts 中ip对应的名字，也就是`--add-host=rabbit01:192.168.199.103`中等号后取的名字，（我这里两个都取的是一样的）

---

## 2.10 停止_删除所有的docker容器和镜像

# 停止 删除所有的docker容器和镜像
### 列出所有的容器 ID

## 停止所有的容器删除所有的容器

删除所有的镜像复制文件
### 删除所有不使用的镜像
```bash
docker ps -aq
```
```bash
docker stop $(docker ps -aq)
```
```bash
docker rm $(docker ps -aq)
```
```bash
docker rmi $(docker images -q)
```
```bash
docker cp mycontainer:/opt/file.txt /opt/local/
```
```bash
docker cp /opt/local/file.txt mycontainer:/opt/
```

---

## 删除所有停止的容器

```bash
docker image prune --force --all
```
- // 或者
```bash
docker image prune -f -a
```
```bash
docker container prune -f
```

---

---

## 2.11 关于我下载了orbstack导致本地docker无法在终端显示的问题

```bash

## 关于我下载了orbstack导致本地docker无法在终端显示的问题

```
```bash
https://www.pomfret.cn/archives/guan-yu-wo-xia-zai-le-orbstack-dao-zhi-ben-de-docker-wu-fa-z
```
ai-zhong-duan-xian-shi-de-wen-ti
```bash
docker info
```
```bash
docker context ls
```
```bash
docker context use desktop-linux
```

---

---
