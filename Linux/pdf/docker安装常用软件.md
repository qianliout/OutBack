## 安装mysql

*   查看所有镜像：docker images
*   删除对应的镜像：docker rmi id
*   删除原来的容器：docker rm -f id
*   把本地对应的数据删除了，
*   以上4部是可选步骤，如果是新装，可以不写，然后执行以下命令。

    docker run -d  --restart=always \
    \--name liuqiang-mysql -p 3306:3306 -v \
    \~/environment/docker/mysql/data:/var/lib/mysql \
    \-e MYSQL\_ROOT\_PASSWORD=root mysql\
    \--collation-server=utf8mb4\_unicode\_ci

参考：<https://hub.docker.com/_/mysql>
注意，推荐使用5.6作为测试，如果不写版本号的话，就是mysql8,会有各种各样的麻烦。

```

  docker run -d  --restart=always \
    --name mysql -p 3306:3306 \
    -v  /Users/liuqianli/work/docker/mysql/data:/var/lib/mysql \
    -e MYSQL_ROOT_PASSWORD=root mysql:8.4.5 \
    --collation-server=utf8mb4_unicode_ci
```

## 安装es

    docker run -d --restart=always \
    --name es-liuqiang  \
    -p 9200:9200 \
    -p 9300:9300 \
    -e "discovery.type=single-node" \
    -v ~/work/docker/elasticsearch/data/node1:/usr/share/elasticsearch/data \
    -v ~/word/docker/elasticsearch/logs/node1:/usr/share/elasticsearch/logs \
    -e ES_JAVA_OPTS="-Xms256m -Xmx256m" \
    elasticsearch:8.18.2

*   \-e ES\_JAVA\_OPTS=“-Xms256” 设置初始内存大小256M
*   \-e ES\_JAVA\_OPTS=“-Xmx256m” 设置最大内存使用256M
*   注意，本地没有文件夹会自动创建

使用docker-compose.yaml的方式

    services:
      elasticsearch:
        image: docker.elastic.co/elasticsearch/elasticsearch:8.12.0
        container_name: elasticsearch
        environment:
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
        image: docker.elastic.co/kibana/kibana:8.12.0
        container_name: kibana
        depends_on:
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

    docker run --restart=always \
    --hostname liuqiang-rabbitmq \
    --name rabbitmq-liuqiang \
    -e RABBITMQ_DEFAULT_USER=guest \ 
    -e RABBITMQ_DEFAULT_PASS=guest \
    -p 15672:15672  \
    -p 5672:5672  \
    -p 25672:25672 \ 
    -p 61613:61613 \
    -p 1883:1883  \
    -v ~/environment/docker/rabbitmq/home:/var/lib/rabbitmq  \
    -v ~/environment/docker/rabbitmq/config:/etc/rabbitmq/rabbitmq.config \
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

    docker run --restart=always \
    -v ~/Public/docker/redis/data/docker/redis/data:/data \
    --name liuqiang-redis \
    -p 6379:6379 -d redis redis-server --appendonly yes

\--appendonly yes 表示：start with persistent storage
如果开启持久化就可以指定义持久化的挂载卷

上述方法安装的 redis 是没有用户名和密码的

加入 redis 配置

    docker run --restart=always \
    -v ~/Public/docker/redis/data/docker/redis/data:/data \
    --name liuqiang-redis \
    -e REDIS_PASSWORD=redis \
    -e REDIS_USER=redis \
    -p 6379:6379 -d redis redis-server --appendonly yes

## 安装mongo

    docker run --restart=always \
    --name liuqiang-mongo \
    -v ~/Public/docker/redis/data/docker/mongo/data/db:/data/db \
    -p 27017:27017 \
    -p 27018:27018 \
    -p 27019:27019 \
    -e MONGO_INITDB_ROOT_USERNAME=admin \
    -e MONGO_INITDB_ROOT_PASSWORD=admin \
    -d mongo

## 安装Jaeger

    docker run -d --name jaeger \
    -e COLLECTOR_ZIPKIN_HTTP_PORT=9411 \
    -p 5775:5775/udp \
    -p 6831:6831/udp \
    -p 6832:6832/udp \
    -p 5778:5778 \
    -p 16686:16686 \
    -p 14268:14268 \
    -p 9411:9411 jaegertracing/all-in-one:1.6

## 安装hystrix-dashboard

    docker run --name hystrix-dashboard -d -p 8081:9002 mlabouardy/hystrix-dashboard

## 安装etcd

    docker run \
      -d \
      -p 2379:2379 \
      -p 2380:2380 \
      -p 4001:4001 \
      -p 7001:7001 \
      -v /root/dockerdata/etcd01/data:/data \
      --name liuqiang-etcd \
      elcolio/etcd:latest \
      -name liuqiang-etcd \
      -discovery=https://discovery.etcd.io/liuqiang \
      -advertise-client-urls http://192.168.1.99:4001 \
      -initial-advertise-peer-urls http://192.168.1.99:7001

## 安装postgres

    docker run -d \
        —name postgres \
        -e POSTGRES_PASSWORD=root \
        -e POSTGRES_USER=postgres \
        -e PGDATA=/var/lib/postgresql/data/pgdata \
        -v /Users/liuqianli/environment/docker/postgres/data:/var/lib/postgresql/data \
        -p 5432:5432  postgres:latest

## 安装redis-sentinel

*   新建两个目录

    cd /Users/liuqianli/environment/docker/redis
    mkdir redis
    mkdir redis-sentinel

*   redis目录下的操作

    vim docker-compose.yml

然后写入如下内容

    version: '2'
    services:
      master:
        image: redis       ## 镜像
        container_name: redis-master
        command: redis-server --requirepass 123456
        ports:
        - "6379:6379"
      slave1:
        image: redis                ## 镜像
        container_name: redis-slave-1
        ports:
        - "6380:6379"           ## 暴露端口
        command: redis-server --slaveof redis-master 6379 --requirepass 123456 --masterauth 123456
        depends_on:
        - master
      slave2:
        image: redis                ## 镜像
        container_name: redis-slave-2
        ports:
        - "6381:6379"           ## 暴露端口
        command: redis-server --slaveof redis-master 6379 --requirepass 123456 --masterauth 123456
        depends_on:
        - master

*   docker-compomse启动

    docker-compose up -d

redis-sentinel目录下的操作

    vim docker-compose.yml

然后写入如下内容

    vim docker-compose.yml

然后写入如下内容

     version: '2'
    services:
      sentinel1:
        image: redis       ## 镜像
        container_name: redis-sentinel-1
        ports:
        - "26379:26379"
        command: redis-sentinel /usr/local/etc/redis/sentinel.conf
        volumes:
        - "./sentinel1.conf:/usr/local/etc/redis/sentinel.conf"
      sentinel2:
        image: redis                ## 镜像
        container_name: redis-sentinel-2
        ports:
        - "26380:26379"
        command: redis-sentinel /usr/local/etc/redis/sentinel.conf
        volumes:
        - "./sentinel2.conf:/usr/local/etc/redis/sentinel.conf"
      sentinel3:
        image: redis                ## 镜像
        container_name: redis-sentinel-3
        ports:
        - "26381:26379"
        command: redis-sentinel /usr/local/etc/redis/sentinel.conf
        volumes:
        - ./sentinel3.conf:/usr/local/etc/redis/sentinel.conf

然后再创建三个配置文件（sentinel1.conf，sentinel2.conf，sentinel3.conf）三个文件的内容如下

    port 26379
    dir /tmp
    #172.18.0.3填写自己的主节点ip
    sentinel monitor mymaster 172.0.0.1 6379 2
    sentinel auth-pass mymaster 123456
    sentinel down-after-milliseconds mymaster 30000
    sentinel parallel-syncs mymaster 1
    sentinel failover-timeout mymaster 10000
    sentinel deny-scripts-reconfig yes

*   port表示端口号

