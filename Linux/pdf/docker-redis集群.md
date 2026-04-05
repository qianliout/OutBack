\#安装docker
`tar -zxvf docker.tar.gz&&cd docker && sh docker.sh`

\#准备redis
把6379 6380 6381 3个文件夹复制到 /home/loyo目录下

分别在192.168.2.30，192.168.2.31上执行如下内容,一共启动6个节点的redis：

```
docker run -v /home/loyo/6379/redis-master.conf:/usr/local/etc/redis/redis.conf --name redis6379  -p 6379:6379 -p 16379:16379 --net=host --restart=always -d redis redis-server /usr/local/etc/redis/redis.conf

docker run -v /home/loyo/6380/redis-master.conf:/usr/local/etc/redis/redis.conf --name redis6380  -p 6380:6380 -p 16380:16380 --net=host --restart=always -d redis redis-server /usr/local/etc/redis/redis.conf

docker run -v /home/loyo/6381/redis-master.conf:/usr/local/etc/redis/redis.conf --name redis6381  -p 6381:6381 -p 16381:16381 --net=host --restart=always -d redis redis-server /usr/local/etc/redis/redis.conf

```

\#构建集群

```
docker exec -it  redis6379 /bin/bash

redis-cli --cluster create 192.168.2.30:6379  192.168.2.30:6380  192.168.2.30:6381  192.168.2.31:6379  192.168.2.31:6380  192.168.2.31:6381  --cluster-replicas 1

```

