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

docker run \
-p 27017:27017 \
--name mongod \
-v /data/mongo/data:/data/db \
--restart=always \
-d 192.168.2.29:5000/mongo:4.loyo


docker run \
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

docker cp init.js mongod:/loyo/init.js
docker exec -it   mongod mongo -u loyo -p "loyo2020" --authenticationDatabase admin admin /loyo/init.js


# 登陆控制台

docker exec -it   mongod mongo -u loyo -p "loyo2020" --authenticationDatabase admin admin

```

# 卸载

```bash
docker rm -f mongod
sudo rm -rf /data/mongo/data

```

