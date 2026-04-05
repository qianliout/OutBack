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


docker run \
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

