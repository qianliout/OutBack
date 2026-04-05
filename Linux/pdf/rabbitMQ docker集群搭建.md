# rabbitMQ docker集群搭建

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

这样两台机器都启动成功

然后把rabbit02加入到集群(在rabbit02上执行)

    docker exec -it rabbit02 bash
    rabbitmqctl stop_app
    rabbitmqctl join_cluster rabbit01@rabbit01
    rabbitmqctl start_app

有几个容易出错的点，

*   在宿主机里更改hosts文件，在docker内部不生效，又因为是在不同的宿主机中，所以不能用netword的方式对docker进行通信，所以可以用 --add-host 方式，这种方式相当于在docker里更改host文件。（这一点坑了好久）
*   rabbitmqctl join\_cluster rabbit01\@rabbit01 这条命令@前面是节点名字，也就是上面docker run 命令中RABBITMQ\_NODENAME所起的名字，@后面是hosts 中ip对应的名字，也就是`--add-host=rabbit01:192.168.199.103`中等号后取的名字，（我这里两个都取的是一样的）

