# 网络与Web服务

## 目录

- [5.1 nginx入门总结](#51-nginx入门总结)
- [5.2 HTTP 优化](#52-HTTP-优化)
- [5.3 TPS_QPS_RT](#53-TPS_QPS_RT)
- [5.4 TimeWait和CloseWait原因_如何规避](#54-TimeWait和CloseWait原因_如何规避)

---

## 5.1 nginx入门总结

视频：
网站：
### 一、安装
### 安装工具
### 建立统计配置目录便于管理（可选）
1. 进入系统后，在目录下建立了一个jspang的文件夹。

## 2. 进入 jspang文件夹  ,命令是 cd jspang。

### 3. 分别使用mkdir建立 app,backup,download,logs,work文件夹。
```bash
我们可以先来查看一下yum是否已经存在，但是yum源中所保存的版本可能比较低，所以要更新yum源
```
```bash
更新yum源（centos7）
```
https://juejin.im/post/6844903701459501070http://jspang.com
```bash
yum -y install gcc gcc-c++ autoconf pcre-devel make automake
```
```bash
yum -y install wget httpd-tools vim
```
```bash
yum list | grep nginx
```
```bash
vim /etc/yum.repos.d/nginx.repo
```
- // 在文件中增加
- [nginx]
- name=nginx repo
- baseurl=http://nginx.org/packages/OS/OSRELEASE/$basearch/
- gpgcheck=0
- enabled=1

---
### 二，配置文件解读（vim nginx.conf）
### yum install nginx  安装服务
### rpm -ql nginx   查看Nginx的安装目录
```bash

## systemctl start nginx.service // 启动

```
- nginx -s stop  立即停止
- nginx -s quit  从容停止服务
- killall nginx 直接杀死
### systemctl stop nginx.service  使用系统命令停止
```bash
systemctl restart nginx.service 重启
```
### nginx -s reload  重新载入配置文件

---
三，访问权限控制
#运行用户，默认即是nginx，可以不进行设置
- user nginx;
#Nginx进程，一般设置为和CPU核数一样
- worker_processes 1;
#错误日志存放目录
- error_log /var/log/nginx/error.log warn;
#进程pid存放位置
- pid /var/run/nginx.pid;
- events {
- worker_connections 1024; # 单个后台进程的最大并发数
- }
- http {
- include /etc/nginx/mime.types; #文件扩展名与类型映射表
- default_type application/octet-stream; #默认文件类型
#设置日志模式
- log_format main '$remote_addr - $remote_user [$time_local] "$request" '
- '$status $body_bytes_sent "$http_referer" '
- '"$http_user_agent" "$http_x_forwarded_for"';
- access_log /var/log/nginx/access.log main; #nginx访问日志存放位置
- sendfile on; # 开启高效传输模式
#tcp_nopush on; #减少网络报文段的数量
- keepalive_timeout 65; #保持连接的时间，也叫超时时间
#gzip on; #开启gzip压缩
### include /etc/nginx/conf.d/*.conf; #包含的子配置项位置和文件

---
只允许45.76.202.231进行访问，其他的IP是禁止访问的。但是如果我们把deny all指令，移动到allow 45.76.202.231之前，会发生什么那？会发现所有的IP都不允许访问了。这说明了一个问题：就
是在同一个块下的两个权限指令，先出现的设置会覆盖后出现的设置（也就是谁先触发，谁起作用）。
四，基于域名设置虚拟主机五，nginx负载均衡
这样访问www.siguoya.name这个域名的所有uri都会被负载均衡到backend的这三台服务器上
### 配置详解
- location / {
- allow 45.76.202.231;
- deny all;
- }
#  backend是名字，可以任意取
- upstream backend {
- ip_hash；# 这里可以不写，不写的话就是轮询算法
- server x.x.x.x:1023;
- server x.x.x.x:1024;
- server x.x.x.x:1025;
- }
- 对特定请求进行分发
- server {
- listen 1111;
- server_name www.siguoya.name;
- location / {
- proxy_pass http://backend;  # 这里要和上面取的名字一样
- }
- }

---
服务器调度详解1，轮询 按时间顺序逐一分配
2，加权轮询 权重值越大，被分配到的次数就越多3，ip_hash 请求按访问IP的hash结果来分配，同一个IP固定的访问某台后端服务器
4，least_conn 哪台服务器连接数少，就分发给哪台服务器5，url_hash 按照访问URL的hash结果来分配，同一URL固定的访问某台后端服务器
6，hash关键数值 hash自定义的key经过实际测试，ip_hash并不能完全保持cookies的连续性，最好还是将cookies存放在redis等内存型数
据库中七，location语法
### Nginx 将配置按照层级关系,用块状形式进行配置,每当一个请求来临时,nginx服务器就会处理这个请求
### 到底会映射到哪个块配置.在 Nginx 的配置文件中,两种主要的块配置是:一 server块配置，二 location
### 块配置
### server的块配置包含一系列的虚拟 server 配置,多个 server 配置就可以对多个 domain name 的请求,也
包括 ip 端口进行处理.
### 而location配置在server块配置中起着至关重要的作用,决定了URI或者资源请求应该如何被处理,这些
### URI 请求可以被拆分为多个location的配置
- upstream backend {
#down表示当前的服务器暂时不可用，不参与负载均衡
- server www.aaa.com down;
#max_conns=100表示此服务器的最大连接数为100
- server www.bbb.com:1234 max_conns=100;
#weight表示权重，权重越大，被分配到的次数就越多
- server www.ccc.com weight=3;
#backup表示这是预留的备份服务器，只在其他的节点都不可用的时候才启用
- server www.ddd.com backup;
#max_fails在连续超过3次失败后，30秒内不要再分发请求到此服务器，默认fail_timeout为10秒
- server x.x.x.x:8080 max_fails=3 fail_timeout=30s;
- }

---
### 如下就是常用的 location 配置的语法格式,其中modifier是可选的,location_match就是制定 URI 应该去
### 往哪个配置的关键.
### Regular expressions(RE)或者字面量都可以用来定义modifier,如果 location 配置中制定了modifier,可
### 能会改变 nginx匹配 location的方式,如下介绍几种最重要的modifier:
(none) 完全没有modifier表示 location会解释为前缀匹配,要确定匹配项，将根据从URI的开头匹配该location.
### = 等号表示当前这个 location 会匹配一个确定的请求,配置什么就匹配什么请求,如果匹配上了,就会停止
搜索.
~ 波浪号表示当前这个 location 会当成一个大小写敏感的RE匹配.
~* 波浪号跟星号标识 location 会按照大小写不敏感的 RE 匹配.
### ^~ 非表达式(RE)匹配,正则表达式将不会生效.
location的匹配顺序
### 1，首先匹配前缀匹配(没有 RE 表达式),针对当前这个请求,每个前缀匹配都匹配一遍.
### 2，搜索=匹配,如果当前请求匹配上了,搜索将会停止,直接使用这个这个 location.

## 3，如果第二步没有匹配上,nginx 会按照如下步骤继续搜索最长前缀匹配:

### 3.1 如果最长前缀匹配有^~这个modifier,nginx 会停止搜索并直接使用这个 location.
### 3.2 如果没有使用 ^~,暂存这个 location并且继续搜索.
4，只要最长前缀匹配被暂存和选中,nginx 就会看当前的 location 是否有大小写敏感的 RE(~和~*),第一个匹配上这种会被当做有效的 location来处理这个请求.
5，如果没有 RE 的 location 匹配上,前面暂存的 location 就会被选中来处理这个请求.
这种写法还没有明白location optional_modifier location_match {
- . . .
- }
- location ~ ^/api/(v1/ent_configs/.*)$ {
- limit_req zone=wx burst=10 nodelay;
- proxy_pass http://hikari/$1?$args;
- }

---

---

## 5.2 HTTP 优化

引用：<https://mp.weixin.qq.com/s?__biz=MzUxODAzNDg4NQ==&mid=2247488081&idx=1&sn=a285752b4b1516830ba1f549323d3580&chksm=f98e56fbcef9dfedcfe006a920722a52bd865f0c1211ba8449cc55b32a7de2df1ddd88a25fb7&scene=178&cur_album_id=1337204681134751744#rd>

## 性能损耗点

*   TLS协议握手过程
*   握手后的对称加密传输

### &#x20;硬件优化

*   选用更好的 CPU

### &#x20;软件优化

*   将 Linux 内核从 2.x 升级到 4.x
*   将 OpenSSL 从 1.0.1 升级到 1.1.1

### 协议优化

*   尽量选用 ECDHE 密钥交换算法替换 RSA 算法
*   LS 1.2 升级成 TLS 1.3 &#x20;

## TLS 1.3 大幅度简化了握手的步骤，**完成 TLS 握手只要 1 RTT**，而且安全性更高

### 证书优化

*   减小证书的大小

### 会话复用

*   Session ID
*   Session Ticket
*   Pre-shared Key：在重连时，客户端会把 Ticket 和 HTTP 请求一同发送给服务端

##### 巨人的肩膀

1.  <http://www.doc88.com/p-8621583210895.html>2.  <https://zhuanlan.zhihu.com/p/33685085>
3.  <https://en.wikipedia.org/wiki/Replay_attack>4.  <https://en.wikipedia.org/wiki/Downgrade_attack>
5.  <https://www.cnblogs.com/racent-Z/p/14011056.html>6.  <http://www.guoyanbin.com/a-detailed-look-at-rfc-8446-a-k-a-tls-1-3/>
7.  <https://www.thesslstore.com/blog/crl-explained-what-is-a-certificate-revocation-list/>

---

## 5.3 TPS_QPS_RT

# TPS QPS RT
### 开发的原因，需要对吞吐量（TPS）、QPS、并发数、响应时间（RT）几个概念做下了解，查自百度
百科，记录如下：
### 1. 响应时间(RT)
响应时间是指系统对请求作出响应的时间。直观上看，这个指标与人对软件性能的主观感受是非常一致的，因为它完整地记录了整个计算机系统处理请求的时间。由于一个系统通常会提供许多功
能，而不同功能的处理逻辑也千差万别，因而不同功能的响应时间也不尽相同，甚至同一功能在不同输入数据的情况下响应时间也不相同。所以，在讨论一个系统的响应时间时，人们通常是指该系统所
有功能的平均时间或者所有功能的最大响应时间。当然，往往也需要对每个或每组功能讨论其平均响应时间和最大响应时间。
对于单机的没有并发操作的应用系统而言，人们普遍认为响应时间是一个合理且准确的性能指标。需要指出的是，响应时间的绝对值并不能直接反映软件的性能的高低，软件性能的高低实际上取
决于用户对该响应时间的接受程度。对于一个游戏软件来说，响应时间小于100毫秒应该是不错的，响应时间在1秒左右可能属于勉强可以接受，如果响应时间达到3秒就完全难以接受了。而对于编译系统
来说，完整编译一个较大规模软件的源代码可能需要几十分钟甚至更长时间，但这些响应时间对于用户来说都是可以接受的。
2. 吞吐量(Throughput)吞吐量是指系统在单位时间内处理请求的数量。对于无并发的应用系统而言，吞吐量与响应时间成
严格的反比关系，实际上此时吞吐量就是响应时间的倒数。前面已经说过，对于单用户的系统，响应时间（或者系统响应时间和应用延迟时间）可以很好地度量系统的性能，但对于并发系统，通常需要
用吞吐量作为性能指标。
### 对于一个多用户的系统，如果只有一个用户使用时系统的平均响应时间是t，当有你n个用户使用
时，每个用户看到的响应时间通常并不是n×t，而往往比n×t小很多（当然，在某些特殊情况下也可能比n×t大，甚至大很多）。这是因为处理每个请求需要用到很多资源，由于每个请求的处理过程中有许
多不走难以并发执行，这导致在具体的一个时间点，所占资源往往并不多。也就是说在处理单个请求
### 时，在每个时间点都可能有许多资源被闲置，当处理多个请求时，如果资源配置合理，每个用户看到
的平均响应时间并不随用户数的增加而线性增加。实际上，不同系统的平均响应时间随用户数增加而增长的速度也不大相同，这也是采用吞吐量来度量并发系统的性能的主要原因。一般而言，吞吐量是
### 一个比较通用的指标，两个具有不同用户数和用户使用模式的系统，如果其最大吞吐量基本一致，则
可以判断两个系统的处理能力基本一致。
3. 并发用户数

---
### 并发用户数是指系统可以同时承载的正常使用系统功能的用户的数量。与吞吐量相比，并发用户
数是一个更直观但也更笼统的性能指标。实际上，并发用户数是一个非常不准确的指标，因为用户不
### 同的使用模式会导致不同用户在单位时间发出不同数量的请求。一网站系统为例，假设用户只有注册
### 后才能使用，但注册用户并不是每时每刻都在使用该网站，因此具体一个时刻只有部分注册用户同时
在线，在线用户就在浏览网站时会花很多时间阅读网站上的信息，因而具体一个时刻只有部分在线用户同时向系统发出请求。这样，对于网站系统我们会有三个关于用户数的统计数字：注册用户数、在
### 线用户数和同时发请求用户数。由于注册用户可能长时间不登陆网站，使用注册用户数作为性能指标
会造成很大的误差。而在线用户数和同事发请求用户数都可以作为性能指标。相比而言，以在线用户作为性能指标更直观些，而以同时发请求用户数作为性能指标更准确些。
4. QPS每秒查询率(Query Per Second)
### 每秒查询率QPS是对一个特定的查询服务器在规定时间内所处理流量多少的衡量标准，在因特网
上，作为域名系统服务器的机器的性能经常用每秒查询率来衡量。对应fetches/sec，即每秒的响应请
### 求数，也即是最大吞吐能力。 （看来是类似于TPS，只是应用于特定场景的吞吐量）

---

---

## 5.4 TimeWait和CloseWait原因_如何规避

# TimeWait和CloseWait原因 如何规避
### TIME_WAIT 和 CLOSE_WAIT 是 TCP 连接关闭过程中的两种状态，分别出现在主动关闭连接和被动关
### 闭连接的一方。它们的存在是为了确保 TCP 连接的可靠关闭，但在高并发场景下，过多的 TIME_WAIT
### 或 CLOSE_WAIT 状态可能导致系统资源耗尽，影响性能。以下是它们的原因及规避方法。
### 是 TCP 连接关闭的最后一个状态，出现在**主动关闭连接的一方**。
close() 或 shutdown() 主动关闭连接时，会进入 TIME_WAIT 状态。
### 一、TIME_WAIT 状态
### 1. **TIME_WAIT 的原因**
### TIME_WAIT
当一方调用
**作用**：
### 确保最后一个 ACK 报文能够到达对方。
防止旧连接的延迟报文干扰新连接。
**持续时间**：
TIME_WAIT 状态的持续时间通常是 **2MSL（Maximum Segment Lifetime）**。
在 Linux 中，MSL 默认为 60 秒，因此 TIME_WAIT 状态持续 120 秒。
### 2. **TIME_WAIT 的影响**
### 在高并发场景下，大量 TIME_WAIT 状态会占用系统资源（如端口和内存），导致新连接无法建
立。
### 常见于短连接频繁创建和关闭的场景（如 HTTP 请求）。
### 3. **规避 TIME_WAIT 的方法**
### **启用 TCP 重用**：
### 修改内核参数，允许重用处于 TIME_WAIT 状态的连接：
- echo 1 > /proc/sys/net/ipv4/tcp_tw_reuse
### **启用 TCP 快速回收**：
### 修改内核参数，加快 TIME_WAIT 状态的回收：
a.
b.

---
- echo 1 > /proc/sys/net/ipv4/tcp_tw_recycle
### **注意**：在 NAT 环境下，启用 tcp_tw_recycle 可能导致连接问题，需谨慎使用。
### **使用长连接**：
避免频繁创建和关闭短连接，改用长连接（如 HTTP Keep-Alive）。
### **调整 TIME_WAIT 超时时间**：
### 修改内核参数，减少 TIME_WAIT 状态的持续时间：
- echo 30 > /proc/sys/net/ipv4/tcp_fin_timeout
### 二、CLOSE_WAIT 状态
### 1. **CLOSE_WAIT 的原因**
### CLOSE_WAIT 是 TCP 连接关闭的中间状态，出现在**被动关闭连接的一方**。
当一方收到对方的 FIN 报文后，会进入 CLOSE_WAIT 状态，等待应用程序调用 close() 关闭连接。
**常见原因**：
应用程序未正确关闭连接（如未调用 close()）。
### 应用程序处理连接过慢，导致连接长时间处于 CLOSE_WAIT 状态。
### 2. **CLOSE_WAIT 的影响**
### 大量 CLOSE_WAIT 状态会占用系统资源（如文件描述符），导致无法建立新连接。
常见于服务器端，尤其是后端服务未正确处理连接关闭的场景。
### 3. **规避 CLOSE_WAIT 的方法**
**检查代码逻辑**：
确保应用程序在收到 FIN 报文后，正确调用 close() 关闭连接。
**设置超时时间**：
为连接设置超时时间，避免连接长时间未关闭：

---
- conn.SetDeadline(time.Now().Add(time.Second * 30))
### **使用连接池**：
### 在高并发场景下，使用连接池管理连接，避免频繁创建和关闭连接。
**监控和报警**：
### 监控 CLOSE_WAIT 状态的数量，及时发现和处理问题。
### 三、TIME_WAIT 和 CLOSE_WAIT 的对比
状态出现位置
原因持续时间

## 规避方法

### TIME_WAIT
主动关闭连接的一方
确保连接可靠关闭，防止旧报文干
扰新连接
### 2MSL（通常 120
秒）
### 启用 TCP 重用、
### 使用长连接、调整
超时时间
### CLOSE_WAIT
被动关闭连接的一方
应用程序未正确关闭连接
直到调用close()
检查代码逻辑、设
### 置超时时间、使用
连接池
### 四、总结
### **TIME_WAIT** 是主动关闭连接的一方进入的状态，确保连接可靠关闭。可以通过启用 TCP 重
### 用、使用长连接等方式规避。
### **CLOSE_WAIT** 是被动关闭连接的一方进入的状态，通常由应用程序未正确关闭连接引起。需要
检查代码逻辑并设置超时时间。
### 通过合理配置和优化，可以有效减少 TIME_WAIT 和 CLOSE_WAIT 状态对系统性能的影响。

---

---
