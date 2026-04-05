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

    TLS 1.3 大幅度简化了握手的步骤，**完成 TLS 握手只要 1 RTT**，而且安全性更高

### 证书优化

*   减小证书的大小

### 会话复用

*   Session ID
*   Session Ticket
*   Pre-shared Key：在重连时，客户端会把 Ticket 和 HTTP 请求一同发送给服务端

##### 巨人的肩膀

1.  <http://www.doc88.com/p-8621583210895.html>
2.  <https://zhuanlan.zhihu.com/p/33685085>
3.  <https://en.wikipedia.org/wiki/Replay_attack>
4.  <https://en.wikipedia.org/wiki/Downgrade_attack>
5.  <https://www.cnblogs.com/racent-Z/p/14011056.html>
6.  <http://www.guoyanbin.com/a-detailed-look-at-rfc-8446-a-k-a-tls-1-3/>
7.  <https://www.thesslstore.com/blog/crl-explained-what-is-a-certificate-revocation-list/>

