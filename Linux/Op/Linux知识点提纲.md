# Linux运维知识点提纲

> 基于Linux/Markdown目录中46个技术文档的知识点层级提纲

## 1. 系统监控与性能分析

### 1.1 iostat的用法
- I/O统计工具
- 常用参数选项
- 输出内容详解
- 监控命令示例

### 1.2 lsof用法
- 列出打开文件
- TCP/UDP连接查看
- 端口占用检查
- 文件系统监控

### 1.3 top_htop_iotop的简单使用
- top基础使用
- htop交互界面
- iotop磁盘I/O监控
- 进程资源分析

### 1.4 vmstat_pidstat_dstat_mpstat的用法
- vmstat虚拟内存统计
- pidstat进程统计
- dstat系统资源统计
- mpstat多处理器统计

## 2. Docker容器技术

### 2.1 docker及docker-compose的安装和配置_centos
- Docker安装步骤
- Docker Compose配置
- 服务启动管理

### 2.2 docker-registry安装
- 私有仓库搭建
- 认证配置
- 镜像推送拉取

### 2.3 harbor安装及配置_centos
- Harbor部署
- Web界面配置
- 用户权限管理

### 2.4 docker_编译Go_image
- Dockerfile编写
- Go应用容器化
- 镜像构建优化

### 2.5 docker安装常用软件
- 常用软件容器化部署
- 数据持久化配置
- 网络配置管理

### 2.6 docker-es搭建集群
- Elasticsearch集群部署
- 节点配置
- 集群监控

### 2.7 docker-mongo复制集群
- MongoDB复制集搭建
- 主从配置
- 数据同步

### 2.8 docker-redis集群
- Redis集群部署
- 分片配置
- 高可用设置

### 2.9 rabbitMQ docker集群搭建
- RabbitMQ集群部署
- 消息队列配置
- 集群管理

### 2.10 停止_删除所有的docker容器和镜像
- 批量容器管理
- 镜像清理
- 系统资源回收

### 2.11 关于我下载了orbstack导致本地docker无法在终端显示的问题
- OrbStack问题排查
- Docker环境修复
- 终端显示问题解决

## 3. Kubernetes容器编排

### 3.1 clusterIP和nodePort
- Service类型对比
- ClusterIP内部访问
- NodePort外部暴露
- 负载均衡配置

### 3.2 readiness_liveness_startup
- 健康检查机制
- Readiness探针
- Liveness探针
- Startup探针

### 3.3 通过文件创建_configMap
- ConfigMap创建方式
- 文件挂载配置
- 环境变量注入
- 配置热更新

### 3.4 ubuntu安装k3d
- K3d轻量级Kubernetes
- 安装配置步骤
- 集群创建管理
- 本地开发环境

## 4. 数据库管理

### 4.1 mysql的安装及配置_centos
- MySQL安装步骤
- 基础配置
- 用户权限管理
- 安全配置

### 4.2 mongo安装及配置_centos
- MongoDB安装
- 配置文件设置
- 用户认证
- 性能优化

### 4.3 redis安装及配置_centos
- Redis安装部署
- 配置参数调优
- 持久化设置
- 集群配置

### 4.4 PostgreSQL12安装和配置_centos
- PostgreSQL安装
- 数据库初始化
- 用户管理
- 性能调优

## 5. 网络与Web服务

### 5.1 nginx入门总结
- Nginx安装配置
- 虚拟主机设置
- 反向代理配置
- 负载均衡

### 5.2 HTTP 优化
- HTTP性能优化
- 缓存策略
- 压缩配置
- 安全设置

### 5.3 TPS_QPS_RT
- 性能指标定义
- 监控方法
- 压力测试
- 性能分析

### 5.4 TimeWait和CloseWait原因_如何规避
- TCP连接状态
- TimeWait问题
- CloseWait问题
- 优化方案

## 6. 系统管理与维护

### 6.1 磁盘空间满和inode满的问题排查方法
- 磁盘空间监控
- inode使用检查
- 问题排查步骤
- 清理方案

### 6.2 禁用swap
- Swap机制理解
- 禁用方法
- 性能影响
- 最佳实践

### 6.3 解决ssh权限不足
- SSH权限问题
- 文件权限修复
- 密钥配置
- 安全设置

### 6.4 怎么查看二进制文件的编译环境
- 文件信息查看
- 编译环境检测
- 依赖库分析
- 调试信息提取

### 6.5 ubuntu_安装_containerd
- Containerd安装
- 运行时配置
- 容器管理
- 故障排查

### 6.6 不退出运行 ubuntu
- 后台运行方法
- Screen使用
- Tmux会话管理
- 进程守护

### 6.7 伙伴算法_slab和内存对齐
- 内存管理机制
- 伙伴算法原理
- Slab分配器
- 内存对齐优化

## 7. 开发工具与环境

### 7.1 idea_激活
- IDEA安装配置
- 激活方法
- 插件管理
- 开发环境设置

### 7.2 ideavimrc
- Vim插件配置
- 快捷键设置
- 编辑器优化
- 个性化定制

### 7.3 iTerm2的Profiles免输入密码
- iTerm2配置
- SSH免密登录
- Profile设置
- 自动化连接

### 7.4 jfrog安装
- JFrog Artifactory部署
- 制品仓库管理
- 权限配置
- 集成设置

### 7.5 vagrant_root_登录虚拟机
- Vagrant虚拟机管理
- Root权限配置
- SSH登录设置
- 网络配置

## 8. 文本处理工具

### 8.1 awk学习
- AWK基础语法
- 模式匹配
- 字段处理
- 统计计算
- 高级用法

### 8.2 sed学习
- SED流编辑器
- 文本替换
- 行操作
- 正则表达式
- 批量处理

## 9. 脚本与自动化

### 9.1 except
- Expect自动化交互
- 脚本编写
- 密码自动输入
- 批量操作

### 9.2 登录服务器时自动输入验证码
- 验证码识别
- 自动化登录
- 图像处理
- 脚本集成

### 9.3 常用脚本
- 系统监控脚本
- 日志清理脚本
- 备份脚本
- 自动化运维

## 10. 消息队列

### 10.1 beanstalk的搭建
- Beanstalk安装
- 队列配置
- 任务管理
- 监控运维

## 11. 其他工具与技巧

### 11.1 统计某文件夹下文件的个数
- 文件统计方法
- 递归统计
- 条件过滤
- 批量处理

---

**总计**: 46个技术文档，涵盖11个主要技术领域的完整知识体系
