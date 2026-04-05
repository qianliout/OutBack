使用orbstark安装好虚拟机之后，可以执行以下脚本，安装常用的软件

## 本机启停

### start
```shell
#!/bin/bash

# 定义服务基础路径
BASE_DIR="$HOME/work/docker"

# 定义服务列表（使用普通数组替代关联数组）
SERVICES=("mysql" "redis" "rabbitmq" "nginx" "kafka" "es" "qdrant" "mongodb" "chromadb" "pg" "outback")

# 显示启动菜单
show_menu() {
  echo "======================================"
  echo "          服务启动脚本"
  echo "======================================"
  echo "请选择要启动的服务 (可多选，用空格分隔):"
  echo "all     - 启动全部服务"
  echo "mysql   - 启动MySQL"
  echo "redis   - 启动Redis"
  echo "rabbitmq - 启动RabbitMQ"
  echo "nginx   - 启动Nginx"
  echo "kafka   - 启动Kafka"
  echo "es      - 启动Elasticsearch"
  echo "qdrant  - 启动Qdrant"
  echo "mongodb - 启动MongoDB"
  echo "chromadb - 启动Chromadb"
  echo "pg      - 启动Pg"
  echo "outback - 启动OrbStack outback 虚拟机"
  echo "======================================"
  read -p "请输入选择: " input_services

  # 如果没有输入任何内容
  if [ -z "$input_services" ]; then
    echo "未选择任何服务!"
    exit 1
  fi

  # 将输入转换为数组
  IFS=' ' read -ra selected_services <<<"$input_services"

  # 处理选择的服务
  process_selected_services
}

# 处理选择的服务
process_selected_services() {
  local services_to_execute=()

  # 检查是否选择了all
  for service in "${selected_services[@]}"; do
    if [ "$service" = "all" ]; then
      # 如果选择了all，则添加所有服务
      services_to_execute=("${SERVICES[@]}")
      break
    fi
  done

  # 如果没有选择all，则添加选中的服务
  if [ ${#services_to_execute[@]} -eq 0 ]; then
    for service in "${selected_services[@]}"; do
      if [[ " ${SERVICES[@]} " =~ " ${service} " ]]; then
        services_to_execute+=("$service")
      else
        echo "警告: 未知服务 '$service'，已跳过"
      fi
    done
  fi

  # 执行操作
  if [ ${#services_to_execute[@]} -gt 0 ]; then
    execute_services "${services_to_execute[@]}"
  else
    echo "没有有效的服务选择!"
    exit 1
  fi
}

# 执行启动服务
execute_services() {
  local services=("$@")

  echo ""
  echo "开始启动以下服务: ${services[*]}"
  echo "--------------------------------------"

  for service in "${services[@]}"; do
    echo -n "启动 $service... "

    # 特殊处理 outback 服务
    if [ "$service" = "outback" ]; then
      echo "执行: orbctl start outback"
      if orbctl start outback; then
        echo "成功"
      else
        echo "失败"
      fi
      continue
    fi

    # 处理其他服务的启动脚本
    local script_path="$BASE_DIR/$service/start.sh"
    echo "调试: 脚本路径 = $script_path"

    # 检查脚本是否存在
    if [ ! -f "$script_path" ]; then
      echo "失败: 启动脚本不存在 - $script_path"
      continue
    fi

    # 检查脚本是否有执行权限
    if [ ! -x "$script_path" ]; then
      chmod +x "$script_path"
    fi

    # 切换到服务目录执行脚本
    local service_dir="$BASE_DIR/$service"
    if [ -d "$service_dir" ]; then
      cd "$service_dir" || {
        echo "失败: 无法切换到目录 $service_dir"
        continue
      }
    fi

    # 执行启动脚本
    if sh "$script_path"; then
      echo "成功"
    else
      echo "失败"
    fi
  done

  echo "--------------------------------------"
  echo "启动操作完成!"
}

# 检查基础目录是否存在
if [ ! -d "$BASE_DIR" ]; then
  echo "错误: 基础目录不存在: $BASE_DIR"
  echo "请修改脚本中的 BASE_DIR 变量为正确的路径"
  exit 1
fi

# 显示菜单并执行
show_menu
```

### stop 

```shell
#!/bin/bash

# 定义服务基础路径
BASE_DIR="$HOME/work/docker"

# 定义服务列表（使用普通数组替代关联数组）
SERVICES=("mysql" "redis" "rabbitmq" "nginx" "kafka" "es" "qdrant" "mongodb" "chromadb" "pg" "outback")

# 显示停止菜单
show_menu() {
    echo "======================================"
    echo "          服务停止脚本"
    echo "======================================"
    echo "请选择要停止的服务 (可多选，用空格分隔):"
    echo "all     - 停止全部服务"
    echo "mysql   - 停止MySQL"
    echo "redis   - 停止Redis"
    echo "rabbitmq - 停止RabbitMQ"
    echo "nginx   - 停止Nginx"
    echo "kafka   - 停止Kafka"
    echo "es      - 停止Elasticsearch"
    echo "qdrant  - 停止Qdrant"
    echo "mongodb - 停止MongoDB"
    echo "chromadb - 停止Chromadb"
    echo "pg      - 停止Pg"
    echo "outback - 停止OrbStack outback 虚拟机"
    echo "======================================"
    read -p "请输入选择: " input_services

    # 如果没有输入任何内容
    if [ -z "$input_services" ]; then
        echo "未选择任何服务!"
        exit 1
    fi

    # 将输入转换为数组
    IFS=' ' read -ra selected_services <<< "$input_services"

    # 处理选择的服务
    process_selected_services
}

# 处理选择的服务
process_selected_services() {
    local services_to_execute=()

    # 检查是否选择了all
    for service in "${selected_services[@]}"; do
        if [ "$service" = "all" ]; then
            # 如果选择了all，则添加所有服务
            services_to_execute=("${SERVICES[@]}")
            break
        fi
    done

    # 如果没有选择all，则添加选中的服务
    if [ ${#services_to_execute[@]} -eq 0 ]; then
        for service in "${selected_services[@]}"; do
            if [[ " ${SERVICES[@]} " =~ " ${service} " ]]; then
                services_to_execute+=("$service")
            else
                echo "警告: 未知服务 '$service'，已跳过"
            fi
        done
    fi

    # 执行操作
    if [ ${#services_to_execute[@]} -gt 0 ]; then
        execute_services "${services_to_execute[@]}"
    else
        echo "没有有效的服务选择!"
        exit 1
    fi
}

# 执行停止服务
execute_services() {
    local services=("$@")

    echo ""
    echo "开始停止以下服务: ${services[*]}"
    echo "--------------------------------------"

    for service in "${services[@]}"; do
        echo -n "停止 $service... "

        # 特殊处理 outback 服务
        if [ "$service" = "outback" ]; then
            echo "执行: orbctl stop outback"
            if orbctl stop outback; then
                echo "成功"
            else
                echo "失败"
            fi
            continue
        fi

        local script_path="$BASE_DIR/$service/stop.sh"
        echo "调试: 脚本路径 = $script_path"

        # 检查脚本是否存在
        if [ ! -f "$script_path" ]; then
            echo "失败: 停止脚本不存在 - $script_path"
            continue
        fi

        # 检查脚本是否有执行权限
        if [ ! -x "$script_path" ]; then
            chmod +x "$script_path"
        fi

        # 切换到服务目录执行脚本
        local service_dir="$BASE_DIR/$service"
        if [ -d "$service_dir" ]; then
            cd "$service_dir" || {
                echo "失败: 无法切换到目录 $service_dir"
                continue
            }
        fi

        # 执行停止脚本
        if sh "$script_path"; then
            echo "成功"
        else
            echo "失败"
        fi
    done

    echo "--------------------------------------"
    echo "停止操作完成!"
}

# 检查基础目录是否存在
if [ ! -d "$BASE_DIR" ]; then
    echo "错误: 基础目录不存在: $BASE_DIR"
    echo "请修改脚本中的 BASE_DIR 变量为正确的路径"
    exit 1
fi

# 显示菜单并执行
show_menu
```



## 安装虚拟机后，可以安装常软件

```
#!/bin/bash

# ===================================================================
# OrbStack Ubuntu 开发环境全自动部署脚本
# 功能：修复环境 + 安装 zsh + 迁移 bash 配置 + 安装 Go + Docker + 常用工具
# 作者：Qwen
# 使用：chmod +x setup-dev-env.sh && ./setup-dev-env.sh
# ===================================================================

# set -euo pipefail

# -----------------------------------------------
# 1. 强制修复 PATH（关键！防止命令全失效）
# -----------------------------------------------
if ! command -v ls &> /dev/null || [ -z "${PATH:-}" ] || [[ "$PATH" == *"\$PATH"* ]]; then
    export PATH="/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
    echo "🔧 已强制修复 PATH 环境变量"
fi

# -----------------------------------------------
# 2. 检查是否已配置完成（幂等）
# -----------------------------------------------
if [ -f "/tmp/dev-env-setup-done" ]; then
    echo "💡 检测到开发环境已配置，跳过重复操作"
    exec zsh
    exit 0
fi

# -----------------------------------------------
# 3. 更新系统并安装基础工具
# -----------------------------------------------
echo "📦 更新系统并安装基础工具..."
apt update -y && apt upgrade -y

# 常用开发工具
apt install -y \
    git \
    curl \
    wget \
    vim \
    htop \
    net-tools \
    dnsutils \
    unzip \
    tar \
    sudo \
    ca-certificates \
    gnupg \
    lsb-release \ 
    vim \ 


echo "✅ 基础工具安装完成"

# -----------------------------------------------
# 4. 安装 zsh（如果未安装）
# -----------------------------------------------
if ! command -v zsh &> /dev/null; then
    echo "📦 安装 zsh..."
    apt install -y zsh
    echo "✅ zsh 安装完成"
else
    echo "💡 zsh 已安装"
fi

# -----------------------------------------------
# 5. 备份现有配置文件
# -----------------------------------------------
BAK_DIR="/root/.backup-dev-env-$(date +%s)"
mkdir -p "$BAK_DIR"

echo "📁 备份现有配置文件到 $BAK_DIR"

for file in ~/.bash_profile ~/.bashrc ~/.zshrc ~/.profile; do
    if [ -f "$file" ] && [ ! -L "$file" ]; then
        cp "$file" "$BAK_DIR/"
        echo "   备份: $file"
    fi
done

# -----------------------------------------------
# 6. 安装 Go 1.25.2（ARM64 适配，使用阿里云镜像）
# -----------------------------------------------
GOROOT="/usr/local/go"
GOPATH="/root/work/golang"
GO_VERSION="1.25.3"
ARCH=$(uname -m)

if [ "$ARCH" == "aarch64" ]; then
    GO_TAR="go${GO_VERSION}.linux-arm64.tar.gz"
elif [ "$ARCH" == "x86_64" ]; then
    GO_TAR="go${GO_VERSION}.linux-amd64.tar.gz"
else
    echo "❌ 不支持的架构: $ARCH"
    exit 1
fi

echo "⬇️ 下载 Go ${GO_VERSION} for ${ARCH}..."
curl -L -o "/tmp/$GO_TAR" "https://mirrors.aliyun.com/golang/$GO_TAR"

echo "📦 安装 Go..."
rm -rf "$GOROOT"
tar -C /usr/local -xzf "/tmp/$GO_TAR"
rm "/tmp/$GO_TAR"

# -----------------------------------------------
# 7. 设置 Go 环境变量（写入 .bash_profile）
# -----------------------------------------------

echo "✅ Go ${GO_VERSION} 安装完成"

# -----------------------------------------------
# 8. 创建 .bashrc（可选）
# -----------------------------------------------
BASHRC="/root/.bashrc"
cat > "$BASHRC" << 'EOF'
# ~/.bashrc

# 别名
alias ll='ls -alF'
alias la='ls -A'
alias l='ls -CF'
alias grep='grep --color=auto'

# Git 提示
if [ -f /usr/share/git-core/contrib/completion/git-prompt.sh ]; then
    source /usr/share/git-core/contrib/completion/git-prompt.sh
    export GIT_PS1_SHOWDIRTYSTATE=1
    export PS1='\u@\h:\w$(__git_ps1 " (%s)")\$ '
fi
EOF

# -----------------------------------------------
# 9. 创建 .zshrc：自动加载 bash 配置
# -----------------------------------------------
ZSHRC="/root/.zshrc"
cat > "$ZSHRC" << 'EOF'
# ~/.zshrc - 自动加载 .bash_profile 和 .bashrc

# 加载 .bash_profile（环境变量）
if [ -f "$HOME/.bash_profile" ]; then
    source "$HOME/.bash_profile"
    echo "📦 已加载 ~/.bash_profile"
fi

# 加载 .bashrc（别名、函数等）
if [ -f "$HOME/.bashrc" ]; then
    source "$HOME/.bashrc"
    echo "📦 已加载 ~/.bashrc"
fi

# 基础提示符
export PS1='%n@%m:%~%# '
EOF

echo "✅ 已生成 $ZSHRC"

# -----------------------------------------------
# 10. 设置 zsh 为默认 Shell
# -----------------------------------------------
if [ "$SHELL" != "/usr/bin/zsh" ] && [ "$SHELL" != "/bin/zsh" ]; then
    echo "🔄 设置 zsh 为默认 Shell..."
    chsh -s "$(which zsh)" root
    echo "✅ 默认 Shell 已设为 zsh"
else
    echo "💡 当前 Shell 已是 zsh"
fi

# -----------------------------------------------
# 11. 安装 Docker CE 和 containerd
# -----------------------------------------------
echo "🐳 安装 Docker 与 containerd..."

# 添加 Docker 官方 GPG 密钥
mkdir -p /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | gpg --dearmor -o /etc/apt/keyrings/docker.gpg

# 添加 Docker 仓库
echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu \
  $(lsb_release -cs) stable" | tee /etc/apt/sources.list.d/docker.list > /dev/null

# 更新包列表
apt update -y

# 安装核心组件：Docker + containerd + compose 插件
apt install -y \
    docker-ce \
    docker-ce-cli \
    containerd.io \
    docker-buildx-plugin \
    docker-compose-plugin

echo "✅ Docker 和 containerd 安装完成"

# -----------------------------------------------
# 12. 验证 containerd 是否正常运行
# -----------------------------------------------
echo "🔍 验证 containerd 服务状态..."

if systemctl is_active --quiet containerd; then
    echo "✅ containerd 服务正在运行"
else
    echo "🔄 启动 containerd 服务..."
    systemctl start containerd
    systemctl enable containerd  # 开机自启
    echo "✅ containerd 已启动并设为开机自启"
fi

# -----------------------------------------------
# 13. 验证 containerd 命令行工具
# -----------------------------------------------
if command -v ctr &> /dev/null; then
    echo "💡 ctr (containerd CLI) 可用：$(command -v ctr)"
    # 可选：查看版本
    # ctr version | head -n 3
else
    echo "❌ 错误：ctr 命令未找到，请检查 containerd.io 安装"
    exit 1
fi

# -----------------------------------------------
# 12. 标记完成，避免重复执行
# -----------------------------------------------
touch /tmp/dev-env-setup-done

# -----------------------------------------------
# 13. 最终提示
# -----------------------------------------------
cat << "EOF"

🎉 开发环境部署完成！

✅ 已安装：
   - zsh（默认 Shell）
   - Go 1.25.2
   - Docker + docker-compose
   - git, curl, vim, htop 等工具

📌 下次登录将自动进入 zsh
📌 所有 .bash_profile 和 .bashrc 配置已继承

💡 本次会话将启动 zsh...
EOF

# 等待用户确认
read -p "按回车键启动 zsh..." || true

# 启动 zsh
exec zsh

```

## ob和本机文件同步
