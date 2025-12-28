#!/bin/bash

# TensorBoard 启动脚本
# 用于快速启动 TensorBoard 查看训练日志

set -e

echo "========================================"
echo "TensorBoard 启动脚本"
echo "========================================"
echo ""

# 默认配置
LOGDIR="runs"
PORT=6006
HOST="localhost"

# 解析命令行参数
while [[ $# -gt 0 ]]; do
    case $1 in
        --logdir)
            LOGDIR="$2"
            shift 2
            ;;
        --port)
            PORT="$2"
            shift 2
            ;;
        --host)
            HOST="$2"
            shift 2
            ;;
        --help)
            echo "用法: bash start_tensorboard.sh [选项]"
            echo ""
            echo "选项:"
            echo "  --logdir DIR    日志目录 (默认: runs)"
            echo "  --port PORT     端口号 (默认: 6006)"
            echo "  --host HOST     主机地址 (默认: localhost)"
            echo "  --help          显示此帮助信息"
            echo ""
            echo "示例:"
            echo "  bash start_tensorboard.sh"
            echo "  bash start_tensorboard.sh --port 6007"
            echo "  bash start_tensorboard.sh --logdir runs/baseline_*"
            exit 0
            ;;
        *)
            echo "未知选项: $1"
            echo "使用 --help 查看帮助"
            exit 1
            ;;
    esac
done

# 检查日志目录是否存在
if [ ! -d "$LOGDIR" ]; then
    echo "❌ 错误: 日志目录不存在: $LOGDIR"
    echo ""
    echo "请先运行训练脚本:"
    echo "  bash run_experiments_lightning.sh --quick"
    exit 1
fi

# 检查是否有日志文件
LOG_FILES=$(find "$LOGDIR" -name "events.out.tfevents.*" 2>/dev/null | wc -l)
if [ "$LOG_FILES" -eq 0 ]; then
    echo "⚠️  警告: 在 $LOGDIR 中没有找到 TensorBoard 日志文件"
    echo ""
    echo "可能的原因:"
    echo "  1. 训练还没有开始"
    echo "  2. 日志目录不正确"
    echo ""
    echo "请先运行训练脚本:"
    echo "  bash run_experiments_lightning.sh --quick"
    echo ""
    read -p "是否继续启动 TensorBoard? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# 检查端口是否被占用
if lsof -Pi :$PORT -sTCP:LISTEN -t >/dev/null 2>&1; then
    echo "⚠️  警告: 端口 $PORT 已被占用"
    echo ""
    
    # 尝试找到占用端口的进程
    PID=$(lsof -ti:$PORT)
    PROCESS=$(ps -p $PID -o comm= 2>/dev/null || echo "未知进程")
    
    echo "占用进程: $PROCESS (PID: $PID)"
    echo ""
    echo "选项:"
    echo "  1. 杀死占用进程并继续"
    echo "  2. 使用不同端口"
    echo "  3. 退出"
    echo ""
    read -p "请选择 (1/2/3): " -n 1 -r
    echo
    
    case $REPLY in
        1)
            echo "正在杀死进程 $PID..."
            kill -9 $PID 2>/dev/null || true
            sleep 1
            ;;
        2)
            PORT=$((PORT + 1))
            echo "使用新端口: $PORT"
            ;;
        3)
            exit 0
            ;;
        *)
            echo "无效选择，退出"
            exit 1
            ;;
    esac
fi

# 显示配置信息
echo "配置信息:"
echo "  日志目录: $LOGDIR"
echo "  端口: $PORT"
echo "  主机: $HOST"
echo ""

# 统计实验数量
EXPERIMENTS=$(find "$LOGDIR" -maxdepth 1 -type d | wc -l)
EXPERIMENTS=$((EXPERIMENTS - 1))  # 减去 runs 目录本身

if [ "$EXPERIMENTS" -gt 0 ]; then
    echo "找到 $EXPERIMENTS 个实验:"
    find "$LOGDIR" -maxdepth 1 -type d -not -path "$LOGDIR" | while read -r exp; do
        exp_name=$(basename "$exp")
        versions=$(find "$exp" -maxdepth 1 -type d -name "version_*" | wc -l)
        echo "  - $exp_name ($versions 个版本)"
    done
    echo ""
fi

# 启动 TensorBoard
echo "========================================"
echo "启动 TensorBoard..."
echo "========================================"
echo ""
echo "访问地址: http://$HOST:$PORT"
echo ""
echo "提示:"
echo "  - 按 Ctrl+C 停止 TensorBoard"
echo "  - 在浏览器中打开上面的地址"
echo "  - 如果在远程服务器，使用 SSH 端口转发:"
echo "    ssh -L $PORT:localhost:$PORT user@server"
echo ""
echo "========================================"
echo ""

# 启动 TensorBoard
tensorboard --logdir="$LOGDIR" --port=$PORT --host=$HOST

