#!/bin/bash

# 多设备并行探索启动脚本
# 使用方法：./run_parallel_exploration.sh [选项]

set -e

# 默认配置
NUM_DEVICES=1
USE_EMULATOR=false
BATCH_MODE=false
OUTPUT_DIR="./exploration_output_parallel"
MAX_BRANCHING_FACTOR=3
MAX_EXPLORATION_STEPS=30
MAX_EXPLORATION_DEPTH=5

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 帮助信息
show_help() {
    echo "多设备并行App探索启动脚本"
    echo ""
    echo "用法: $0 [选项]"
    echo ""
    echo "选项:"
    echo "  -h, --help              显示此帮助信息"
    echo "  -n, --num-devices NUM   设备数量 (默认: 1)"
    echo "  -e, --use-emulator      使用模拟器"
    echo "  -b, --batch-mode        批处理模式"
    echo "  -p, --package PKG       单个应用包名"
    echo "  -o, --output DIR        输出目录 (默认: ./exploration_output_parallel)"
    echo "  -f, --app-file FILE     应用列表文件"
    echo "  --emulator-exe PATH     模拟器可执行文件路径"
    echo "  --avd-name NAME         AVD名称"
    echo "  --avd-home PATH         AVD主目录"
    echo "  --sdk-path PATH         Android SDK路径"
    echo ""
    echo "示例:"
    echo "  # 单设备探索"
    echo "  $0 -p com.android.camera2"
    echo ""
    echo "  # 多设备并行探索默认应用列表"
    echo "  $0 -b -n 3 -e --emulator-exe /path/to/emulator --avd-name MyAVD --avd-home /path/to/avd"
    echo ""
    echo "  # 物理设备并行探索"
    echo "  $0 -b -n 2"
    echo ""
}

# 检查Python环境
check_python() {
    if ! command -v python &> /dev/null; then
        echo -e "${RED}错误: 未找到Python命令${NC}"
        exit 1
    fi

    echo -e "${BLUE}检查Python依赖...${NC}"
    python -c "import concurrent.futures, queue, threading" 2>/dev/null || {
        echo -e "${RED}错误: 缺少必要的Python模块${NC}"
        exit 1
    }
}

# 检查ADB
check_adb() {
    if ! command -v adb &> /dev/null; then
        echo -e "${RED}错误: 未找到ADB命令${NC}"
        echo "请确保Android SDK已安装并添加到PATH中"
        exit 1
    fi

    echo -e "${BLUE}ADB版本:${NC}"
    adb version | head -1
}

# 解析命令行参数
while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help)
            show_help
            exit 0
            ;;
        -n|--num-devices)
            NUM_DEVICES="$2"
            shift 2
            ;;
        -e|--use-emulator)
            USE_EMULATOR=true
            shift
            ;;
        -b|--batch-mode)
            BATCH_MODE=true
            shift
            ;;
        -p|--package)
            PACKAGE_NAME="$2"
            shift 2
            ;;
        -o|--output)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        -f|--app-file)
            APP_FILE="$2"
            shift 2
            ;;
        --emulator-exe)
            EMULATOR_EXE="$2"
            shift 2
            ;;
        --avd-name)
            AVD_NAME="$2"
            shift 2
            ;;
        --avd-home)
            AVD_HOME="$2"
            shift 2
            ;;
        --sdk-path)
            SDK_PATH="$2"
            shift 2
            ;;
        *)
            echo -e "${RED}未知选项: $1${NC}"
            show_help
            exit 1
            ;;
    esac
done

# 打印配置
echo -e "${GREEN}=== 多设备并行App探索 ===${NC}"
echo -e "${BLUE}配置:${NC}"
echo "  设备数量: $NUM_DEVICES"
echo "  使用模拟器: $USE_EMULATOR"
echo "  批处理模式: $BATCH_MODE"
echo "  输出目录: $OUTPUT_DIR"

if [ "$BATCH_MODE" = false ] && [ -z "$PACKAGE_NAME" ]; then
    echo -e "${RED}错误: 非批处理模式需要指定包名 (-p)${NC}"
    exit 1
fi

if [ "$USE_EMULATOR" = true ] && [ -z "$EMULATOR_EXE" ]; then
    echo -e "${RED}错误: 使用模拟器需要指定模拟器路径 (--emulator-exe)${NC}"
    exit 1
fi

# 环境检查
echo -e "${BLUE}环境检查...${NC}"
check_python
check_adb

# 构建Python命令
PYTHON_CMD="python main.py"
PYTHON_CMD="$PYTHON_CMD -output_dir $OUTPUT_DIR"
PYTHON_CMD="$PYTHON_CMD -num_devices $NUM_DEVICES"
PYTHON_CMD="$PYTHON_CMD -max_branching_factor $MAX_BRANCHING_FACTOR"
PYTHON_CMD="$PYTHON_CMD -max_exploration_steps $MAX_EXPLORATION_STEPS"
PYTHON_CMD="$PYTHON_CMD -max_exploration_depth $MAX_EXPLORATION_DEPTH"

if [ "$USE_EMULATOR" = true ]; then
    PYTHON_CMD="$PYTHON_CMD -use_emulator"
    [ -n "$EMULATOR_EXE" ] && PYTHON_CMD="$PYTHON_CMD -emulator_exe $EMULATOR_EXE"
    [ -n "$AVD_NAME" ] && PYTHON_CMD="$PYTHON_CMD -source_avd_name $AVD_NAME"
    [ -n "$AVD_HOME" ] && PYTHON_CMD="$PYTHON_CMD -source_avd_home $AVD_HOME"
    [ -n "$SDK_PATH" ] && PYTHON_CMD="$PYTHON_CMD -android_sdk_path $SDK_PATH"
fi

if [ "$BATCH_MODE" = true ]; then
    PYTHON_CMD="$PYTHON_CMD -batch_mode"
    [ -n "$APP_FILE" ] && PYTHON_CMD="$PYTHON_CMD -app_list_file $APP_FILE"
else
    PYTHON_CMD="$PYTHON_CMD -package_name $PACKAGE_NAME"
fi

# 显示将要执行的命令
echo -e "${BLUE}执行命令:${NC}"
echo "$PYTHON_CMD"
echo ""

# 确认执行
read -p "是否继续执行? [y/N]: " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo -e "${YELLOW}用户取消操作${NC}"
    exit 0
fi

# 执行命令
echo -e "${GREEN}开始执行...${NC}"
$PYTHON_CMD

echo -e "${GREEN}执行完成!${NC}"