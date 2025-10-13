#!/bin/bash
# 基于脚本路径的构建与运行脚本（完全不依赖 cd）

WORK_DIR="$(cd "$(dirname "$0")" && pwd)"
BUILD_DIR="$WORK_DIR/build"
CONFIG_DIR="$WORK_DIR/config"

source "$WORK_DIR/env.bash"

blue="\033[1;34m"
yellow="\033[1;33m"
reset="\033[0m"
red="\033[1;31m"

# 设置 /dev/shm/debug_frame 权限
chmod 666 /dev/shm/debug_frame

# 清理并建立符号链接
rm -rf "$BUILD_DIR/config"
ln -sf "$CONFIG_DIR" "$BUILD_DIR/config"
ln -sf "$WORK_DIR/env.bash" "$BUILD_DIR/env.bash"

# rebuild 时清理 build（增加确认）
if [ "$1" == "rebuild" ]; then
    echo -e "${yellow}<--- Rebuilding: This will REMOVE the entire build directory --->${reset}"
    read -p "Are you sure you want to rebuild? [y/N]: " confirm
    confirm=${confirm,,} # 转小写
    if [[ "$confirm" != "y" && "$confirm" != "yes" ]]; then
        echo -e "${red}Rebuild cancelled.${reset}"
        exit 0
    fi
    echo -e "${yellow}<--- Removing build directory --->${reset}"
    rm -rf "$BUILD_DIR"
    mkdir -p "$BUILD_DIR"
else
    mkdir -p "$BUILD_DIR"
fi

# --- 主逻辑区 ---
if [[ "$1" == "build" || "$1" == "rebuild" || "$1" == "run" ]]; then

    echo -e "${yellow}<--- Start CMake --->${reset}"
    cmake -S "$WORK_DIR" -B "$BUILD_DIR" \
      -DCMAKE_EXPORT_COMPILE_COMMANDS=YES \
      -DCMAKE_CXX_COMPILER=clang++ -DCMAKE_C_COMPILER=clang \
      -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_CXX_FLAGS="--gcc-toolchain=/usr"

    if [ $? -ne 0 ]; then
        echo -e "${red}\n--- CMake Failed ---${reset}"
        exit 1
    fi

    echo -e "${yellow}\n<--- Start Make --->${reset}"
    max_threads=$(grep -c "processor" /proc/cpuinfo)
    make -C "$BUILD_DIR" -j "$max_threads"
    if [ $? -ne 0 ]; then
        echo -e "${red}\n--- Make Failed ---${reset}"
        exit 1
    fi

    # 统计总行数
    echo -e "${yellow}\n<--- Total Lines --->${reset}"
    total=$(find "$WORK_DIR" \
        -type d \( \
            -path "$BUILD_DIR" -o \
            -path "$WORK_DIR/model" -o \
            -path "$WORK_DIR/3rdparty" -o \
            -path "$WORK_DIR/.cache" \
        \) -prune -o \
        -type f \( \
            -name "*.cpp" -o -name "*.hpp" -o -name "*.c" -o -name "*.h" \
            -o -name "*.py" -o -name "*.html" -o -name "*.sh" -o -name "*.md" \
            -o -name "*.yaml" -o -name "*.json" -o -name "*.css" -o -name "*.js" \
            -o -name "*.cu" -o -name "*.txt" \
        \) -exec wc -l {} + | awk 'END{print $1}')
    echo -e "${blue}        $total${reset}"

    # Only build
    if [ "$1" == "build" ] || [ "$1" == "rebuild" ]; then
        echo -e "${yellow}\n<--- Only building --->${reset}"
        echo -e "${yellow}<----- OVER ----->${reset}"
        exit 0
    fi

    # Run mode
    if [ "$1" == "run" ]; then
        echo -e "${yellow}\n<--- Running WUST_VISION --->${reset}"
        RUN_PROGRAM="$BUILD_DIR/$2"
        ORIGINAL_ARGS=("$@")
        shift 2
        "$RUN_PROGRAM" "$@"
        set -- "${ORIGINAL_ARGS[@]}"
        if [ $? -ne 0 ]; then
            echo -e "${red}\n--- Program crashed, running guard.sh ---${reset}"
            pkill "$2"
            timeout=10
            while pgrep "$2" > /dev/null; do
                sleep 0.5
                timeout=$((timeout - 1))
                if [ $timeout -le 0 ]; then
                    echo "$2 did not exit after 10 seconds, forcing kill"
                    pkill -9 "$2"
                    break
                fi
            done

            GUARD_SCRIPT="$(realpath "$CONFIG_DIR/guard.sh")"
            TARGET_PATH="$(realpath "$RUN_PROGRAM")"
            "$GUARD_SCRIPT" "$TARGET_PATH" "$@"
            exit 1
        fi
    fi

    echo -e "${yellow}<----- OVER ----->${reset}"

else
    # ⚠️ 参数不符合选项时发出警告提示但不执行任何操作
    echo -e "${yellow}Warning:${reset} Invalid argument '$1'."
    echo -e "${yellow}Usage:${reset} $0 {build|rebuild|run <program> [args...]}"
    echo -e "${yellow}No action performed.${reset}"
    exit 0
fi
