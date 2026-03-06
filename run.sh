#!/bin/bash

WORK_DIR="$(dirname "$(realpath "${BASH_SOURCE[0]}")")"
BUILD_DIR="$WORK_DIR/build"
CONFIG_DIR="$WORK_DIR/config"
BIN_DIR="$WORK_DIR/bin"

source "$WORK_DIR/env.bash"
export VISION_ROOT="$WORK_DIR"

blue="\033[1;34m"
yellow="\033[1;33m"
reset="\033[0m"
red="\033[1;31m"


chmod 777 /dev/shm/debug_frame


rm -rf "$BIN_DIR/config"
ln -sf "$CONFIG_DIR" "$BIN_DIR/config"
ln -sf "$WORK_DIR/env.bash" "$BUILD_DIR/env.bash"

if [ "$1" == "rebuild" ]; then
    echo -e "${yellow}<--- Rebuilding: This will REMOVE the entire build directory --->${reset}"
    read -p "Are you sure you want to rebuild? [y/N]: " confirm
    confirm=${confirm,,}
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

if [[ "$1" == "build" || "$1" == "rebuild" || "$1" == "run" ]]; then

    echo -e "${yellow}<--- Start CMake (Ninja) --->${reset}"
    cmake -S "$WORK_DIR" -B "$BUILD_DIR" \
        -G Ninja \
        -DCMAKE_EXPORT_COMPILE_COMMANDS=YES \
        -DCMAKE_CXX_COMPILER=clang++ \
        -DCMAKE_C_COMPILER=clang

    if [ $? -ne 0 ]; then
        echo -e "${red}\n--- CMake Failed ---${reset}"
        exit 1
    fi
    SECONDS=0
    echo -e "${yellow}\n<--- Start Ninja Build --->${reset}"
    ninja -C "$BUILD_DIR"
    if [ $? -ne 0 ]; then
        echo -e "${red}\n--- Ninja Build Failed ---${reset}"
        exit 1
    fi

    build_time=$SECONDS
    printf "${blue}\n<--- Build Time --->\n        %02d:%02d (mm:ss)\n${reset}" \
        $((build_time / 60)) $((build_time % 60))
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
        RUN_PROGRAM="$BIN_DIR/$2"
        sudo pkill -9 $2
        ORIGINAL_ARGS=("$@")
        shift 2

        "$RUN_PROGRAM" "$@"
        RET=$?
        set -- "${ORIGINAL_ARGS[@]}"

        if [ $RET -ne 0 ]; then
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

            GUARD_SCRIPT="$CONFIG_DIR/guard.sh"
            TARGET_PATH="$RUN_PROGRAM"

            if [ ! -f "$GUARD_SCRIPT" ]; then
                echo -e "${red}guard.sh not found: $GUARD_SCRIPT${reset}"
                exit 1
            fi

            echo -e "${yellow}Starting guard.sh ...${reset}"
            exec "$GUARD_SCRIPT" "$TARGET_PATH" "$@"
        fi
    fi

    echo -e "${yellow}<----- OVER ----->${reset}"

else
    echo -e "${yellow}Warning:${reset} Invalid argument '$1'."
    echo -e "${yellow}Usage:${reset} $0 {build|rebuild|run <program> [args...]}"
    echo -e "${yellow}No action performed.${reset}"
    exit 0
fi
