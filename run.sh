#!/bin/bash
cd "$(dirname "$0")"

source env.bash
blue="\033[1;34m"
yellow="\033[1;33m"
reset="\033[0m"
red="\033[1;31m"

rm /dev/shm/debug_frame.jpg /dev/shm/cmd_log.json /dev/shm/aim_log.json /dev/shm/target_log.json
chmod 666 /dev/shm/debug_frame
rm -rf build/config
ln -s ../config build/config

# 如果参数为 rebuild，则删除 build 文件夹
if [ "$1" == "rebuild" ]; then
    echo -e "${yellow}<--- Rebuilding: Removing build directory --->${reset}"
    rm -rf build
fi

if [ ! -d "build" ]; then 
    mkdir build
fi

echo -e "${yellow}<--- Start CMake --->${reset}"
cd build
cmake -DCMAKE_EXPORT_COMPILE_COMMANDS=YES .. \
  -DCMAKE_CXX_COMPILER=clang++ -DCMAKE_C_COMPILER=clang \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CXX_FLAGS="--gcc-toolchain=/usr"
if [ $? -ne 0 ]; then
    echo -e "${red}\n--- CMake Failed ---${reset}"
    exit 1
fi

echo -e "${yellow}\n<--- Start Make --->${reset}"
max_threads=$(grep -c "processor" /proc/cpuinfo)
make -j "$max_threads"
if [ $? -ne 0 ]; then
    echo -e "${red}\n--- Make Failed ---${reset}"
    exit 1
fi

echo -e "${yellow}\n<--- Total Lines --->${reset}"
total=$(find .. \
    -type d \( \
        -path ../build -o \
        -path ../hikSDK -o \
        -path ../model -o \
        -path ../3rdparty -o \
        -path ../.cache \
    \) -prune -o \
    -type f \( \
        -name "*.cpp" -o \
        -name "*.hpp" -o \
        -name "*.c" -o \
        -name "*.h" -o \
        -name "*.py" -o \
        -name "*.html" -o \
        -name "*.sh" -o \
        -name "*.md" -o \
        -name "*.yaml" -o \
        -name "*.json" -o \
        -name "*.css" -o \
        -name "*.js" -o \
        -name "*.cu" -o \
        -name "*.txt"\
    \) -exec wc -l {} + | awk 'END{print $1}')
echo -e "${blue}        $total${reset}"

# Only build
if [ "$1" == "build" ] || [ "$1" == "rebuild" ]; then
    echo -e "${yellow}\n<--- Only building and copying executable --->${reset}"
    cp ./wust_vision /usr/local/bin/
    if [ $? -ne 0 ]; then
        echo -e "${red}\n--- Failed to copy wust_vision to /usr/local/bin ---${reset}"
        exit 1
    fi
    exit 0
fi

# cp ./wust_vision /usr/local/bin/
# if [ $? -ne 0 ]; then
#     echo -e "${red}\n--- Failed to copy wust_vision to /usr/local/bin ---${reset}"
#     exit 1
# fi

# Run mode
if [ "$1" == "run" ]; then
    echo -e "${yellow}\n<--- Running WUST_VISION --->${reset}"
    ./$2
    if [ $? -ne 0 ]; then
        echo -e "${red}\n--- Program crashed, running guard.sh ---${reset}"
        pkill $2
        timeout=10
        while pgrep $2 > /dev/null; do
            sleep 0.5
            timeout=$((timeout - 1))
            if [ $timeout -le 0 ]; then
                echo "$2 did not exit after 10 seconds, forcing kill"
                pkill -9 $2
                break
            fi
        done
        cd ..
        
        #./config/guard.sh $2
        exit 1
    fi
elif [ "$1" == "cal" ]; then
    echo -e "${yellow}\n<--- Running camera_calibrator --->${reset}"
    ./camera_calibrator
else
    echo -e "${red}\n--- Invalid argument: Please specify 'run', 'build', 'cal' or 'rebuild' ---${reset}"
    exit 1
fi

echo -e "${yellow}<----- OVER ----->${reset}"
