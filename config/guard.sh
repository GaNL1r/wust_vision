#!/bin/bash
sleep 5

while true; do

    pkill $1

    # 等待进程全部退出，最多等10秒
    timeout=10
    while pgrep $1 > /dev/null; do
        sleep 0.5
        timeout=$((timeout - 1))
        if [ $timeout -le 0 ]; then
            echo "$1 did not exit after 10 seconds, forcing kill"
            pkill -9 $1
            break
        fi
    done

    source env.bash
    ORIGINAL_ARGS=("$@")
    shift 1
    $1 "$@"
    set -- "${ORIGINAL_ARGS[@]}"
    
    sleep 1
done

#ps -ef | grep $1