#!/bin/bash


while true; do
    pkill wust_vision

    # 等待进程全部退出，最多等10秒
    timeout=10
    while pgrep wust_vision > /dev/null; do
        sleep 0.5
        timeout=$((timeout - 1))
        if [ $timeout -le 0 ]; then
            echo "wust_vision did not exit after 10 seconds, forcing kill"
            pkill -9 wust_vision
            break
        fi
    done

    source env.bash
    wust_vision
    sleep 1
done

#ps -ef | grep wust_vision