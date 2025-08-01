#!/bin/bash

sleep 5

while true; do
    pkill wust_vision
    export MVCAM_SDK_PATH=/opt/MVS
    export MVCAM_COMMON_RUNENV=/opt/MVS/lib
    export MVCAM_GENICAM_CLPROTOCOL=/opt/MVS/lib/CLProtocol
    export ALLUSERSPROFILE=/opt/MVS/MVFG

    export LD_LIBRARY_PATH=/opt/MVS/lib/64:/opt/MVS/lib/32:$LD_LIBRARY_PAT
    export LD_LIBRARY_PATH=/home/hy/TensorRT-8.5.2.2/lib:$LD_LIBRARY_PATH
    wust_vision
    sleep 1
done