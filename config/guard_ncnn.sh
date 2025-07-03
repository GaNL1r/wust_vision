#!/bin/bash

sleep 5

while true; do
    pkill wust_vision_ncnn
    wust_vision_ncnn
    sleep 1
done