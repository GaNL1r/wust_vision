#!/bin/bash

sleep 5

while true; do
    pkill wust_vision
    wust_vision
    sleep 1
done