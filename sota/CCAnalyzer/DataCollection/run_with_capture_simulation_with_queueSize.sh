#!/bin/bash

# 参数
DELAY_TIME=$1
TARGET_IP=$2
INTERVAL_TIME=$3
CAPTURE_FILE=$4
QUEUE_SIZE=$5
# echo "DELAY_TIME=$DELAY_TIME"
# echo "TARGET_IP=$TARGET_IP"
# echo "INTERVAL_TIME=$INTERVAL_TIME"
# echo "CAPTURE_FILE=$CAPTURE_FILE"


timeout 60s mm-delay $DELAY_TIME ./curl_simulation_with_queueSize.sh  None None $TARGET_IP $INTERVAL_TIME 1 &

sleep 1

mm_delay_pid=$(ps aux | grep "mm-delay" | grep -v grep | awk '{print $2}')

first_pid=$(echo "$mm_delay_pid" | sort -n | head -1)   # timeout id
second_pid=$(echo "$mm_delay_pid" | sort -n | head -2 | tail -1)  # mm-delay id
echo "The PID of mm-delay is: $second_pid, the pid of timeout is: $first_pid"


sudo ip link set delay-$second_pid txqueuelen $QUEUE_SIZE
sudo tcpdump -i delay-$second_pid -w $CAPTURE_FILE host ""


wait $first_pid

sudo killall wget curl tcpdump
sudo pkill -f 'probe_simulation_no_drop' 
