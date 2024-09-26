#!/bin/bash

DELAY_TIME=$1
TARGET_URL=$2
TARGET_DOMAIN=$3
TARGET_IP=$4
INTERVAL_TIME=$5
CAPTURE_FILE=$6

timeout 30s mm-delay $DELAY_TIME ./curl_realUrls.sh $TARGET_URL $TARGET_DOMAIN $TARGET_IP $INTERVAL_TIME 1 &

sleep 1
mm_delay_pid=$(ps aux | grep "mm-delay" | grep -v grep | awk '{print $2}')

first_pid=$(echo "$mm_delay_pid" | sort -n | head -1)   # timeout id
second_pid=$(echo "$mm_delay_pid" | sort -n | head -2 | tail -1)  # mm-delay id
echo "The PID of mm-delay is: $second_pid, the pid of timeout is: $first_pid"

sudo tcpdump -i delay-$second_pid -w $CAPTURE_FILE host $TARGET_IP

wait $first_pid

sudo killall wget curl tcpdump
sudo pkill -f 'probe_realUrls'  
