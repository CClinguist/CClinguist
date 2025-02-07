#!/bin/bash
dump=$1
# post delay in ms
postDelay=$2
# Buffer size in bytes
buff=$3
# buffer AQM
aqm=$4
# target website
targetLink=$5
targetIP=$6

# echo "dump: $dump"
# echo "postDelay: $postDelay"
# echo "buff: $buff"
# echo "aqm: $aqm"
# echo "targetLink: $targetLink"


# sudo ifconfig ingress mtu 100
sudo tcpdump -i ingress -w $dump &  
mm-delay $postDelay mm-link bw.trace bw.trace --uplink-queue=$aqm --downlink-queue=$aqm --downlink-queue-args="bytes=$buff" --uplink-queue-args="bytes=$buff" ./client.sh $targetLink $targetIP

sleep 1
sudo killall tcpdump mm-link mm-delay