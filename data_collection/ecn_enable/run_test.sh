#!/bin/bash

preDelay=1
postDelay=$1 #ms
targetLink=$2 #url
targetIP=$3
targetBw=$4 #Kbps
targetDelay=$5 #ms
#echo "targetBW is $targetBw"

./simnet.sh $preDelay $postDelay "$targetLink" "$targetIP" $targetBw $targetDelay

./pcap2csv.sh capture.pcap host.pcap "$targetIP" "$targetLink"

echo "done---------------"
rm -f index*
