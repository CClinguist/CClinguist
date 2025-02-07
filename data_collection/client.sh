#!/bin/bash
targetLink=$1
targetIP=$2
# targetLink="https://www.google.com"
# targetIP="172.217.163.36"

domain=$(echo $targetLink | sed -E 's|https?://([^/]+).*|\1|')

dump=host.pcap


sudo ifconfig ingress mtu 1500
sudo sysctl net.ipv4.tcp_sack=0

echo "Launching client..."


echo $targetLink 

sudo tcpdump -i ingress -w $dump &  # host 端口

#ping -c 5 $targetLink
#wget --tries=1 --timeout=20 -U Mozilla $targetLink -O index

#timeout 15s h2load -n 100 -c 1 -m 1 $targetLink

#./h2load.sh $targetLink $targetIP $domain
timeout 15s ./curl.sh $targetLink $targetIP $domain


# echo "Removing temporary entry from /etc/hosts"
# sudo sed -i "/$targetIP $domain/d" /etc/hosts


sleep 1

echo "DONE!"

sudo killall tcpdump

