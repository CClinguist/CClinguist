#!/bin/bash

sudo sysctl net.ipv4.tcp_sack=0
sudo ifconfig ingress mtu 1500

# no warning
gcc -w -o probe_realUrls ../probe_realUrls.c -lnfnetlink -lnetfilter_queue -lpthread -lm
sudo iptables -I INPUT -d 100.64.0.2 -m state --state ESTABLISHED -j NFQUEUE --queue-num 0

sleep 5

# probe_realUrls.c:
# argv[1] - target url
# argv[2] - target domain
# argv[3] - target ip
# argv[4] - first delay
# argv[5] - second delay
# argv[6] - switch point

sudo ./probe_realUrls $1 $2 $3 $4 3000 15000 > ../Data/time_seqNum.csv

rm -f index*
sudo iptables --flush

