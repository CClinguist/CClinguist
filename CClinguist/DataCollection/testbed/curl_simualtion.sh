#!/bin/bash

sudo sysctl net.ipv4.tcp_sack=0
sudo ifconfig ingress mtu 1500

# no warning
gcc -w -o probe_simualtion ./probe_simualtion.c -lnfnetlink -lnetfilter_queue -lpthread -lm
sudo iptables -I INPUT -d 100.64.0.2 -m state --state ESTABLISHED -j NFQUEUE --queue-num 0

sleep 5

sudo ./probe_simualtion $1 $2 $3 $4 3000 15000 > ../Data/time_seqNum.csv


rm -f index*
sudo iptables --flush

