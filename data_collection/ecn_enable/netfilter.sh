#!/bin/bash

rm -f test
iptables -F
iptables -t mangle -F
#sudo iptables -t mangle -I INPUT 1 -p tcp -i eth0 -j NFQUEUE --queue-num 80
sudo iptables -t mangle -I INPUT 1 -p tcp --dport 6000 -i eth0 -j NFQUEUE --queue-num 80

iptables -t mangle -L -v -n

gcc -Wall -w -o test output.c -lnfnetlink -lnetfilter_queue -lpthread
./test




