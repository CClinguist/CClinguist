#!/bin/bash


timestamp=$(date +"%Y%m%d_%H%M%S")


tcpdump port 6000 -w capture_$timestamp.pcap



#tcpdump -i lo tcp -w local_$timestamp.pcap


#sudo tcpdump -i ifb0 -vv -w ifb0.pcap
