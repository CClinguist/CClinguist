#!/bin/bash
link=$1
dump=$2


sudo ifconfig ingress mtu 100
sudo sysctl net.ipv4.tcp_sack=0


echo "Launching client..."



echo $link 

wget --tries=1 --timeout=20 -U Mozilla $link -O index



sleep 1
# sudo killall tcpdump
# rm -f index
# tshark -r $dump -T fields -E separator=, -E quote=d -e frame.number -e frame.time -e ip.src -e ip.dst -e frame.len -e ip.proto -e tcp.seq -e tcp.ack -e tcp.flags > output.csv



# echo "Conversion complete. Output saved to output.csv"

echo "DONE!"
