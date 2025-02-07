#!/bin/bash
sudo sysctl -w net.ipv4.tcp_ecn=1
# Change to the target directory
cd /home/admin/tc/bpf-developer-tutorial/src/ecn-xdp

target_ip=$1

hex_ip=$(printf '0x%02X%02X%02X%02X\n' $(echo $target_ip | tr '.' ' '))

sudo sed -i "s/#define SOURCE_IP .*/#define SOURCE_IP $hex_ip/" ecn.bpf.c

echo "Updated SOURCE_IP $target_ip to $hex_ip in ecn.bpf.c"

# Execute the script
./load.sh
