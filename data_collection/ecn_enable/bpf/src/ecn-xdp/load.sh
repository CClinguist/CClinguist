make;
ip link set dev eth0 xdpgeneric off;
sudo ip link set dev eth0 xdpgeneric obj .output/ecn.bpf.o sec xdp;
ip link show dev eth0;
