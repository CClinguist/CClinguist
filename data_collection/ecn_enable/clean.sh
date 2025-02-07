tc qdisc del dev eth0 ingress;
tc qdisc del dev ifb0 root;
tc qdisc del dev eth0 root;

killall wget curl tcpdump
