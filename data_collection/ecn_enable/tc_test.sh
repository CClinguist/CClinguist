#tc qdisc del dev eth0 ingress;
#tc qdisc del dev ifb0 root;
#tc qdisc del dev eth0 root;
#modprobe ipt_ECN;
#modprobe xt_ecn;

#modprobe ifb numifbs=1;
#ip link set dev ifb0 down;
ip link set dev ifb0 up;   # ifb0 is the ingress interface


#tc qdisc del dev eth0 ingress;
#tc qdisc del dev ifb0 root;
#tc qdisc del dev eth0 root;
modprobe ifb numifbs=1;
ip link set dev ifb0 up;

tc qdisc del dev eth0 ingress;
tc qdisc del dev ifb0 root;
tc qdisc del dev eth0 root;

# Redirect ingress traffic to ifb0
tc qdisc add dev eth0 handle ffff: ingress
tc filter add dev eth0 parent ffff: protocol ip u32 match ip src 208.80.154.224 action mirred egress redirect dev ifb0;
#tc qdisc add dev ifb0 root netem limit 100000 delay 100ms loss random 90% ecn rate 100kbit;
tc qdisc add dev ifb0 root handle 1: htb default 1;

# tc class add dev ifb0 parent 1: classid 1:1 htb rate 100kbit;
# tc qdisc add dev ifb0 parent 1:1 handle 20: netem delay 100ms;

tc class add dev ifb0 parent 1: classid 1:1 htb rate 100kbit;
tc qdisc add dev ifb0 parent 1:1 handle 20: netem delay 100ms;
tc qdisc add dev ifb0 parent 20: handle 10: red min 5000 max 20000 probability 0.2 limit 1000000 avpkt 1000 bandwidth 100kbit ecn;

tc qdisc del dev eth0 root
tc qdisc add dev eth0 root handle 1: htb default 30
tc class add dev eth0 parent 1: classid 1:20 htb rate 400kbit
tc qdisc add dev eth0 parent 1:20 handle 10: netem delay 100ms
tc class add dev eth0 parent 1: classid 1:30 htb rate 500mbit
tc filter add dev eth0 protocol ip parent 1:0 prio 1 u32 match ip dport 80 0xffff flowid 1:20

tc filter add dev eth0 protocol ip parent 1:0 prio 1 u32 match ip dport 443 0xffff flowid 1:20
tc qdisc show dev eth0
tc qdisc show dev ifb0
