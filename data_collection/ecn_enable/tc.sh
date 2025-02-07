postDelay=$1
bdw=$2
bdp=$3
targetIp=$4

avpkt=1000
#avpkt=1500
#min_value=$((bdp/1))
min_value=$((bdp/2))
#min_value=5000
max_value=$((bdp * 3))

#delay=$((postDelay * 2))
delay=$postDelay

# burst = (2min+max)/(3*avpkt), limit = max+burst
burst=$(echo "scale=2; (2 * $min_value + $max_value) / (3 * $avpkt)" | bc)
burst=$(printf "%.0f" $burst)  
limit=$(($max_value + $burst * $avpkt))

echo "postDelay: $postDelay"
echo "bdw: $bdw"
echo "bdp: $bdp"
echo "targetIp: $targetIp"
echo "min_value: $min_value"
echo "max_value: $max_value"
echo "delay: $delay"
echo "burst: $burst"
echo "limit: $limit"

#tc qdisc del dev eth0 ingress;
#tc qdisc del dev ifb0 root;
#tc qdisc del dev eth0 root;
#modprobe ipt_ECN;
#modprobe xt_ecn;
modprobe ifb numifbs=1;
ip link set dev ifb0 up;   # ifb0 is the ingress interface
# tc qdisc del dev eth0 ingress;
# tc qdisc del dev ifb0 root;
# tc qdisc del dev eth0 root;
# Redirect ingress traffic to ifb0
tc qdisc add dev eth0 handle ffff: ingress
tc filter add dev eth0 parent ffff: protocol ip u32 match ip src ${targetIp} action mirred egress redirect dev ifb0;
#tc filter add dev eth0 parent ffff: protocol ip u32 match ip src ${targetIp}/32 action mirred egress redirect dev ifb0
#tc qdisc add dev ifb0 root netem limit 10 delay 100ms loss random 90% ecn rate 10kbit;
tc qdisc add dev ifb0 root handle 1: htb default 1;

# tc class add dev ifb0 parent 1: classid 1:1 htb rate 100kbit;
# tc qdisc add dev ifb0 parent 1:1 handle 20: netem delay 100ms;
tc class add dev ifb0 parent 1: classid 1:1 htb rate ${bdw}kbit;
tc qdisc add dev ifb0 parent 1:1 handle 20: netem delay ${delay}ms;

#tc qdisc del dev eth0 root
tc qdisc add dev eth0 root handle 1: htb default 30
tc class add dev eth0 parent 1: classid 1:20 htb rate ${bdw}kbit


# tc qdisc add dev eth0 parent 1:20 handle 10: netem delay ${delay}ms  

tc filter add dev eth0 protocol ip parent 1:0 prio 1 u32 match ip dst ${targetIp}/32 flowid 1:20  
#tc filter add dev eth0 protocol ip parent 1:0 prio 1 u32 match ip dst ${targetIp}/32 match ip protocol 6 0xff match u8 0x10 0xff at 33 flowid 1:20 
tc qdisc add dev eth0 parent 1:20 handle 10: netem delay ${delay}ms

tc class add dev eth0 parent 1: classid 1:30 htb rate 500mbit

tc filter add dev eth0 protocol ip parent 1:0 prio 1 u32 match ip dport 80 0xffff flowid 1:20
tc filter add dev eth0 protocol ip parent 1:0 prio 1 u32 match ip dport 443 0xffff flowid 1:20


#tc qdisc add dev ifb0 parent 20: handle 10: red min 5000 max 20000 probability 0.2 limit 2000000 avpkt 1000 bandwidth 100kbit ecn;



#tc qdisc add dev ifb0 parent 20: handle 10: red min $min_value max $max_value probability 0.02 limit $limit avpkt $avpkt bandwidth ${bdw}kbit ecn burst ${burst};
tc qdisc add dev ifb0 parent 20: handle 10: red min ${min_value} max ${max_value} probability 0.4 limit 400000 avpkt $avpkt bandwidth ${bdw}kbit ecn burst ${burst};
tc qdisc show dev ifb0;
tc qdisc show dev eth0;
