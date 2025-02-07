#pre_delay in ms
preDelay=$1

# post_delay in ms
postDelay=$2

# target website
targetLink=$3

# target ip of url
targetIP=$4

# target bandwidth in kbps
targetBw=$5

# target delay in ms
targetDelay=$6

buffBDP=1

# target buffer size in byte
#bdp=$(($(($(($basedelay+$preDelay+$postdelay))*$bw*$buffBDP))/4))
bdp=$(($(($targetDelay*$targetBw*$buffBDP))/4))

sudo sysctl net.ipv4.tcp_sack=0

echo "Launching client..."

./tc.sh $postDelay $targetBw $bdp $targetIP

tcpdump -i eth0 -w capture.pcap "src host ${targetIP} or src host 172.19.46.29" &
tcpdump -i ifb0 -w host.pcap &

timeout 15s ./curl.sh $targetLink $targetIP

sleep 1
sudo killall tcpdump
# echo "dump: $dump"
# echo "buff: $buff"
# echo "aqm: $aqm"
# echo "preDelay: $preDelay"
# echo "postDelay: $postDelay"
# echo "targetLink: $targetLink"
# echo "targetBw: $targetBw"
# echo "targetDelay: $targetDelay"


