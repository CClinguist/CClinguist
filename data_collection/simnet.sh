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

# Buffer size in bytes, set to 1 BDP
buffBDP=1

# target buffer size in byte
#bdp=$(($(($(($basedelay+$preDelay+$postdelay))*$bw*$buffBDP))/4))
bdp=$(($(($targetDelay*$targetBw*$buffBDP))/4))

buff=$bdp 

# buffer AQM
aqm=droptail

#./get_bwTrace.sh $bw

dump=capture.pcap

# echo "dump: $dump"
# echo "buff: $buff"
# echo "aqm: $aqm"
# echo "preDelay: $preDelay"
# echo "postDelay: $postDelay"
# echo "targetLink: $targetLink"
# echo "targetBw: $targetBw"
# echo "targetDelay: $targetDelay"

mm-delay $preDelay ./btl.sh $dump $postDelay $buff $aqm $targetLink $targetIP
#ssh edith killall iperf
sudo killall mm-delay