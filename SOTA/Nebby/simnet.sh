basedelay=$1

#pre delay in ms
predelay=$2
# post delay in ms
postdelay=$3
# the website
link=$4

# btl bandwidth in kbps
bw=200
# Buffer size in bytes, set to 1 BDP
buffBDP=2

bdp=$(($(($(($basedelay+$predelay+$postdelay))*$bw*$buffBDP))/4))
buff=$bdp #(($(($buffBDP))*$(($bdp))))
echo $buff
# buffer AQM
aqm=droptail

#ssh edith iperf -s -p 3000 &

num=$(($bw/10))

rm -f bw.trace
touch bw.trace
for (( c=1; c<=$num; c++ ))
do
echo $(($(($c*1000))/$num)) >> bw.trace
done
dump=test.pcap

mm-delay $predelay ./btl.sh $dump $postdelay $buff $aqm $link
#ssh edith killall iperf
sudo killall mm-delay