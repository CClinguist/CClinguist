# ./run_test.sh 10 50 google.com


basedelay=$1
postdelay=$2
link=$3

predelay=1

./clean.sh

#sudo echo "0" > /proc/sys/net/ipv4/tcp_sack

./simnet.sh $basedelay $predelay $postdelay $link
./pcap2csv.sh test.pcap

rm -f index*