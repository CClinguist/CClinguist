# ./run_test.sh 50 "https://www.google.com" "172.217.163.36" 200 100
preDelay=1
postDelay=$1 #ms
targetLink=$2 #url
targetIP=$3
targetBw=$4 #Kbps
targetDelay=$5 #ms


./simnet.sh $preDelay $postDelay "$targetLink" "$targetIP" $targetBw $targetDelay

./pcap2csv.sh capture.pcap host.pcap "$targetIP"

rm -f index*