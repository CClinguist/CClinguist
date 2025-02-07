#!/bin/bash
link=$1
targetBw=$2
#bdp=$(($targetBw * 5)) 

bdp=$((targetBw * 30 / 4)) 

#bdp=$((targetBw * 15 / 4)) 


./tc.sh 1 $targetBw $bdp $link

echo "tc has finished"

./get_basedelay.sh $link

#./clean.sh
