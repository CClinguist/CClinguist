bw=$1


num=$(($bw/12))

rm -f bw.trace
touch bw.trace
for (( c=1; c<=$num; c++ ))
do
echo $(($(($c*1000))/$num)) >> bw.trace
done

echo