
link=$1

mm-link bw.trace bw.trace ./get_basedelay.sh $link

sudo killall tcpdump
sudo killall mm-link mm-delay