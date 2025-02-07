link=$1


sudo ifconfig ingress mtu 1500
sudo sysctl net.ipv4.tcp_sack=0


echo "Measuring base_delay..."



echo $link 

ping -c 5 $link
#wget $link


echo "DONE!"
