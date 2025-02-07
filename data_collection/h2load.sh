#!/bin/bash

targetLink=$1
targetIP=$2
domain=$3

if [[ ! $targetLink =~ ^https?:// ]]; then
    echo "Error: targetLink is not a valid URI. Adding http:// prefix."
    targetLink="http://$targetLink"
fi

if [[ ! $targetIP =~ ^[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
    echo "Error: targetIP is not a valid IP address"
    exit 1
fi


echo "Adding temporary entry to /etc/hosts"
sudo sh -c "echo '$targetIP $domain' >> /etc/hosts"

h2load_cmd=$(printf "h2load --verbose -n 1000 -c 1 -m 1 --duration=15s \
--header 'Host: %s' \
--header 'User-Agent: Mozilla/5.0 (Windows NT 10.0; Win64; x64) \
AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3' \
%s" "$domain" "$targetLink")

#h2load_cmd=$(printf "h2load --verbose -n 1000 -c 1 -m 1 --header 'Host: %s' %s" "$domain" "$targetLink")
#h2load_cmd=$(printf "h2load --verbose -n 1000 -c 1 -m 1 --duration=15s --header 'Host: %s' --header 'User-Agent: Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3' %s" "$domain" "$targetLink")

#echo "Running h2load command: $h2load_cmd"

eval $h2load_cmd



