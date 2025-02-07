#!/bin/bash

# URL of the homepage

# Number of requests to send
NUM_REQUESTS=1500

# Target link and IP address
targetLink=$1
targetIP=$2



# Determine the port based on the protocol (HTTP or HTTPS)
if [[ $targetLink == https* ]]; then
  port=443
else
  port=80
fi

domain=$(echo $targetLink | sed -E 's|https?://([^/]+).*|\1|')

# Construct the curl command with --next for each request
CURL_CMD="curl --http1.1 --keepalive-time 60"
for ((i=1; i<=$NUM_REQUESTS; i++))
do
  CURL_CMD+=" -L -A 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10.13; rv:62.0) Gecko/20100101 Firefox/62.0' "
  if [[ $targetLink == "None" ]]; then
    CURL_CMD+="--connect-timeout 10 --max-time 30 -o /dev/null $targetIP:80 --next"
  else
    CURL_CMD+="--connect-timeout 10 --max-time 30 -o /dev/null --header 'Host: $domain' --resolve $domain:$port:$targetIP \"$targetLink\" --next"
  fi

done

# Remove the last --next
CURL_CMD=${CURL_CMD% --next}

# Execute the constructed curl command
eval $CURL_CMD

sleep 1

sudo killall tcpdump