# -*- coding: utf-8 -*-
import subprocess
import itertools
from datetime import date, timezone, datetime
import time
import os
import requests
import csv
import signal
import re
import statistics
from dateutil import parser
import pandas as pd
import urllib.parse
import math


def measure_rtt(targetUrl):
    targetUrl = re.sub(r'^https?://', '', targetUrl)
    try:
        result = subprocess.run(
            f'ping -c 4 {targetUrl}',
            capture_output=True,
            text=True,
            check=True,
            shell=True
        )
        output = result.stdout
    except subprocess.CalledProcessError as e:
        output = e.output

    match = re.search(r'(?<=min/avg/max/mdev = )([\d.]+)/([\d.]+)/([\d.]+)/([\d.]+)', output)
    if match:
        avg_rtt = float(match.group(2))
        return avg_rtt
    else:
        return 0


def convert_to_kbps(speed, unit):
    if unit == "KB/s":
        return float(speed) * 8 
    elif unit == "MB/s":
        return float(speed) * 8 * 1e3
    elif unit == "GB/s":
        return float(speed) * 8 * 1e6
    else:
        return None

def measure_bw(targetUrl):
    error_log_file = 'errors.txt'
    tmp_file = 'bw_tmp.html'
    command = f"wget -O {tmp_file} -U \"Mozilla/5.0 (Macintosh; Intel Mac OS X 10.13; rv:62.0) Gecko/20100101 Firefox/62.0\" -t 5 -T 10 '{targetUrl}' --no-check-certificate"
    
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        output = result.stdout + result.stderr

    except subprocess.CalledProcessError as e:
        output = e.stderr
        error_match = re.search(r'ERROR\s+(\d+:\s+.*)', output)
        if error_match:
            error_message = error_match.group(1)
            with open(error_log_file, 'a') as error_log:
                error_log.write(f'{targetUrl},{error_message}\n')
        return 0, None


    match = re.search(r'(\d+\.?\d*)\s*(KB/s|MB/s|GB/s)', output)
    if match:
        speed = match.group(1)
        unit = match.group(2)
        speed_kbps = convert_to_kbps(speed, unit)
        #print(output)
        ip_match = re.search(r'Connecting to [^ ]+ \([^)]+\)\|([^\|]+)\|', output)
        if ip_match:
            ip_address = ip_match.group(1)
        else:
            ip_address = None


        os.remove(tmp_file)
        return speed_kbps, ip_address
    else: 
        if os.path.exists(tmp_file):
            os.remove(tmp_file)
        return 0, None


def change_tcp(remote_machine, tcpCC):
    subprocess.run(f'bash -c "ssh {remote_machine} sudo sysctl net.ipv4.tcp_congestion_control={tcpCC}"', shell=True, executable='/bin/bash')


    
def save_data(dropPoint, targetRTT, targetBW, tcpCC, iter, targetDomain=None):
    today = date.today().strftime("%m_%d")
    if targetDomain:
        output_path = f'./Data/{today}_dp_{dropPoint}_curl_http2/{targetDomain}/rtt_{targetRTT}ms_bdw_{targetBW}Kbps'
    else:
        output_path = f'./Data/{today}_realNebby_hk2jp/rtt_{targetRTT}ms_bdw_{targetBW}Kbps'
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    
    subprocess.run([f'cp ./output.csv {output_path}/capture_{tcpCC}_{iter}.csv'], shell=True, executable='/bin/bash')

    
def clear_setting():
    subprocess.run("./clean.sh", shell=True, executable='/bin/bash')
    

def run_launch_with_queueSize(remote_machine, targetIP, rtt, bw, tcpCC_List, iter_idx, iteration_num):

    # -------------- Nebby --------------
    delayTime_List = [50, 100]  # post_delay
    base_delay = round(rtt/2)  # base_delay
    pre_delay = 1

    TargetBW_List = [200]
    intervalTime_List = [0]
    TargetRTT_List = [round(102+rtt), round(202+rtt)]

    rtt_delay_pairs = list(zip(TargetRTT_List, delayTime_List))
    bw_interval_pairs = list(zip(TargetBW_List, intervalTime_List))

    nebby_env_list = list(itertools.product(rtt_delay_pairs, bw_interval_pairs))

    combined_env_list = nebby_env_list
    # -----------------------------------


    for (targetRTT, delayTime), (targetBW, intervalTime) in combined_env_list:
        queue_size = round((targetRTT*targetBW)/(1500*4)) # 2BDP
        
        for tcpCC in tcpCC_List:
            change_tcp(remote_machine, tcpCC)
            time.sleep(1)
            for iter in range(iteration_num): 
                print(f'tcpCC: {tcpCC}, targetRTT: {targetRTT}, targetBW: {targetBW}, queue_size: {queue_size}p, endswith: {iter_idx*iteration_num+iter}')    
                clear_setting()
                time.sleep(1)
                try:
                    subprocess.run(f'./run_test.sh {base_delay} {delayTime} {targetIP}', check=True, shell=True, executable='/bin/bash')
                except subprocess.CalledProcessError as e:
                    print(f"Error occurred: {e.stderr}")
                    continue
                save_data(0, targetRTT, targetBW, tcpCC, iter_idx*iteration_num+iter)

    

if __name__ == "__main__":
    

    targetIP = "" 
    remote_machine = ''
    
    bw, _ = measure_bw(targetIP)
    if bw is not None:
        print(f'BW between {targetIP} and host is: {bw} Kbps')
    else:
        print(f"Failed to get BW to {targetIP}")
    
    rtt = measure_rtt(targetIP)
    if rtt is not None:
        print(f'RTT between {targetIP} and host is: {rtt}ms')
    else:
        print(f"Failed to get RTT to {targetIP}")

    tcpCC_List = ["bbr", "bic", "highspeed", "htcp", "illinois", "reno", "scalable", "vegas", "veno", "westwood", "yeah", "cubic"]

    for i in range(10):
        run_launch_with_queueSize(remote_machine, targetIP, rtt, 80000, tcpCC_List, i, 1)

