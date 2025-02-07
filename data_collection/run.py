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





def measure_base_rtt(targetIP):
    command = f'mm-delay 1 ./btl_onlybw.sh {targetIP}'

    try:
        result = subprocess.run(command, shell=True, executable='/bin/bash', check=True, capture_output=True, text=True)
        output = result.stdout
        
        match = re.search(r'rtt min/avg/max/mdev = [\d.]+/([\d.]+)/[\d.]+/[\d.]+ ms', output)
        if match:
            avg_rtt = float(match.group(1))
            avg_rtt_int = int(round(avg_rtt))
            print(f"Average RTT: {avg_rtt_int} ms")
            return avg_rtt_int
        else:
            print("Average RTT not found in the output.")
            return None
    except subprocess.CalledProcessError as e:
        print("Error:", e.stderr)
        return None

def get_bw_trace(bw):
    subprocess.run(f'./get_bwTrace.sh {bw}', shell=True, executable='/bin/bash')

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
    #subprocess.run(f'bash -c "ssh {remote_machine} sudo sysctl net.ipv4.tcp_congestion_control={tcpCC}"', shell=True, executable='/bin/bash')
    try:
        result = subprocess.run(
            f'bash -c "ssh {remote_machine} sudo sysctl net.ipv4.tcp_congestion_control={tcpCC}"',
            shell=True,
            executable='/bin/bash',
            check=True,
            capture_output=True,
            text=True
        )
        print(result.stdout)
    except subprocess.CalledProcessError as e:
        if "No such file or directory" in e.stderr:
            print("Error: No such file or directory")
            sys.exit(1)
        else:
            print(f"An error occurred: {e.stderr}")
            sys.exit(1)

    
def save_data(targetRTT, targetBW, tcpCC, iter, targetDomain=None):
    today = date.today().strftime("%m_%d")
    if targetDomain:
        output_path = f'./Data/pk2usa_cdn_{today}/{targetDomain}/rtt_{targetRTT}ms_bdw_{targetBW}Kbps'
    else:
        output_path = f'./Data/simulation_pk2hk_dctcp_{today}/rtt_{targetRTT}ms_bdw_{targetBW}Kbps'
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    if tcpCC == 'pcc':
        tcpCC = 'pccLoss'
    subprocess.run([f'cp ./capture.csv {output_path}/capture_{tcpCC}_{iter}.csv'], shell=True, executable='/bin/bash')
    subprocess.run([f'cp ./host.csv {output_path}/host_{tcpCC}_{iter}.csv'], shell=True, executable='/bin/bash')
    subprocess.run([f'cp ./capture.pcap {output_path}/capture_{tcpCC}_{iter}.pcap'], shell=True, executable='/bin/bash')
    subprocess.run([f'cp ./host.pcap {output_path}/host_{tcpCC}_{iter}.pcap'], shell=True, executable='/bin/bash')

    
def clear_setting():
    # subprocess.call(["./clean.sh"], shell=True, executable='/bin/bash')
    subprocess.run("./clean.sh", shell=True, executable='/bin/bash')
    


def run_launch(remote_machine, targetLink, targetIP, base_bw, tcpCC_List, iter_idx, iteration_num):
    # cclinguist: target_delay, target_bw, target_url
    if remote_machine:
        #targetDelay_List = list(range(40, 220, 20)) # ms
        targetDelay_List = list(range(80, 100, 20)) # ms
        #targetDelay_List = list(range(40, 60, 20)) # ms
        targetBW_List = list(range(600, 800, 200))    
        # targetDelay_List = list(range(40, 220, 80)) # ms
        # targetBW_List = list(range(400, 1100, 300))    # Kbps
        # targetDelay_List = list(range(80, 100, 80)) # ms
        # targetBW_List = list(range(400, 500, 300))    # Kbps
    else:
        # targetDelay_List = list(range(90, 220, 20)) # ms
        # targetBW_List = list(range(400, 600, 200))    # Kbps

        # ----- cdn test -------
        targetDelay_List = list(range(40, 220, 20)) # ms
        targetBW_List = list(range(400, 1100, 200))    # Kbps
        
    targetBW_List = [x for x in targetBW_List if x < base_bw]
    #targetBW_List.sort(reverse=True)
    if len(targetBW_List) == 0:
        return

    # if not remote_machine:  
    #     targetBW_List = [targetBW_List[-1]]  

    for targetBW in targetBW_List:
        get_bw_trace(targetBW)
        
        base_rtt = measure_base_rtt(targetIP)
        if not base_rtt:
            continue
        base_delay = round(base_rtt/2)  

        print(f'base_delay while bdw is {targetBW}: {base_delay}')
        targetDelay_List = [x for x in targetDelay_List if x > base_delay]
        if len(targetDelay_List) == 0:
            #return
            continue
            #targetDelay_List = [40]

        # if not remote_machine:
        #     targetDelay_List = [targetDelay_List[0]]

        for targetDelay in targetDelay_List:
            post_delay = targetDelay - base_delay
            # if post_delay <= 0:
            #     post_delay = 1
            #     print(f'-----post_delay is 1!-----')
            print(f'current profile: {targetDelay}ms,  {targetBW}Kbps')
            for tcpCC in tcpCC_List:
                time.sleep(5)
                if remote_machine:  
                    change_tcp(remote_machine, tcpCC)
                
                for iter in range(0,iteration_num): 
                    if remote_machine: 
                        change_tcp(remote_machine, tcpCC)
                #for iter in range(8,10): 
                    clear_setting()
                    time.sleep(1)
                    if targetLink:  # real Urls
                        try:
                            subprocess.run(f'./run_test.sh {post_delay} {targetLink} {targetIP} {targetBW} {targetDelay}', check=True, shell=True, executable='/bin/bash')
                            
                        except subprocess.CalledProcessError as e:
                            print(f"Error occurred: {e.stderr}")
                            continue
                        targetDomain = re.sub(r'^https?://', '', targetLink)
                        save_data(targetDelay*2, targetBW, tcpCC, iter_idx*iteration_num+iter, targetDomain)
                    else:  # simulation
                        try:
                            subprocess.run(f'./run_test.sh {post_delay} "None" {targetIP} {targetBW} {targetDelay}', check=True, shell=True, executable='/bin/bash')
                            
                        except subprocess.CalledProcessError as e:
                            print(f"Error occurred: {e.stderr}")
                            continue
                        save_data(targetDelay*2, targetBW, tcpCC, iter_idx*iteration_num+iter)
                    
                    clear_setting()





if __name__ == "__main__":
    
    #------------ get train data from our own servers wiz IP ------------


    targetIP = ""

    remote_machine = "" 
    base_bw, _ = measure_bw(targetIP)
    if base_bw is not None:
        print(f'BW between {targetIP} and host is: {base_bw} Kbps')
    else:
        print(f"Failed to get BW to {targetIP}")

    #tcpCC_List = ["bbr", "bic", "highspeed", "htcp", "illinois", "reno", "scalable", "vegas", "veno", "westwood", "yeah", "cubic"]
    #tcpCC_List = ["pcc"]
    tcpCC_List = ["dctcp"]
    for i in range(1):  
        #run_launch_with_queueSize(remote_machine, targetIP, rtt, 80000, tcpCC_List, i, 1)
        #run_launch(remote_machine, None, targetIP, base_bw, tcpCC_List, i, 1)
        run_launch(remote_machine, None, targetIP, base_bw, tcpCC_List, i, 1) 
        break

   

    # ------------ get cdn test ------------

    # df = pd.read_csv('./webtest/test_urls.csv')
    # for index, row in df.iterrows():
    #     #targetLink, targetIP, bw = row['root_url'], row['target_ip'], 8000 
    #     targetLink, targetIP, bw = row['wget_url'], row['target_ip'], 8000 
    #     # if bw != 0 and targetIp:
    #     #     print(f'BW between {targetLink}-{targetIp} and client is: {bw} Kbps')
    #     # else:
    #     #     print(f"Failed to get BW or IP of {targetLink} ")
    #     #     continue
    #     remote_machine = None

    #     tcpCC_List = ["unknown"]
    #     base_bw, _ = measure_bw(targetIP)
    #     base_bw = 8000 
    #     for i in range(3):
    #         run_launch(remote_machine, targetLink, targetIP, base_bw, tcpCC_List, i, 1)
    #         break