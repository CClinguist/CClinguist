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
        output_path = f'../Data/{today}_dp_{dropPoint}_curl_http2/{targetDomain}/rtt_{targetRTT}ms_bdw_{targetBW}Kbps'
    else:
        output_path = f'../Data/{today}_Nebby_200kbps_hk2jp/rtt_{targetRTT}ms_bdw_{targetBW}Kbps'
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    
    subprocess.run([f'cp ../Data/time_seqNum.csv {output_path}/host_{tcpCC}_{iter}.csv'], shell=True, executable='/bin/bash')
    subprocess.run([f'cp ../Data/combined_capture.csv {output_path}/capture_{tcpCC}_{iter}.csv'], shell=True, executable='/bin/bash')

    
def clear_setting():
    subprocess.run("./clean.sh", shell=True, executable='/bin/bash')
    


def convert_to_utc(output_csv):
    temp_csv = output_csv + '.tmp'
    with open(output_csv, 'r') as infile, open(temp_csv, 'w', newline='') as outfile:
        reader = csv.reader(infile, delimiter=' ')
        writer = csv.writer(outfile, delimiter=' ')

        for row in reader:
            frame_time = row[0]
            local_time = parser.parse(frame_time)
            utc_time = local_time.astimezone(timezone.utc)
            row[0] = utc_time.strftime('%H:%M:%S.%f')
            writer.writerow(row)
    try:
        subprocess.run(f"mv {temp_csv} {output_csv}", check=True, shell=True, executable='/bin/bash')
    except subprocess.CalledProcessError as e:
        print(f"Error occurred: {e.stderr}")


def convert_pcap_to_csv(input_pcap, ip_filter):
    '''
        ip_filter:[server_ip, host_ip]
        return:
        csv: [type, time, ip_length, tcp_length, raw_seq/ack_num]
        Type Time DataLength Raw_Seq/Ack        
    '''
    output_csv_seq = "../Data/capture_seq.csv"
    output_csv_ack = "../Data/capture_ack.csv"
    combined_csv = "../Data/combined_capture.csv"

    # ========= tcp_seq ========= 
    tshark_command_seq = (
        f'tshark -r {input_pcap} -Y "ip.src == {ip_filter[0]} && ip.dst == {ip_filter[1]}" -T fields '
        f'-e frame.time -e frame.len -e tcp.len -e tcp.seq_raw '
        f'-E separator=" " -E quote=d -E occurrence=f '
        f'> {output_csv_seq}'
    )
    
    # ========= tcp_all ========= 
    # tshark_command_seq = (    # get all trace to debug
    #     f'tshark -r {input_pcap} -T fields '
    #     f'-e ip.src -e ip.dst -e frame.len -e tcp.len -e tcp.seq -e tcp.ack -e _ws.col.Protocol '
    #     f'-E header=y -E separator=, -E quote=d -E occurrence=f '
    #     f'> {output_csv_seq}'
    # )
    
    subprocess.run(tshark_command_seq, shell=True, executable='/bin/bash')


    # ========= tcp_ack ========= 
    tshark_command_ack = (
        f'tshark -r {input_pcap} -Y "ip.src == {ip_filter[1]} && ip.dst == {ip_filter[0]}" -T fields '
        f'-e frame.time -e frame.len -e tcp.len -e tcp.ack_raw '
        f'-E separator=" " -E quote=d -E occurrence=f '
        f'> {output_csv_ack}'
    )
    
    subprocess.run(tshark_command_ack, shell=True, executable='/bin/bash')

    convert_to_utc(output_csv_seq)
    convert_to_utc(output_csv_ack)
    combined_data = []
    with open(output_csv_seq, 'r') as seqfile:
        csvreader = csv.reader(seqfile, delimiter=' ')
        for row in csvreader:
            time, ip_len, tcp_len, seq = row
            combined_data.append(['seq', time, ip_len, tcp_len, seq])

    with open(output_csv_ack, 'r') as ackfile:
        csvreader = csv.reader(ackfile, delimiter=' ')
        for row in csvreader:
            time, ip_len, tcp_len, ack = row
            combined_data.append(['ack', time, ip_len, tcp_len, ack])


    combined_data.sort(key=lambda x: datetime.strptime(x[1], '%H:%M:%S.%f'))
    

    with open(combined_csv, 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile, delimiter=' ')
        csvwriter.writerow(['Type', 'Time', 'Length', 'DataLength', 'Raw_Seq/Ack'])
        csvwriter.writerows(combined_data)

    #print(f"Combined CSV file saved as {combined_csv}")





def find_nearest_power_of_two(n):
    if n <= 0:
        return 1
    lower_power = 2**math.floor(math.log2(n))
    upper_power = 2**math.ceil(math.log2(n))
    if n - lower_power < upper_power - n:
        return lower_power
    else:
        return upper_power


def run_launch_with_queueSize(remote_machine, targetIP, hostIP, rtt, bw, tcpCC_List, iter_idx, iteration_num):
    # -------------- CCAnalyzer --------------
    TargetRTT_List = [85, 130, 275]
    TargetBW_List = [5000]

    TargetRTT_List = [x for x in TargetRTT_List if x > rtt]
    TargetBW_List = [x for x in TargetBW_List if x < bw]

    if len(TargetBW_List) == 0 or len(TargetRTT_List) == 0:
        return None
    
    delayTime_List = []
    for target_rtt in TargetRTT_List:
        delay = 1 if int((target_rtt - rtt) / 2) == 0 else int((target_rtt - rtt) / 2)
        delayTime_List.append(delay)

    intervalTime_List = []
    packet_size_bits = 1500 * 8
    for target_bitrate_kbps in TargetBW_List:
        target_bitrate_bps = target_bitrate_kbps * 1000
        packets_per_second = target_bitrate_bps / packet_size_bits
        interval_per_packet = 1 / packets_per_second * 1e6  # us
        intervalTime_List.append(interval_per_packet)

    rtt_delay_pairs = list(zip(TargetRTT_List, delayTime_List))
    bw_interval_pairs = list(zip(TargetBW_List, intervalTime_List))


    ccanalyzer_env_list = list(itertools.product(rtt_delay_pairs, bw_interval_pairs))

    combined_env_list = ccanalyzer_env_list
    # -----------------------------------


    	
    output_pcap = "../Data/capture.pcap" 


    for (targetRTT, delayTime), (targetBW, intervalTime) in combined_env_list:
        queue_size = find_nearest_power_of_two(round((targetRTT*targetBW)/(1500*8)))  # 1BDP & 2^n
        #queue_size = round((targetRTT*targetBW)/(1500*4)) # 2BDP
        
        for tcpCC in tcpCC_List:
            change_tcp(remote_machine, tcpCC)
            time.sleep(1)
            for iter in range(iteration_num): 
                print(f'tcpCC: {tcpCC}, targetRTT: {targetRTT}, targetBW: {targetBW}, queue_size: {queue_size}p, delayTime: {delayTime}, intervalTime: {intervalTime}, endswith: {iter_idx*iteration_num+iter}')    
                clear_setting()
                time.sleep(1)
                
                try:
                    subprocess.run(f'./run_with_capture_simulation_with_queueSize.sh {delayTime} {targetIP} {intervalTime} {output_pcap} {queue_size}', check=True, shell=True, executable='/bin/bash')
                except subprocess.CalledProcessError as e:
                    print(f"Error occurred: {e.stderr}")
                    continue
                
                convert_pcap_to_csv(output_pcap, ['', hostIP])  #combined_capture.csv
                save_data(0, targetRTT, targetBW, tcpCC, iter_idx*iteration_num+iter)


if __name__ == "__main__":
    
    #------------ get train data from our own servers wiz IP ------------
    targetIP = "" 
    hostIP = "100.64.0.2"  # ingress
    remote_machine = '' 
    
    bw, _ = measure_bw(targetIP)
    if bw is not None:
        print(f'BW between {targetIP} and {hostIP} is: {bw} Kbps')
    else:
        print(f"Failed to get BW to {targetIP}")
    
    rtt = measure_rtt(targetIP)
    if rtt is not None:
        print(f'RTT between {targetIP} and {hostIP} is: {rtt}ms')
    else:
        print(f"Failed to get RTT to {targetIP}")

    targetIP = targetIP + "/100MB_test"
    tcpCC_List = ["bbr", "bic", "highspeed", "htcp", "illinois", "reno", "scalable", "vegas", "veno", "westwood", "yeah", "cubic"]
    for i in range(8):
        run_launch_with_queueSize(remote_machine, targetIP, hostIP, rtt, 100000, tcpCC_List, i, 1)

