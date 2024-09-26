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
from urllib.parse import urlparse, urlunparse

def measure_rtt(targetIp):
    try:
        result = subprocess.run(
            f'ping -c 4 {targetIp}',
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
    """Convert speed to Kbps based on unit."""
    if unit == "KB/s":
        return float(speed) * 8 
    elif unit == "MB/s":
        return float(speed) * 8 * 1e3
    elif unit == "GB/s":
        return float(speed) * 8 * 1e6
    else:
        return None

def convert_to_kb(size, unit):
    size = float(size)
    if unit == 'K':
        return size  
    elif unit == 'M':
        return size * 1024  
    elif unit == 'G':
        return size * 1024 * 1024 
    else:
        return size / 1024  # Assuming size is in bytes if no unit is provided

def measure_bw(targetUrl):
    error_log_file = 'errors.txt'
    tmp_file = 'bw_tmp.html'
    command = f"wget -O {tmp_file} -U \"Mozilla/5.0 (Macintosh; Intel Mac OS X 10.13; rv:62.0) Gecko/20100101 Firefox/62.0\" -t 2 -T 10 '{targetUrl}' --no-check-certificate"
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True, timeout=10)
        output = result.stdout + result.stderr
    except Exception as e:
        output = e.stderr
        print(output)
        return 0, None, None, 0

    match = re.search(r'(\d+\.?\d*)\s*(KB/s|MB/s|GB/s)', output)
    if match:
        speed = match.group(1)
        unit = match.group(2)
        speed_kbps = convert_to_kbps(speed, unit)

        size_match = re.search(r'Length: (\d+)', output)
        if size_match:
            file_size_KB = int(size_match.group(1)) / 1024 
        else:
            size_match = re.search(r'saved $$(\d+)$$', output)
            if size_match:
                file_size_KB = int(size_match.group(1)) / 1024
            else:
                file_size_KB = 0

        pattern = re.compile(r'Connecting to [^\s]+ $[^$]+\)\|([^\|]+)\|.*?\nHTTP request sent, awaiting response\.\.\. (\d+)', re.S)
        matches = pattern.findall(output)
        last_ip = None
        for ip, status in matches:
            last_ip = ip.strip()  # Update the last attempted IP address
            if status == '200':
                break  

        ok_pattern = re.compile(r'(.*)\nHTTP request sent, awaiting response\.\.\. 200 OK')
        ok_match = ok_pattern.search(output)
        
        last_domain = None
        if ok_match:
            preceding_line = ok_match.group(1).strip()
            to_pattern = re.compile(r'to ([^\s:]+)')
            to_match = to_pattern.search(preceding_line)
            if to_match:
                last_domain = to_match.group(1)

        os.remove(tmp_file)
        return speed_kbps, last_ip, last_domain, file_size_KB

    else: 
        if os.path.exists(tmp_file):
            os.remove(tmp_file)
        return 0, None, None, 0
    
def save_data(dropPoint, targetRTT, targetBW, tcpCC, iter, targetDomain=None):
    today = date.today().strftime("%m_%d")
    if targetDomain:
        output_path = f'../Data/{today}_realUrls/{targetDomain}/rtt_{targetRTT}ms_bdw_{targetBW}Kbps'
    else:
        output_path = f'../Data/{today}_dp_{dropPoint}_simulation/rtt_{targetRTT}ms_bdw_{targetBW}Kbps'
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    
    subprocess.run([f'cp ../Data/time_seqNum.csv {output_path}/host_{tcpCC}_{iter}.csv'], shell=True, executable='/bin/bash')
    subprocess.run([f'cp ../Data/combined_capture.csv {output_path}/capture_{tcpCC}_{iter}.csv'], shell=True, executable='/bin/bash')

def save_data_next_step(dropPoint, targetRTT, targetBW, tcpCC, iter, targetDomain=None):
    today = date.today().strftime("%m_%d")
    output_path = f'../Data/{today}_dp_{dropPoint}_realUrls_next_step_1/{targetDomain}/rtt_{targetRTT}ms_bdw_{targetBW}Kbps'
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
            row[0] = utc_time.strftime('%H:%M:%S.%f')  # Retain microseconds
            writer.writerow(row)

    try:
        subprocess.run(f"mv {temp_csv} {output_csv}", check=True, shell=True, executable='/bin/bash')
    except subprocess.CalledProcessError as e:
        print(f"Error occurred: {e.stderr}")


def parse_csv_and_filter(input_csv, ip_filter):  # just for debug
    count_src = 0
    count_dst = 0

    with open(input_csv, mode='r', newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        header = reader.fieldnames

        print(','.join(header))

        for row in reader:
            ip_src = row['ip.src']
            ip_dst = row['ip.dst']

            if ip_src == ip_filter[0] and ip_dst == ip_filter[1]:
                count_src += 1
                print(','.join([row[field] for field in header]))
            elif ip_src == ip_filter[1] and ip_dst == ip_filter[0]:
                count_dst += 1
                print(','.join([row[field] for field in header]))

    print(f"Count of packets with ip.src == {ip_filter[0]} and ip.dst == {ip_filter[1]}: {count_src}")
    print(f"Count of packets with ip.src == {ip_filter[1]} and ip.dst == {ip_filter[0]}: {count_dst}")

def convert_pcap_to_csv(input_pcap, ip_filter):
    print(f'ip_filter: {ip_filter}')
    output_csv_seq = "../Data/capture_seq.csv"
    output_csv_ack = "../Data/capture_ack.csv"
    combined_csv = "../Data/combined_capture.csv"

    tshark_command_seq = (
        f'tshark -r {input_pcap} -Y "ip.src == {ip_filter[0]} && ip.dst == {ip_filter[1]}" -T fields '
        f'-e frame.time -e frame.len -e tcp.len -e tcp.seq_raw '
        f'-E separator=" " -E quote=d -E occurrence=f '
        f'> {output_csv_seq}'
    )
    subprocess.run(tshark_command_seq, shell=True, executable='/bin/bash')

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

def generate_environment_combinations(rtt, bw, rtt_range, bw_range, dropPoint_List):
    """
    para:
    rtt_range (tuple): (lower, upper, stride)
    bw_range (tuple): (lower, upper, stride)
    dropPoint_List (list): [dp]
    """
    rtt_lower, rtt_upper, rtt_stride = rtt_range
    bw_lower, bw_upper, bw_stride = bw_range

    TargetRTT_List = list(range(rtt_lower, rtt_upper+1, rtt_stride))
    TargetBW_List = list(range(bw_lower, bw_upper+1, bw_stride))

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

    env_List = list(itertools.product(dropPoint_List, rtt_delay_pairs, bw_interval_pairs))

    return env_List


def encode_url(url):
    return urllib.parse.quote(url, safe=':/')

def get_real_urls(targetUrl, targetIp, targetDomain, hostIP, rtt, bw, iteration=1):
    dropPoint_List = [40]
    TargetRTT_List = list(range(180, 310, 20)) 
    TargetBW_List = list(range(200, 410, 200))
    TargetRTT_List = [x for x in TargetRTT_List if x > rtt]
    TargetBW_List = [x for x in TargetBW_List if x < bw]
    if len(TargetBW_List) == 0 or len(TargetRTT_List) == 0:
        return
    TargetRTT_List = [TargetRTT_List[0]]
    TargetBW_List = [TargetBW_List[-1]]

    delayTime_List = []
    for target_rtt in TargetRTT_List:
        delay = 1 if int((target_rtt - rtt) / 2) == 0 else int((target_rtt - rtt) / 2)
        delayTime_List.append(delay)

    intervalTime_List = []
    packet_size_bits = 1500 * 8
    for target_bitrate_kbps in TargetBW_List:
        target_bitrate_bps = target_bitrate_kbps * 1000
        packets_per_second = target_bitrate_bps / packet_size_bits
        interval_per_packet = 1 / packets_per_second * 1e6
        intervalTime_List.append(interval_per_packet)

    rtt_delay_pairs = list(zip(TargetRTT_List, delayTime_List))
    bw_interval_pairs = list(zip(TargetBW_List, intervalTime_List))
    
    env_List = list(itertools.product(dropPoint_List, rtt_delay_pairs, bw_interval_pairs))
    
    output_pcap = "../Data/capture.pcap"
    tcpCC = 'unknown'
                
    for dropPoint, (targetRTT, delayTime), (targetBW, intervalTime) in env_List:
        for iter in range(iteration): 
            print(f'=================================\ntcpCC: {tcpCC}, targetUrl: "{targetUrl}" targetIp: {targetIp} targetRTT: {targetRTT}, targetBW: {targetBW}, delayTime: {delayTime}, intervalTime: {intervalTime}\n=================================')       
            clear_setting()
            time.sleep(1)
            try:
                subprocess.run(f'./run_with_capture_realUrls.sh {delayTime} "{targetUrl}" {targetDomain} {targetIp} {intervalTime} {output_pcap}', check=True, shell=True, executable='/bin/bash')
        
            except subprocess.CalledProcessError as e:
                print(f"Error occurred: {e.stderr}")
                continue
            convert_pcap_to_csv(output_pcap, [targetIp, hostIP])  #combined_capture.csv
            save_data(dropPoint, targetRTT, targetBW, tcpCC, iter, targetDomain)

def get_real_urls_next_step(targetUrl, targetIp, targetDomain, hostIP, rtt, bw, target_rtt, target_bw, iteration = 1):
    dropPoint_List = [40]
    TargetRTT_List = [target_rtt]
    TargetBW_List = [target_bw]

    if len(TargetBW_List) == 0 or len(TargetRTT_List) == 0:
        return

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
    
    env_List = list(itertools.product(dropPoint_List, rtt_delay_pairs, bw_interval_pairs))
    #print(f'env_list: {env_List}')
    
    output_pcap = "../Data/capture.pcap"
    tcpCC = 'unknown'
                
    for dropPoint, (targetRTT, delayTime), (targetBW, intervalTime) in env_List:
            
        for iter in range(iteration): 
            print(f'=================================\ntcpCC: {tcpCC}, targetUrl: "{targetUrl}" targetIp: {targetIp} targetRTT: {targetRTT}, targetBW: {targetBW}, delayTime: {delayTime}, intervalTime: {intervalTime}\n=================================')       
            clear_setting()
            time.sleep(1)
            #print(f'./run_with_capture.sh {delayTime} "{targetUrl}" {targetDomain} {targetIp} {intervalTime} {output_pcap}')
            try:
                subprocess.run(f'./run_with_capture_realUrls.sh {delayTime} "{targetUrl}" {targetDomain} {targetIp} {intervalTime} {output_pcap}', check=True, shell=True, executable='/bin/bash')
        
            except subprocess.CalledProcessError as e:
                print(f"Error occurred: {e.stderr}")
                continue
            convert_pcap_to_csv(output_pcap, [targetIp, hostIP])  #combined_capture.csv
            save_data_next_step(dropPoint, targetRTT, targetBW, tcpCC, iter, targetDomain)

def replace_domain(url, new_domain):
    parsed_url = urlparse(url)
    new_netloc = new_domain
    new_url = urlunparse(parsed_url._replace(netloc=new_netloc))
    return new_url

if __name__ == "__main__":
    # ------------ get real data from urls ------------
    df = pd.read_csv('./webtest/full_urls_2024_v3.csv') #
    
    hostIP = "100.64.0.2"  # ingress
    for index, row in df.iterrows():
        root_url = row['root_url']
        wget_url = row['wget_url']

        pageSize, rtt, bw = 0, 0, 0
        targetIp = None
        if wget_url != "error":
            wget_url = encode_url(wget_url)
            targetDomain = re.sub(r'^https?://', '', root_url)
            bw, targetIp, targetDomain_new, pageSize = measure_bw(wget_url)
            if targetDomain_new and targetDomain != targetDomain_new:
                print(f'targetDomain_pre: {targetDomain}, targetIP_pre: {targetIp}, root_url_pre: {root_url}, wget_url_pre: {wget_url}')
                wget_url = replace_domain(wget_url, targetDomain_new)
                root_url = replace_domain(root_url, targetDomain_new)
                bw, targetIp, targetDomain, pageSize = measure_bw(wget_url)
                print(f'targetDomain_new: {targetDomain}, targetIP_new: {targetIp}, root_url_new: {root_url}, wget_url_new: {wget_url}')

            if bw != 0 and targetIp:
                print(f'BW between {wget_url}-{targetIp} and hostUrl is: {bw} Kbps')
            else:
                print(f"Failed to get BW, IP or targetDomain of {wget_url} ")

            # 测量 RTT
            rtt = measure_rtt(targetIp)
            if rtt != 0:
                print(f'RTT between {root_url}-{targetIp} and hostUrl is: {rtt} ms')
            else:
                print(f"Failed to get RTT to {root_url}-{targetIp}")
        
       
        data = [root_url, wget_url, targetIp, int(pageSize), int(rtt), int(bw)]
        if bw == 0 or rtt == 0 or pageSize < 95 or targetIp is None:
            csv_error_filename = '../webtest/full_urls_2024_v3_error.csv'
            if not os.path.exists(csv_error_filename):
                with open(csv_error_filename, mode='w', newline='') as file:
                    writer = csv.writer(file)
                    writer.writerow(['root_url', 'wget_url', 'target_ip', 'wget_url_size(KB)', 'rtt(ms)', 'bandwidth(Kbps)'])
            with open(csv_error_filename, mode='a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(data)
            continue
        else: 
            csv_filename = '../webtest/full_urls_2024_v3.csv'
            if not os.path.exists(csv_filename):
                with open(csv_filename, mode='w', newline='') as file:
                    writer = csv.writer(file)
                    writer.writerow(['root_url', 'wget_url', 'target_ip', 'wget_url_size(KB)', 'rtt(ms)', 'bandwidth(Kbps)'])

            with open(csv_filename, mode='a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(data)
        
            get_real_urls(wget_url, targetIp, targetDomain, hostIP, rtt, bw, iteration=1)

     # ------------ get real data from urls next step------------
    # df = pd.read_csv('./webtest/full_urls_2024_v3.csv') 
    
    # txt_file_path = './webtest/best_env_next_step_1.txt'
    # domain_info = {}
    # with open(txt_file_path, 'r') as file:
    #     lines = file.readlines()
    #     for line in lines:
    #         parts = line.split()
    #         domain = parts[0]
    #         target_rtt = int(parts[1][1:-1])  # "[80,"
    #         target_bw = int(parts[2][0:-1]) # "180]"
    #         domain_info[domain] = (target_rtt, target_bw)

    # df = df[df['root_url'].apply(lambda x: re.sub(r'^https?://', '', x) in domain_info)]
    # hostIP = "100.64.0.2"  # ingress
    
    # for index, row in df.iterrows():
    #     root_url = row['root_url']
    #     wget_url = row['wget_url']
    #     wget_url = encode_url(wget_url)
    #     targetDomain = re.sub(r'^https?://', '', root_url)
    #     target_rtt, target_bw = domain_info[targetDomain]
    #     print(f'targetDomain: {targetDomain}, wget_url: {wget_url}, target_rtt: {target_rtt}, target_bw: {target_bw}')

    #     bw, targetIp = measure_bw(wget_url)
    #     if bw != 0 and targetIp:
    #         print(f'BW between {wget_url}-{targetIp} and hostUrl is: {bw} Kbps')
    #     else:
    #         print(f"Failed to get BW or IP of {wget_url} ")
    #         continue

    #     rtt = measure_rtt(targetIp)
    #     if rtt != 0:
    #         print(f'RTT between {root_url}-{targetIp} and hostUrl is: {rtt} ms')
    #     else:
    #         print(f"Failed to get RTT to {root_url}-{targetIp}")
    #         continue
        
    #     get_real_urls_next_step(wget_url, targetIp, targetDomain, hostIP, rtt, bw, target_rtt, target_bw, iteration=1)