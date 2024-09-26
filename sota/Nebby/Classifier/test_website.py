'''
source from: https://github.com/NUS-SNL/Nebby
'''




pre_i=2
post_i=3
bw_i=4
bf_i=5

import csv
import matplotlib.pyplot as plt
import sys
import math

import matplotlib.pyplot as plt
from scipy.fft import rfft, rfftfreq
from scipy.fft import irfft
import numpy as np
import statistics
from statistics import mean, pstdev
import pandas as pd

yellow = '\033[93m'
green = '\033[92m'
red = '\033[91m'
blue = '\033[94m'
pink = '\033[95m'
black = '\033[90m'

ccs = ['bic', 'dctcp', 'highspeed', 'htcp', 'lp', 'nv', 'scalable', 'vegas', 'veno', 'westwood', 'yeah', 'cubic', 'reno']


def smoothen(time, data, rtt):
    # Smoothening 
    left = 0
    right = 0
    run_sum = 0
    avg_data = []
    new_time = []
    roll_time = time
    roll_data = data
    while right < len(roll_time):
        while(right < len(roll_time) and (roll_time[right]-roll_time[left] < 2*rtt)):
            run_sum+=roll_data[right]
            right+=1
        new_time.append(float(roll_time[right-1]+roll_time[left])/2)
        avg_data.append(float(run_sum)/(right-left))
        run_sum-=roll_data[left]
        left+=1
    return new_time, avg_data


def get_fft(data):
    n = len(data)
    data_step = 0.002
    yf = rfft(data)
    xf = rfftfreq(n,data_step)
    return yf,xf

def get_fft_smoothening(data, time, ax,rtt,p):
    rtt=rtt
    yf, xf = get_fft(data)
    thresh  = (1/rtt)
    thresh_ind = 0
    for i in range(len(xf)) :
        freq = xf[i]
        if(freq > thresh):
            thresh_ind = i
            break
            
    yf_clean = yf
    yf_clean[thresh_ind+1:] = 0
    new_f_clean = irfft(yf_clean)
    start_len = len(time) - len(new_f_clean)

    plot_data = new_f_clean
#     if p=="y":
#         ax.plot(time[start_len:], plot_data, 'k', label='FFT smoothening', linewidth=1.5)

    plot_time = time[start_len:] 
    return plot_time, plot_data

def plot_d(ax, time, data, c, l, alpha=1):
    ax.plot(time, data, color=c, lw=2, label = l,alpha=alpha)


def srs_log(message):
    print("srs log: ", message)

def plot_one_bt(f, p,t=1):
    # print(f)
    file_name = f.split("/")[-1]
    fs = file_name.split("-")

    pre = int(fs[2])
    post = int(fs[3])
    # print("attention", pre, post)
    rtt = float(((pre+post)*2))/1000
    ax = 0
    # srs_log("alright in plot ont bt?")
    if t==1:
        # srs_log("t-=1?")
        data, time, retrans = get_window(f,"n",t)
    elif t==2:
        # srs_log("t=2?")
        data, time, retrans, OOA, DA = get_window(f,"n",t)
    if p == 'y':
        fig, ax = plt.subplots(1,1, figsize=(15,8))
        for t in retrans :
            plt.axvline(x = t, color = 'm',alpha=0.5)
        if t == 2:
            for t in OOA :
                plt.axvline(x = t, color = 'k', lw=2)
            for t in DA:
                plt.axvline(x = t, color = 'g', lw=0.5, alpha = 0.5)
        plot_d(ax,time,data, "r","Original")
        plt.close()
    
    time, data = get_fft_smoothening(data, time, ax,rtt,"y")
    # srs_log("alright in plot ont bt a?")
    #         plot_d(ax,time,data,"b","FFT Smoothened" )
#         print(len(time), len(data))
    time, data = smoothen(time, data, rtt)
#         print(len(time), len(data))
    if p == 'y':
        plot_d(ax, time, data, "b", "Smoothened",alpha=0.5)
        ax.legend()
#             plt.savefig("./plots/"+f+".png")
        plt.title("FFT Smoothened vs Original")
        # plt.show()
#     return time, data, grad_time, grad_data, rtt
#     print("Black : OOA, Green : DA, Magenta : RP"
    return time, data, retrans, rtt 


def get_time_features(retrans,time,rtt):
    time_thresh = 5*rtt
    features = []
    for i in range(1, len(retrans)):
        a=retrans[i]-retrans[i-1]
        if retrans[i]-retrans[i-1] >= time_thresh:
            features.append([retrans[i-1], retrans[i]])
    # add a feature that finished when the experiment ends
    if len(retrans)>0:
        if time[-1] - retrans[-1] > 20*rtt :
            features.append([retrans[-1],time[-1]])
    return features

def get_features(time, features):
    left = 0
    right = 0
    feature_index = 0
    in_feature = 0
    index_features = []
    while right < len(time) and feature_index < len(features): 
        if in_feature == 0 and time[right]>=features[feature_index][0]:
            in_feature = 1
            left = right
        elif in_feature == 1 and time[right] > features[feature_index][1]:
            in_feature = 0
            index_features.append([left, right-1])
            feature_index+=1
        right+=1
    if in_feature == 1:
        index_features.append([left, right-1])
    return index_features

def get_plot_features(curr_file, p):
    time, data, retrans, rtt = plot_one_bt(curr_file,p=p,t=1)
    #print(time, data, rtt)
    print("retrans", retrans)
    time_features = get_time_features(retrans,time,rtt)
    print("time_features in plot", time_features)
    features = get_features(time, time_features)
    print("features in plot_features", features)    
    if p == 'y':
        fig, ax = plt.subplots(1,1, figsize=(15,8))
        plot_d(ax, time, data, "b", "Smoothened")
        for ft in features : 
#             print(time[ft[1]]-time[ft[0]])
            ax.plot(time[ft[0]:ft[1]+1], data[ft[0]:ft[1]+1], color = 'r')
        plt.title("Red features")
        # plt.show()
        plt.close()
    return time, data, features


SHOW=True
MULTI_GRAPH=False
SMOOTHENING=False
ONLY_STATS=False
s_factor=0.9


PKT_SIZE = 88


'''
TODO: 
o Add functionality where you only plot flows that send more than x bytes of data
o Sort stats and graphs by flow size
o Organize plots by flow size (larger flows have larger graphs)
o Custom smoothening function
'''

fields=["Type", "Time", "DataLength", "Raw_Seq/Ack"]

class pkt:
    contents=[]
    def __init__(self, fields) -> None:
        self.contents=[]
        for f in fields:
            self.contents.append(f)

    def get(self, field):
        return self.contents[fields.index(field)]
        
from datetime import datetime

def process_flows(cc, dir,p="y"):
    name = dir+cc+".csv"
    with open(name) as csv_file:
        print("Reading "+name+"...")
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        '''
        Flow tracking:
        o Identify all packets that are either sourced from or headed to 100.64.0.2
        o Group different flows by client's port
        '''
        flows={}
        data_sent=0
        # ACK and RTX measurement
        ooa = set()
        rp = set()
        ooaCount=0
        rpCount=0
        daCount=0
        backAck=0
        start_time = 0.0
        flag=False
        for row in csv_reader:
            reTx=0
            
            ooAck=0
            dupAck=0
            retPacket=0
            
            
            packet=pkt(row)
            ackPkt=False
            validPkt=False
            if flag==False:
                if row[0]=='seq' and int(row[-1])>0 and int(row[-2])>0:   #排除建立连接的过程，从seq包长度不为0开始
                    flag=True
                continue
            time_interval = float(packet.get("Time"))
            if packet.get("Type") == "ack": 
                # we care about this ACK packet
                validPkt=True
                ackPkt=True
                port=443
                #PORTCHECK
#                 if int(port) != 50468:
#                     continue
                if port not in flows:
                    flows[port]={"OOA":[],"DA":[],"max_seq":0,"loss_bif":0,"max_ack":int(packet.get("Raw_Seq/Ack")), "act_times":[],"times":[], "windows":[], "cwnd":[], "bif":0, "last_ack":0, "last_seq":0, "pif":0, "drop":[], "next":0, "retrans":[]}
                else:
                    # check for Out of Order Ack (OOA)
                    if int(packet.get("Raw_Seq/Ack")) <= int(flows[port]["max_ack"]):
                        if int(packet.get("Raw_Seq/Ack")) == backAck :
                            dupAck = True
                            flows[port]["DA"].append(time_interval)
                        else :
                            ooAck = True
                            flows[port]["OOA"].append(time_interval)
                        backAck = int(packet.get("Raw_Seq/Ack"))
                    # update max_ack
                    flows[port]["max_ack"] = max(flows[port]["max_ack"], int(packet.get("Raw_Seq/Ack")))
                    if int(packet.get("Raw_Seq/Ack")) < flows[port]["max_ack"]:
                        reTx += int(packet.get("DataLength"))
#                     flows[port]["times"].append(float(packet.get("frame_time_rel")) )
                    
            elif packet.get("Type") == "seq":
                #we care about this Data packet
                validPkt=True
                port=5000
                #PORTCHECK
#                 if int(port) != 50468:
#                     continue
                if port not in flows:
                    flows[port]={"OOA":[],"DA":[],"max_seq":int(packet.get("Raw_Seq/Ack")),"loss_bif":0,"max_ack":0,"act_times":[], "times":[], "windows":[], "cwnd":[], "bif":0, "last_ack":0, "last_seq":0, "pif":0, "drop":[], "next":0, "retrans":[]}
                
                else:
                    flows[port]["max_seq"] = max(flows[port]["max_seq"], int(packet.get("Raw_Seq/Ack")))
                
                
                if int(packet.get("Raw_Seq/Ack")) < flows[port]["max_seq"] :
                    retPacket = True
                    flows[port]["retrans"].append(flows[port]["times"][-1])
                    
            if validPkt==True:
                bif = 0
                normal_est_bif = int(flows[port]["max_seq"]) - int(flows[port]["max_ack"]) + PKT_SIZE#+ reTx
                loss_est_bif = flows[port]["loss_bif"]
                if ackPkt and dupAck and len(flows[port]["windows"]) > 10:
                    if dupAck:
                        # if we have received a duplicate ack then we need to reduce the bytes in flight by packet size
                        # we also increase max ack to correct for the consolidated ack being sent later
                        loss_est_bif = int(flows[port]["windows"][-1]) - PKT_SIZE
                        flows[port]["max_ack"] += PKT_SIZE
                        
                        bif = min( normal_est_bif, loss_est_bif)
                        if p == "y" :
                            print(green+"Duplicate Ack",int(packet.get("Raw_Seq/Ack")),"Max Ack",flows[port]["max_ack"],"BIF",bif)
                        
#                     elif ooAck:
#                         # first out of order ack that we have recieved not a duplicated ack
#                         # the reason would be restransimitted packet so dont need to correct for this
                        
                elif ackPkt :
                    loss_est_bif = normal_est_bif 
                    bif = normal_est_bif    
                    if ooAck :
                        ooaCount+=1
                        if p == "y":
                            print(red+"Out of Order Ack",int(packet.get("ack")),"Max Ack",flows[port]["max_ack"],"BIF",normal_est_bif)
                        ooa.add(int(packet.get("Raw_Seq/Ack")))
                    else:
                        if p == "y":
                            print(black+"Inorder Ack",int(packet.get("ack")),"Max Seq",flows[port]["max_seq"],"BIF",normal_est_bif)
                else :
                    bif = normal_est_bif
                    if retPacket==True:
                        rpCount+=1
                        rp.add(int(packet.get("Raw_Seq/Ack")))
                        if p == "y":
                            print(pink+"Retransmitted Packet",int(packet.get("seq")), "Next", flows[port]["max_seq"]+PKT_SIZE, "BIF",bif)
                    else :
                        if p == "y":
                            print(blue+"Inorder Packet", int(packet.get("seq")), "Next", flows[port]["max_seq"]+PKT_SIZE, "BIF",bif)
                flows[port]["loss_bif"] = loss_est_bif
                flows[port]["windows"].append( int(bif) )
                flows[port]["times"].append( time_interval )
                
#                 if ackPkt and dupAck and len(flows[port]["windows"]) > 10: # we have received atleast the first window
# #                     if len(flows[port]["windows"]) < 2000: # print reTx in first 200 packets
# #                         print( packet.get("ack"), flows[port]["max_ack"])
#                     loss_est_bif = int(flows[port]["windows"][-1]) - PKT_SIZE
#                     flows[port]["max_ack"] += PKT_SIZE
#                     bif = min( normal_est_bif, loss_est_bif )
#                 elif ackPkt:
#                     loss_est_bif = normal_est_bif 
#                     bif = normal_est_bif
#                 else:
#                     bif = normal_est_bif
#                 flows[port]["loss_bif"] = loss_est_bif
#                 flows[port]["windows"].append( int(bif) )
            
            
            line_count+=1
        if p == "y":
            print("Out of Order Acks",len(ooa),"Retransmitted Packets",len(rp))
            print("Count Out of Order Acks",ooaCount,"Retransmitted Packets",rpCount)
            print("OOA",ooa,"RP",rp)
    return flows

def custom_smooth_function():
    pass

def get_flow_stats(flows):
    num=len(flows.keys())
    print("FLOW STATISTICS: \nNumber of flows: ", num)
    print("------------------------------------------------------------------------------")
    print('%6s'%"port", '%15s'%"SrcIP", '%8s'%"SrcPort",  '%8s'%"duration",  '%8s'%"start",  '%8s'%"end", '%8s'%"Sent (B)", '%8s'%"Recv (B)",)
    for k in flows.keys():
        print('%6s'%k, '%15s'%flows[k]["serverip"], '%8s'%flows[k]["serverport"], '%8s'%str('%.2f'%(flows[k]["times"][-1]-flows[k]["times"][0])), '%8s'%str('%.2f'%flows[k]["times"][0]), '%8s'%str('%.2f'%flows[k]["times"][-1]), '%8s'%flows[k]["last_seq"], '%8s'%flows[k]["last_ack"])
        #print("    * Flow "+str(k)+": ", flows[k]["last_ack"], " ", flows[k]["last_seq"], " bytes transfered.")
    return num


def split_path(f):
    path = f.split("/")
    file_name = path[-1][:-4]
    folder_path = "/".join(path[:-1])
    folder_path = folder_path + "/"
    algo_cc = file_name
    return algo_cc, folder_path

def get_window(f,p,t=1):
    algo_cc, folder_path = split_path(f)
    # print(algo_cc, folder_path)
    flows = process_flows(algo_cc, PATH,p=p)
    # srs_log("alright after process flows?")
#     flows = process_flows(algo_cc, "../measurements/m/",p=p)
#     flows = process_flows(algo_cc, "./Nebby/measurements/",p=p)    
#     flows = process_flows(algo_cc, "./Nebby/measurements-new-btl/50-200-2-60/",p=p)
#     flows = process_flows(algo_cc, "./Nebby/measurements-100-150/",p=p)
    data = []
    time = []
    drops = []
    retrans = []
    OOA = []
    DA = []
    use_port = 0
    maxx = 0
    print("All Ports : ", flows.keys())
    for port in flows.keys():
        if len(flows[port]['windows']) > maxx:
            maxx = len(flows[port]['windows'])
            use_port = port
    print("Port",use_port)
    data = flows[use_port]['windows']
    time = flows[use_port]['times']
    retrans = flows[use_port]['retrans']
    OOA = flows[use_port]['OOA']
    DA = flows[use_port]['DA']
    if p == "y":
        plt.plot(time, data)
#     time_index = len(time)-1
#     for index in range(len(flows[use_port]['times'])-1):
#         if flows[use_port]['times'][index+1] - flows[use_port]['times'][index] > :
#             time_index = index
#     time_last = flows[use_port]['times'][time_index]
#     data = data[:time_index+1]
#     time = time[:time_index+1]
    if t==2:
        return data, time, retrans, OOA, DA
    if t==1:
        return data, time, retrans


def getProbes(time, data, rtt, bdp, bw=200):
    if bw==200:
        thresh = 8*rtt
        st_thresh = 0.025
        error = 0.08
        alpha = 1.10
    if bw==1000:
        thresh = 4*rtt
        st_thresh = 0.025
        error = 0.02
        alpha = 1
    probe_index = []
    left = 0
    right = 0
    bdp_thresh = bdp/2
    print("bdp: ", bdp_thresh, bdp)
    prev_right = 0
    end = 0
    while right < len(data):
        while right < len(data) and (time[right]-time[left]) < thresh:
            right+=1
        if right == len(data):
            end = 1
            right-=1
        mid = math.floor(left + (right - left)/2)
        mid_val = 0
        left_mid = mid
        right_mid = mid
#         print(left,right)
#         while left_mid > 0 and abs(time[mid]-time[left_mid])<rtt:
#             if data[left_mid] > data[left] and data[left_mid] > data[right]:
#                 mid_val+=1
#             left_mid-=1
#         while right_mid < len(data)-1 and abs(time[right_mid]-time[mid])<rtt:
#             if data[right_mid] > data[left] and data[right_mid] > data[right]:
#                 mid_val+=1
#             right_mid+=1
#         total = right_mid - left_mid
#         peak = 0
#         if round(float(mid_val)/total,2) > 0.99 :
#             print(round(float(mid_val)/total,2))
#             peak=1
        go = 0
        if data[mid] > data[left] and data[mid] > data[right]:
#         if peak == 1:
            #this  is  peak
#             go = 1
            t_l = left
            t_r = right
            while(t_l > 0 and time[left]-time[t_l] < thresh/2):
                t_l-=1
            while(t_r < len(data)-1 and time[t_r]-time[right] < thresh/2):
                t_r+=1
            left_sd = round(np.std(data[t_l:left])/(bdp_thresh*2),3)
            right_sd = round(np.std(data[right:t_r])/(bdp_thresh*2),3)
            if float(abs(data[left]-data[right]))/data[left] < error:
                # this has the left and right points not too different from each other
                side_avg = float((data[left]+data[right]))/2
                local_max = max(data[left:right+1])
#                 go = 1
#                 print("Yes")
                if float(local_max)/side_avg > alpha and local_max > bdp_thresh: 
                    # this means that the peak is quite steep
#                     print("This")
#                     go = 1
                    if (left_sd < st_thresh) and (right_sd < st_thresh):
                        # this means that the lest and right are quite stable respectvely
                        go = 1
#                         print("Out", left_sd, right_sd, st_thresh)
        if go == 1: 
            try :
                probe_index.append([left,right,float(local_max)/side_avg, time[right]-time[left], left_sd, right_sd, t_l,t_r])
            except :
                probe_index.append([left,right])
            #Once you have found something you directly move past it
            left = right-1
        if end:
            right+=1
        left+=1
    return probe_index


def checkBBR(files,p="n"):
    classi = []
    for f in files:
        file_name = f.split("/")[-1]
        file_name = file_name[:-4]
        print(file_name)
        para = file_name.split("-")
        rtt = (int(para[3])+int(para[2]))*2
        bw = int(para[4])
        bf = int(para[5])
        bdp = float(bw*rtt*bf)/8
        # print("??",bw, bf)
        if bw == 1000 :
            l=5
            r=15
        if bw == 200:
            # print("?")
            l=10
            r=20
        # print("srs error test1: ")
        time, data, retrans,rtt = plot_one_bt(f,p="n",t=1)
        # print("srs error test1.2:")
        probe_index = getProbes(time, data, rtt, bdp)
        if p=="y":
            print_red(time, data, probe_index)
        prev = 0
        time_dis = []
        # print("srs error test2:")
        for p in probe_index:
            curr_ind = data.index(max(data[p[0]:p[1]+1]))
            if curr_ind > p[1] or curr_ind < p[0]:
                print("Something Wrong")
            if prev != 0:
                time_dis.append(abs(time[curr_ind]-prev))
            prev = time[curr_ind]
#             print(time_dis)
        count = 0
        isBBR = 0
        # print("srs error test3:")
        for t in time_dis : 
            if  t > l*rtt and t < r*rtt :
                count+=1
                if count >= 2: # there are three peaks consecutively
                    isBBR=1
                    break
            else:
                count = 0
        # print("srs error test4:")
        if isBBR:
            classi.append("YES BBR")
        else:
            if len(probe_index) <= 1 :
                classi.append("NO BBR")
            else:
                classi.append("MAYBE BBR")
    return classi

def print_red(time,data,probe_index):
    fig, ax = plt.subplots(1,1, figsize=(15,8))
    offset=0
    # ax.title("BBR Check Graph")
    ax.plot(time[offset:],data[offset:])
    plt.title("BBR Checking")
    # for t in retrans :
    #     if t > time[offset]:
    #         plt.axvline(x = t, color = 'm',alpha=0.5)
    # ax.plot(new_time[offset:], new_data[offset:])
    for p in probe_index:
        ax.plot(time[p[0]:p[1]+1], data[p[0]:p[1]+1], color='r', lw=2)
    # plt.show()
    plt.close()

def getDivision(classi, test_files):
    yes = []
    no = []
    maybe = []
    nan = []
    for i in range(len(classi)):
        if classi[i] == "YES BBR":
            yes.append(test_files[i])
        elif classi[i] == "NO BBR":
            no.append(test_files[i])
        elif classi[i] == "MAYBE BBR":
            maybe.append(test_files[i])
        else:
            nan.append(test_files[i])
    return yes, no,maybe, nan

import bisect
def lower_bound(arr, target):
    index = bisect.bisect_left(arr, target)
    return index

def sample_data_time(time, data, ss, m):
    curr_time, curr_data = adjust(time, data)
    tp = curr_time[len(curr_time)-1] - curr_time[0]
    step = tp/m
    samp_time = [curr_time[0] + i*step for i in range(m)]
    x = np.random.uniform(0,math.pi,ss)
    tr_x = np.cos(x)
    tr_x += 1
    tr_x *= (m-1)/2
    ind = [int(round(e, 0)) for e in tr_x]
    sort_ind = sorted(ind)
    tr_time = [samp_time[i] for i in sort_ind]
    new_time = []
    new_data = []
    for t in tr_time :
        i = lower_bound(curr_time, t)
        temp_t = 0
        temp_d = 0
        if round(t,6) == round(curr_time[i],6):
            temp_t = curr_time[i]
            temp_d = curr_data[i]
        else : 
            if i == 0 :
                temp_t = curr_time[i]
                temp_d = curr_data[i]
            elif i == len(curr_time)-1 :
                temp_t = curr_time[i]
                temp_d = curr_data[i]
            else :
                temp_t = (curr_time[i-1] + curr_time[i])/2
                temp_d = (curr_data[i-1] + curr_data[i])/2
        new_time.append(temp_t)
        new_data.append(temp_d)
    new_data.insert(0,curr_data[0])
    new_data.append(curr_data[-1])
    new_time.insert(0,curr_time[0])
    new_time.append(curr_time[-1])
#     return curr_time, curr_data
    return new_time, new_data

def adjust(time, data):
    start = data.index(min(data[:int(len(data)/2)]))
    end = data.index(max(data[int(len(data)/2):]))
#     print("Difference in max and min ", end-start)
    if end - start <= 0: 
        return time, data
    new_time = time[start:end+1]
    new_data = data[start:end+1]
    return new_time, new_data

# Taking 100 as the threshold is fine 
# Now we have to see how the graphs look if we use it. 
# var = ["reno", "cubic", "bbr"]
def getRed_R(files,ss=125,p="y", ft_thresh=100):
    results = []
    for curr_file in files : 
        # print(curr_file)
        file, folder_path = split_path(curr_file)
        f_split = file.split("-")
        v = f_split[0]
        # print("attention!", f_split, v)
        rtt = float((int(f_split[pre_i]) + int(f_split[post_i]))*2)/1000
        bdp = float(rtt*1000*int(f_split[bw_i])*int(f_split[bf_i]))/8
        time, data, features = get_plot_features(curr_file, p=p)
        count = 1
        # print("features", features)
        for ft in features : 
            if count > ft_thresh:
                break
            curr_time = time[ft[0]:ft[1]+1]
            curr_data = data[ft[0]:ft[1]+1]
            tr_time, tr_data = sample_data_time(curr_time, curr_data, ss, 1000)
            tr_time_pd = pd.DataFrame(tr_time)
            tr_data_pd = pd.DataFrame(tr_data)
            tr_time = list(tr_time_pd.rolling(25, center=True).mean().dropna()[0])
            tr_data = list(tr_data_pd.rolling(25, center=True).mean().dropna()[0])
            print("Feature Length ", len(tr_data))
            if p == "y" :
                plt.title(v)
                plt.plot(curr_time, curr_data, c='b', alpha = 0.5, lw = 5)
                plt.plot(tr_time, tr_data, c='r', alpha = 1)
                plt.scatter(tr_time, tr_data, c='k')
#                 plt.scatter(tr_time, tr_data, c='r', s=10)
                
                # plt.show()
            results.append(
                {v+"_"+"data"+"_"+str(count):tr_data,
                     v+"_"+"time"+"_"+str(count):tr_time,
                        v+"_"+"rtt"+"_"+str(count):rtt, 
                             v+"_"+"bdp"+"_"+str(count):bdp})
            count+=1
    return results

def getRed(files,ss=125,p="y", ft_thresh=100):
    results = []
    for file in files :   
        f_split = file.split("-")
        v = f_split[0] + "-" + f_split[1]
        rtt = float((int(f_split[2]) + int(f_split[3]))*2)/1000
        bdp = float(rtt*1000*int(f_split[4])*int(f_split[5]))/8
        # print(file)
        # print("RTT",rtt,"BDP",bdp)
        time, data, features = get_plot_features(file, p=p)
        count = 1
        for ft in features : 
            if count > ft_thresh:
                break
            curr_time = time[ft[0]:ft[1]+1]
            curr_data = data[ft[0]:ft[1]+1]
            tr_time, tr_data = sample_data_time(curr_time, curr_data, ss, 1000)
            tr_time_pd = pd.DataFrame(tr_time)
            tr_data_pd = pd.DataFrame(tr_data)
            tr_time = list(tr_time_pd.rolling(25, center=True).mean().dropna()[0])
            tr_data = list(tr_data_pd.rolling(25, center=True).mean().dropna()[0])
            print("Feature Length ", len(tr_data))
            if p == "y" :
                plt.plot(curr_time, curr_data, c='b', alpha = 0.5, lw = 5)
                plt.plot(tr_time, tr_data, c='r', alpha = 1)
                plt.scatter(tr_time, tr_data, c='k')
#                 plt.scatter(tr_time, tr_data, c='r', s=10)
                plt.title(v)
                # plt.show()
            results.append(
                {v+"_"+"data"+"_"+str(count):tr_data,
                     v+"_"+"time"+"_"+str(count):tr_time,
                        v+"_"+"rtt"+"_"+str(count):rtt, 
                             v+"_"+"bdp"+"_"+str(count):bdp})
            count+=1
    return results
# results = getRed(var)

def get_degree_all(time,data, p="n", max_deg=3):
    p_net = []
    mse_l = []
    fit_net = []
    for d in range(1,max_deg+1):
        p_temp = np.polyfit(time,data,d)
        p_net.append(p_temp)
        fit_net.append(np.polyval(p_temp,time))
        mse_l.append(mse(data,fit_net[-1]))
    if p =='y':
#         print("1 ", p1, "MSE ", mse(data, fit_l))
        plt.plot(time, data,c='k',label='Truth')
#         plt.plot(time, fit_l)
        for d in range(0, max_deg):
            plot_label = "degree"+str(d+1)
            plt.plot(time, fit_net[d],label="degree" + str(d+1))
        plt.title("Fitting the different degree polynomial")
        plt.legend()
        # plt.show()
    return max_deg,p_net, mse_l

MAX_DEG=3
from sklearn.metrics import mean_squared_error as mse
def get_degree(time,data, p="n", max_deg=MAX_DEG):
    p_net = []
    mse_l = []
    fit_net = []
    for d in range(1,max_deg+1):
        p_temp = np.polyfit(time,data, d)
        p_net.append(p_temp[0:-1])
        fit_net.append(np.polyval(p_temp,time))
        mse_l.append(mse(data,fit_net[-1]))
    if p =='y':
#         print("1 ", p1, "MSE ", mse(data, fit_l))
        plt.plot(time, data,c='k',label='Truth')
#         plt.plot(time, fit_l)
        for d in range(max_deg-1, max_deg):
            plot_label = "degree"+str(d)
            plt.plot(time, fit_net[d],label="degree" + str(d+1))
        plt.legend()
        # plt.show()
    return max_deg,p_net[max_deg-1], mse_l

def normalize(time, data, rtt, bdp):
    new_time = time - min(time)
    new_data = data - min(data)
    
    new_time = (new_time/rtt)
#     new_time = (new_time/max(new_time))*10
    new_data = (new_data/max(new_data))*10
    
    return new_time, new_data
    
from statistics import mean 
def get_feature_degree_R(files,ss=225,p='n',ft_thresh=3,max_deg=MAX_DEG):
    results = getRed_R(files,ss,p=p,ft_thresh=ft_thresh)
    # print("results? ", results)
    count_features = 0
    mp = {}
    for item in results :
        for ele in list(item.keys()):
            name_list = ele.split("_")
            website = name_list[0]
            if "data"==name_list[1] :
                curr_data = np.array(item[ele])
            if "time"==name_list[1] :
                curr_time = np.array(item[ele])
            if "rtt"==name_list[1] :
                curr_rtt = item[ele]
            if "bdp"==name_list[1] :
                curr_bdp = item[ele]
#         print(website)
#         print(curr_data, curr_time)
        curr_time, curr_data = normalize(curr_time, curr_data, curr_rtt, curr_bdp)
        count_features += 1
        max_deg,p_net,mse_l = get_degree_all(curr_time, curr_data,p=p,max_deg=max_deg)
        mp[website] = {
        'data':curr_data,
        'time':curr_time,
        "max_deg":max_deg,
        "p_net":p_net,
        "mse_l":mse_l
    }
    return mp

from statistics import mean 
def get_feature_degree(files,ss=225,p='n',ft_thresh=3,max_deg=MAX_DEG):
    results = getRed(files,ss,p=p,ft_thresh=ft_thresh)
    errors = []
    mp = {}
    cc_mp = {}
    count_features = 0
    for item in results :
        for ele in list(item.keys()):
            name_list = ele.split("-")
            cc = name_list[0]
            name = name_list[0] + name_list[-1]
            if "data" in ele :
                curr_data = np.array(item[ele])
            if "time" in ele :
                curr_time = np.array(item[ele])
            if "rtt" in ele :
                curr_rtt = item[ele]
            if "bdp" in ele :
                curr_bdp = item[ele]
        curr_time, curr_data = normalize(curr_time, curr_data, curr_rtt, curr_bdp)
        count_features += 1
        print("Name :",name)
        degree, coeff, error_item = get_degree(curr_time, curr_data,p=p,max_deg=max_deg)
        # Adding time feature here
#         coeff_list = list(coeff)
#         coeff_list.append(abs(curr_time[-1]-curr_time[0]))
#         coeff = np.array(coeff_list)
#         print(coeff)
        #Adding new for scalalble and yeah
        cc_curr = cc.split("-")[0]
        # Adding another check for scalable and yeah
#         if cc_curr in ['scalable','yeah']:
#             # Look at the fifth coefficient
#             if round(coeff[0],6) < 0.000015:
#                 print(coeff[0])
#                 continue
#         coeff.append(abs(curr_time[-1]-curr_time[0]))
        mp[name] = {'d':degree, 'coeff':coeff, 'error':error_item, 'data':curr_data, 'time':curr_time}
        if cc not in cc_mp :
            cc_mp[cc] = []
        cc_mp[cc].append(mp[name])
    return cc_mp

from scipy.stats import multivariate_normal as mvn
def getPDensity(curr, cc_gaussian_params):
    prob = {}
    for cc in cc_gaussian_params:
        mn = cc_gaussian_params[cc]['mean']
        covar = cc_gaussian_params[cc]['covar']
#         print("CC being checked against",cc)
#         print(np.linalg.det(covar))
        curr_p = mvn.pdf(curr,mean=mn, cov=covar, allow_singular=True)
        prob[cc]=curr_p
    return prob

def getBestDegree(nmp,p="y"):
    results = {}
    for name in nmp.keys():
        print(name)
    #     curr_time, curr_data, retrans, rtt = plot_one_bt(name+"-0-50-200-2-aws-88-60",'y',1)
        data = nmp[name]['data']
        time = nmp[name]['time']
        max_deg = nmp[name]['max_deg']
        p_net = nmp[name]['p_net']
        mse_l = nmp[name]['mse_l']
    #     plt.plot(time, data,c='k',label='Truth')
    #         plt.plot(time, fit_l)
    #     for d in range(0, max_deg):
    #         plot_label = "degree"+str(d+1)
    #         fit_net = np.polyval(p_net[d],time)
    #         plt.plot(time, fit_net,label="degree" + str(d+1))
    #     plt.legend()
    #     plt.show()

        names = []
        for d in range(0, max_deg):
            names.append("degree "+str(d+1))
        loss = []
        lambd = 0.09
        for i in range(len(mse_l)):
            loss.append((i+1)*sum(p_net[i])*lambd)
        print("Degree",names)
        print("loss", loss)
        print("mse", mse_l)
        if p == "y":
            plt.title("Error with respect to polynomial fit")
            plt.bar(names,mse_l,color='blue',width=0.4,label="mse")
            plt.bar(names,loss,bottom=mse_l,color='maroon',width=0.4,label="reg_loss")
            plt.legend()
            # plt.show()
        errors = []
        # The code for deciding the categories
        for i in range(0,max_deg):
            errors.append(loss[i]+mse_l[i])
        deg = errors.index(min(errors))+1
    #     if deg < 3:
    #         deg = mse_l.index(min(mse_l[0:2]))+1
    #     print(deg)
        current_cc = name.split("-")[0]
        results[current_cc]={
            'deg':deg,
            'coeff':p_net[deg-1],
            'error':errors[deg-1]
        }
    return results

from scipy.stats import multivariate_normal as mvn
def getPDensityR(curr, cc_gaussian_params):
    curr_coeff = curr[:-1]
    prob = {}
    # print("cc_gp", cc_gaussian_params)
    for cc in cc_gaussian_params:
        mn = cc_gaussian_params[cc]['mean']
        covar = cc_gaussian_params[cc]['covar']
        if len(curr_coeff) == len(mn):
#             print(curr_coeff)
#             print(mn)
#             print(covar)
            curr_p = mvn.pdf(curr_coeff,mean=mn, cov=covar, allow_singular=True)
            prob[cc]=curr_p
#         print("CC being checked against",cc)
#         print(np.linalg.det(covar))
            print("check density", curr_p, mn, covar, curr_coeff)
    return prob

def getWebDensity(mp, cc_gaussian_params):
    acc_w = {}
    for web in mp:
        p_dense = getPDensityR(mp[web]['coeff'], cc_gaussian_params)
        print("p_dense", p_dense)
        if web not in acc_w:
            acc_w[web]={}
        ccs = np.array(list(p_dense.keys()))
        vals = np.array(list(p_dense.values()))
        
        p_ind = list(np.argsort(vals))
        p_ind.reverse()
        acc_w[web]['ccs'] = [ccs[i] for i in p_ind]
        acc_w[web]['density'] = np.array([vals[i] for i in p_ind])
        maxx = max(acc_w[web]['density'])
        minn = min(acc_w[web]['density'])
        # print("? ", acc_w, maxx, minn)
        acc_w[web]['relative'] = (acc_w[web]['density']-minn)/(maxx-minn) 
    return acc_w

def predictCC(acc_w):
    pred = {}
    for web in acc_w:
        if acc_w[web]['density'][0] < 0.01:
            pred[web]='new'
        elif acc_w[web]['relative'][1] > 0.01:
            pred[web]='confused'
        else :
            pred[web]=acc_w[web]['ccs'][0]
    return pred, acc_w

def getPredictions(cc_mp,cc_gp,no):
    acc_w = getWebDensity(cc_mp, cc_gp)
    pred, acc_w = predictCC(acc_w)
    predictions = {}
    no_feature = []
    for f in no:
        tp_mp = get_feature_degree_R([f],ss=225,p="y",ft_thresh=1,max_deg=3)
        print(cc_mp)
        if len(cc_mp.keys()) == 0:
            continue
        new_f = f.split("/")[-1]
        w = new_f.split("-")[0]
        # print("???", w)
        if w not in acc_w.keys():
            print("No Feature")
            no_feature.append(w)
            predictions[w]="No feature"
        else:
            print(cc_mp[new_f.split("-")[0]]['coeff'])
    #         time, data, retrans, rtt= plot_one_bt(f,"y",1)
            predictions[w] = {}
            print("Top 5 CCS :", acc_w[w]['ccs'][0:5])
            predictions[w]['ccs'] = acc_w[w]['ccs'][0:5]
            
            print("Density Values:",acc_w[w]['density'][0:5])
            predictions[w]['density'] = acc_w[w]['density'][0:5]
            
            print("Relative Density Values:",acc_w[w]['relative'][0:5])
            predictions[w]['relative'] = acc_w[w]['relative'] = acc_w[w]['relative'][0:5]
            
            print("Prediction:",pred[w])
            predictions[w]['final'] = pred[w]
    return predictions

import sys
import pickle
import os

folder ='nebby_data_pro/nebby_data_pro/nebby_data_hk2jp_basertt_0'
PATH = folder +  "/"

with open('train_gp.pkl', 'rb') as f:
    cc_gp = pickle.load(f)

file_num = 0
correct_num = 0

for file in os.listdir(folder):
# Check whether file is required
    file_split = file.split("-")
    cc = file_split[0]
    
    file_num += 1
    print("---------start checking------------", cc)
    classi = checkBBR([file],p="y")
    if classi[0] == "YES BBR":
        print("This is BBR")
        if cc == 'bbr':
            correct_num += 1
        continue
    elif classi[0] == "MAYBE BBR":
        print("This is maybe BBR")
        continue
    elif "NC" in classi[0]:
        print("Not Classified with error", classi[0])
        continue
    else:
        print("Not BBR, continuing")
    
    if cc == 'bbr':
        continue

    if cc not in ccs:
        file_num -= 1
        continue

    mp = get_feature_degree_R([file],ss=225,p="n",ft_thresh=1,max_deg=3)
    # print("mp: ", mp)
    cc_mp = getBestDegree(mp,p="y")
    # print("cc_mp:", cc_mp)
        
    predictions = getPredictions(cc_mp,cc_gp,[file])
    print("predictions", predictions)
    if cc in predictions:
        print("prediction first:", predictions[cc]['ccs'][0], cc)
        if (predictions[cc]['ccs'][0] == cc):
            correct_num += 1
    else:
        file_num-=1

    print("now state: ", correct_num, file_num)
        
print("file num: ", file_num)
print("correct_num: ", correct_num)
print("accuracy: ", correct_num / file_num)