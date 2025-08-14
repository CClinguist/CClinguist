import os
import re
import pandas as pd
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import torch.nn as nn
import random
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

class Dataset_build(nn.Module):
    def __init__(self, config):
        super(Dataset_build, self).__init__()
        self.batch_size=config.batch_size
        self.data_position=config.data_position
        self.max_data_len = config.max_len

    def get_merged_df(self, host_path, capture_path):
    
        host_df = pd.read_csv(host_path, sep=',', quotechar='"', 
                            names=['_ws.col.Time', 'frame.len', 'ip.src', 'tcp.srcport', 'ip.dst', 'tcp.dstport', 'tcp.len', 'tcp.seq_raw', 'tcp.ack_raw', 'tcp.options.timestamp.tsval', 'tcp.options.timestamp.tsecr'],
                            skiprows=1) 
        
        # print("Host DataFrame (before filtering):\n", host_df.head())
        # print("Data types:\n", host_df.dtypes)
        
   
        if host_df.isna().any().any():
            host_df = host_df.dropna()

    
        host_df = host_df.rename(columns={'_ws.col.Time': 'Time', 'frame.len': 'Length', 'tcp.len': 'DataLength', 'tcp.seq_raw': 'SeqNum', 'tcp.options.timestamp.tsval': 'Timestamp'})
        host_df[['Length', 'DataLength', 'SeqNum', 'Timestamp']] = host_df[['Length', 'DataLength', 'SeqNum', 'Timestamp']].astype(int)
        selected_columns = ['Time', 'Length', 'DataLength', 'SeqNum', 'Timestamp']
        host_df = host_df[selected_columns]
       
        if (host_df['DataLength'] > 1000).any():
            first_data_pkt_idx = host_df[host_df['DataLength'] > 1000].index[0]
        else:
            print(f"No data packets with DataLength > 1000 found in {host_path}.")
            return pd.DataFrame(), None
        
        host_df = host_df.iloc[first_data_pkt_idx:].reset_index(drop=True)
        target_length, target_datalength, target_seqnum, target_timestamp = host_df.loc[0, ['Length', 'DataLength', 'SeqNum', 'Timestamp']]
        
        
        capture_df = pd.read_csv(capture_path, sep=',', quotechar='"', 
                                names=['_ws.col.Time', 'frame.len', 'ip.src', 'tcp.srcport', 'ip.dst', 'tcp.dstport', 'tcp.len', 'tcp.seq_raw', 'tcp.ack_raw', 'tcp.options.timestamp.tsval', 'tcp.options.timestamp.tsecr'],
                                skiprows=1) 
        
        capture_df = capture_df.rename(columns={'_ws.col.Time': 'Time', 'frame.len': 'Length', 'tcp.len': 'DataLength', 'tcp.seq_raw': 'SeqNum', 'tcp.options.timestamp.tsval': 'Timestamp'})
        seq_df = capture_df.reset_index(drop=True)
        try:
            seq_df = seq_df.iloc[:-1]    
            seq_df[['Length', 'DataLength', 'SeqNum', 'Timestamp']] = seq_df[['Length', 'DataLength', 'SeqNum', 'Timestamp']].astype(int)
        except ValueError as e:
            na_exists = seq_df[['Length', 'DataLength', 'SeqNum', 'Timestamp']].isna().any().any()
            na_indices = seq_df[seq_df.any(axis=1)]
            last_row = seq_df.tail(1)
            print(last_row)
            print(f"ValueError: {e}")
            print(seq_df.head())
        seq_df = seq_df[selected_columns]

        
        matching_rows = seq_df[(seq_df['Length'] == target_length) & 
                            (seq_df['DataLength'] == target_datalength) & 
                            (seq_df['SeqNum'] == target_seqnum) & 
                            (seq_df['Timestamp'] == target_timestamp)]
        
        if matching_rows.empty:
            print("No matching rows found in capture file.")
            return pd.DataFrame(), None

        matching_idx = matching_rows.index[0]
        seq_df = seq_df.iloc[matching_idx:].reset_index(drop=True)

       
        merged_df = pd.merge(host_df, seq_df, on=['Length', 'DataLength', 'SeqNum', 'Timestamp'], suffixes=('_host', '_seq'))
        # print(merged_df.head())
        merged_df['Time_host'] = pd.to_datetime(merged_df['Time_host'], format='%H:%M:%S.%f')
        merged_df['Time_seq'] = pd.to_datetime(merged_df['Time_seq'], format='%H:%M:%S.%f')
        # print(merged_df.head())
        base_time = merged_df['Time_seq'].iloc[0]
        return merged_df, base_time
    
 
        
    
    def forward(self, path, profile = None, target_cca='pcc', max_data_len=250):
        
       
        envs=os.listdir(path)
        
        alo_id=0
        print(path)
        y=[]
        x_trace=[]
        x_env=[]
        x_len=[]
        max_length=0
        
        # alo_dic={'htcp': 0, 'bbr': 1, 'vegas': 2, 'westwood': 3, 'scalable': 4, 'highspeed': 5, 'veno': 6, 'reno': 7, 'yeah': 8, 'illinois': 9, 'bic': 10, 'cubic': 11}
        # alo_dic={'htcp': 0, 'bbr': 1, 'vegas': 2, 'westwood': 3, 'scalable': 4, 'highspeed': 5, 'veno': 6, 
        #          'reno': 7, 'yeah': 8, 'illinois': 9, 'bic': 10, 'cubic': 11, 'pcc': 12, 'Pccloss': 13, 'astraea': 14}
        # alo_dic={'htcp': 0, 'bbr': 1, 'vegas': 2, 'westwood': 3, 'scalable': 4, 'highspeed': 5, 'veno': 6, 
        #          'reno': 7, 'yeah': 8, 'illinois': 9, 'bic': 10, 'cubic': 11, 'dctcp': 12}

        alo_dic={'htcp': 0, 'bbr': 1, 'vegas': 2, 'westwood': 3, 'scalable': 4, 'highspeed': 5, 'veno': 6, 
                 'reno': 7, 'yeah': 8, 'illinois': 9, 'bic': 10, 'cubic': 11, 'pccLatency': 12, 'astraea': 13, 'pccLoss': 14,  'pccvariant1': 15, 'pccvariant2': 16}

        target_ccas = ['pccLoss', 'pccLatency']
        for env in envs:
            y_tmp={}    
            x_trace_tmp={}
            x_env_tmp={}
            x_len_tmp={}
        
            if profile != None:
                if profile not in env:
                    # print(f"Env {env} does not match the profile {profile}, skipping.")
                    continue
                
            alos=os.listdir(os.path.join(path,env))
            if len(alos) < 30*9:     
                print(f"Env {env} has less than 15*9 data, skipping.")
            
            for alo in alos:
                # 
                if alo.endswith('.pcap'):
                    continue
                if 'host' in alo:
                    alo_name = re.search(r'_(\w+)_', alo).group(1)
                    id = int(re.findall(r'\d+', alo)[0])
                    # if id not in [5]:
                    #     continue
                    # print(alo_name)
                    if alo_name == 'pcc':
                        alo_name = 'pccLoss'
                    if target_cca == 'pcc':
                        target_ccas = ['pccLoss', 'pccLatency']
                    else:
                        target_ccas = [target_cca]
                            
                    
                    # pcc 
                    if alo_name not in target_ccas:
                        continue
                   
                    # if alo_name!= target_cca:
                    #     continue
                    print(alo)
                    alo_idx = re.search(r'(\d+)(\..*)$', alo).group(1) # 后缀
                    if alo_name not in y_tmp.keys():
                        y_tmp[alo_name] = []
                        x_env_tmp[alo_name] = []
                        x_len_tmp[alo_name] = []
                        x_trace_tmp[alo_name] = []
                        
                    label_i=[alo_dic[alo_name]]

                    
                    capture_path = os.path.join(path, env, alo.replace("host", "capture"))  #获取capture_file_path
                    host_path = os.path.join(path, env, alo)    #获取host_file_path
                    print(f"Processing host file: {host_path}, capture file: {capture_path}")
                    
                    merged_df, base_time = self.get_merged_df(host_path, capture_path)
                    if merged_df.empty:
                        continue
                    
                   
                    numbers = re.findall(r'\d+', env)
                    env_numbers = list(map(int, numbers[:2])) 
                    
                    
                    merged_df['Time_y'] = merged_df['Time_host'] - merged_df['Time_seq']
                    merged_df['Time_y'] = merged_df['Time_y'].apply(
                        lambda x: x if x >= pd.Timedelta(0) else x + pd.Timedelta(days=1)
                    )
                    merged_df['Time_y'] = merged_df['Time_y'].apply(lambda x: x.total_seconds())
                    
                  
                    merged_df['Time_x'] = merged_df['Time_host'] - base_time
                    merged_df['Time_x'] = merged_df['Time_x'].apply(
                        lambda x: x if x >= pd.Timedelta(0) else x + pd.Timedelta(days=1)
                    )
                    merged_df['Time_x'] = merged_df['Time_x'].apply(lambda x: x.total_seconds())
                    
                    
                    data_x_trace = [[row['Time_x'], row['Time_y']] for index, row in merged_df.iterrows()]
                    
                  
                    data_x_trace = sorted(data_x_trace, key=lambda x: x[0])
                   
                    data_x_trace = [item for item in data_x_trace if item[0] <= 10]

                    
                    if max_data_len != -1:
                        data_x_trace = data_x_trace[:min(len(data_x_trace), max_data_len)]  
                    
                    y_tmp[alo_name].append(label_i) 
                    x_env_tmp[alo_name].append(env_numbers)
                    if len(data_x_trace) == 130:
                        a=0
                    x_len_tmp[alo_name].append([len(data_x_trace)])
                    x_trace_tmp[alo_name].append(torch.Tensor(data_x_trace))
    
                    
            
            for alo_name in x_trace_tmp.keys():
                for index in range(len(x_trace_tmp[alo_name])):
                    data_x_trace_tmp = x_trace_tmp[alo_name][index]
                    
                    scaler = MinMaxScaler()
                    # data_x_trace_tmp[:,1] = torch.tensor(scaler.fit_transform(data_x_trace_tmp), dtype=torch.float32)
                    data_x_trace_tmp = torch.tensor(scaler.fit_transform(data_x_trace_tmp), dtype=torch.float32)
                    
                    y.append(y_tmp[alo_name][index])
                    x_env.append(x_env_tmp[alo_name][index])
                    x_len.append(x_len_tmp[alo_name][index])
                    x_trace.append(data_x_trace_tmp)

                    if len(data_x_trace_tmp) > max_length:
                        max_length = len(data_x_trace_tmp)
                        if max_length == 130:
                            s=0
        
        data_trace = pad_sequence(x_trace, batch_first=True, padding_value=0)  # [batch_size, seq_len, 2]
        print(f"x_len shape: {torch.Tensor(x_len).shape}")
        print(f"x_env shape: {torch.Tensor(x_env).shape}")
        print(f"y shape: {torch.Tensor(y).shape}")
        print(f"Padded sequence length (seq_len after padding): {data_trace.shape}") 
        
        data_trace=torch.reshape(data_trace,[len(y),-1])   # [batch_size, seq_len*2]  

        input=torch.cat([torch.Tensor(x_len),torch.Tensor(x_env),torch.Tensor(y),data_trace],dim=1)
        
        
        return input, torch.tensor(max_length)
    

