import os
import re
import pandas as pd
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import torch.nn as nn
from config_module import Config
import random
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

class delayDataset_build(nn.Module):
    def __init__(self, config):
        super(delayDataset_build, self).__init__()
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

        # print(host_df.head())
        # print(seq_df.head())
        
        
        # host_df['count'] = host_df.groupby(['Length', 'DataLength', 'SeqNum', 'Timestamp']).cumcount()
        # seq_df['count'] = seq_df.groupby(['Length', 'DataLength', 'SeqNum', 'Timestamp']).cumcount()

       
        merged_df = pd.merge(host_df, seq_df, on=['Length', 'DataLength', 'SeqNum', 'Timestamp'], suffixes=('_host', '_seq'))
        # print(merged_df.head())
        merged_df['Time_host'] = pd.to_datetime(merged_df['Time_host'], format='%H:%M:%S.%f')
        merged_df['Time_seq'] = pd.to_datetime(merged_df['Time_seq'], format='%H:%M:%S.%f')
        # print(merged_df.head())
        base_time = merged_df['Time_seq'].iloc[0]
        return merged_df, base_time
    
    def forward(self, path,train,train_data_num,rank=None):
        envs=os.listdir(path)
        print(f"Before data loading - Memory allocated: {torch.cuda.memory_allocated()} bytes")
        envs=envs[:2]
        
        alo_id=0
        print(path)
        y=[]
        x_trace=[]
        x_env=[]
        x_len=[]
        max_length=0
        
        alo_dic={'htcp': 0, 'bbr': 1, 'vegas': 2, 'westwood': 3, 'scalable': 4, 'highspeed': 5, 'veno': 6, 'reno': 7, 'yeah': 8, 'illinois': 9, 'bic': 10, 'cubic': 11}
        # alo_dic={'htcp': 0, 'bbr': 1, 'vegas': 2, 'westwood': 3, 'scalable': 4, 'highspeed': 5, 'veno': 6, 
        #          'reno': 7, 'yeah': 8, 'illinois': 9, 'bic': 10, 'cubic': 11, 'pcc': 12, 'Pccloss': 13, 'astraea': 14}
        # alo_dic={'htcp': 0, 'bbr': 1, 'vegas': 2, 'westwood': 3, 'scalable': 4, 'highspeed': 5, 'veno': 6, 
        #          'reno': 7, 'yeah': 8, 'illinois': 9, 'bic': 10, 'cubic': 11, 'dctcp': 12}

        # alo_dic={'htcp': 0, 'bbr': 1, 'vegas': 2, 'westwood': 3, 'scalable': 4, 'highspeed': 5, 'veno': 6, 
        #          'reno': 7, 'yeah': 8, 'illinois': 9, 'bic': 10, 'cubic': 11, 'pccLatency': 12, 'pccLoss': 12, 'astraea': 13}

        for env in envs:
            y_tmp={}     
            x_trace_tmp={}
            x_env_tmp={}
            x_len_tmp={}
            
            alos=os.listdir(os.path.join(path,env))
            
            for alo in alos:
                
                if 'host' in alo:
                    alo_name = re.search(r'_(\w+)_', alo).group(1)
                    if alo_name not in alo_dic:
                        continue
                    if alo_name not in alo_dic.keys():
                        print(f"alo_name: {alo_name} not found in alo_dic, skipping.")
                        continue
                    alo_idx = re.search(r'(\d+)(\..*)$', alo).group(1)
                    if alo_name not in y_tmp.keys():
                        y_tmp[alo_name] = []
                        x_env_tmp[alo_name] = []
                        x_len_tmp[alo_name] = []
                        x_trace_tmp[alo_name] = []
                        
                    label_i=[alo_dic[alo_name]]

                    
                    capture_path = os.path.join(path, env, alo.replace("host", "capture"))  
                    host_path = os.path.join(path, env, alo)   
                    
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

                    
                    if self.max_data_len != -1:
                        data_x_trace = data_x_trace[:min(len(data_x_trace), self.max_data_len)] 
                    
                        #print(f"Max_Data_Len is available: {torch.Tensor(data_x_trace).shape}")
                    
                    y_tmp[alo_name].append(label_i) 
                    x_env_tmp[alo_name].append(env_numbers)
                    
                    x_len_tmp[alo_name].append([len(data_x_trace)])
                    x_trace_tmp[alo_name].append(torch.Tensor(data_x_trace))
    
                    
            
            for alo_name, data_list in x_trace_tmp.items(): 
                #print(f'alo_name: {alo_name}, len of data_list: {len(data_list)}')
                if not data_list:
                    print(f"No data found for {alo_name} at env: {env}, skipping.")
                    continue
                last_column_values = [data[-1][0] for data in data_list]
                #print(f'last_column_values: {last_column_values}')
                sorted_indices = np.argsort(last_column_values)
                
                median_pos = len(last_column_values) // 2
                
                if len(sorted_indices) == 1:  
                    indices_to_add = [sorted_indices[0]]
                else:
                    # if train_data_num % 2 == 0:
                    #     raise ValueError("num must be an odd number")

                    neighbors_count = (train_data_num - 1) // 2
                    #print(f'sorted_indices: {sorted_indices}, median_pos: {median_pos}, neighbors_count: {neighbors_count}')
                    if train:  # train_loader  
                        # indices_to_add = sorted_indices[median_pos - neighbors_count : median_pos + neighbors_count + 1]
                        
                        indices_to_add = sorted_indices[:9]
                        
                        

                    else: # test_loader    
                        # print(len(sorted_indices))
                        # indices_to_add = np.concatenate((sorted_indices[:median_pos - neighbors_count], sorted_indices[median_pos + neighbors_count + 1:]))
                        # print(len(indices_to_add))
                        # indices_to_add = [0, 1,2,3,4]
                        # print(len(indices_to_add))
                        # if len(indices_to_add) != 2:
                        #     indices_to_add = indices_to_add[:2]
                        indices_to_add = sorted_indices[:9]
                        
                        
                        

                        
                #print(f'indices_to_add: {indices_to_add}')    
                # for index in indices_to_add:
                #     data_x_trace_tmp = x_trace_tmp[alo_name][index]
                
              
                # indices_to_add = []
                # for index in sorted_indices:
                #     indices_to_add.append(index)
                #print(f'indices_to_add: {indices_to_add}')
                for index in indices_to_add:
                    data_x_trace_tmp = x_trace_tmp[alo_name][index]
                    
                    scaler = MinMaxScaler()
                    # data_x_trace_tmp[:,1] = torch.tensor(scaler.fit_transform(data_x_trace_tmp), dtype=torch.float32)
                    data_x_trace_tmp = torch.tensor(scaler.fit_transform(data_x_trace_tmp), dtype=torch.float32)
                    
                    y.append(y_tmp[alo_name][index])
                    x_env.append(x_env_tmp[alo_name][index])
                    x_len.append(x_len_tmp[alo_name][index])
                    x_trace.append(data_x_trace_tmp)

                    
                    #print(f'max_length: {max_length}')
                    
                    if len(data_x_trace_tmp) > max_length:
                        max_length = len(data_x_trace_tmp)
        #lengths = [len(x) for x in x_trace]
        #print(f'x_trace:{len(x_trace)}, {lengths}')
        data_trace = pad_sequence(x_trace, batch_first=True, padding_value=0)  # [batch_size, seq_len, 2]
        print(f"x_len shape: {torch.Tensor(x_len).shape}")
        print(f"x_env shape: {torch.Tensor(x_env).shape}")
        print(f"y shape: {torch.Tensor(y).shape}")
        print(f"Padded sequence length (seq_len after padding): {data_trace.shape}") 
        
        #print(f'data_trace:{data_trace.shape}')
        data_trace=torch.reshape(data_trace,[len(y),-1])   # [batch_size, seq_len*2] 
        #print(f'data_trace:{data_trace.shape}')
        input=torch.cat([torch.Tensor(x_len),torch.Tensor(x_env),torch.Tensor(y),data_trace],dim=1)
        loader = torch.utils.data.DataLoader(input, 2*self.batch_size, shuffle=True, drop_last=True)
        print(f'input.shape: {input.shape}, max_length of data: {max_length}')
        
        # DDP
        #loader = torch.utils.data.DataLoader(input, batch_size=self.batch_size, shuffle=False, sampler=torch.utils.data.DistributedSampler(input), drop_last=True)
        
        
        # loader = torch.utils.data.DataLoader(input, self.batch_size, shuffle=True, drop_last=True)
        
        num_times=len(indices_to_add)
        num_envs=len(envs)
        num_alos=len(alo_dic)
       
        # num_alos=2
        
        
        # num_times = len(data_x_trace) // (num_envs * num_alos)
        print(f"num_envs: {num_envs}, num_alos: {num_alos}, num_times: {num_times}")
        print(f"input.shape before reshape: {input.shape}")
        print(f"Input data size: {input.size()}, Total elements: {input.numel()}")

        input=input.reshape([num_envs,num_alos,num_times,input.shape[-1]])   
        input=input.permute(0,2,1,3)
        input=input.reshape([-1,num_alos,input.shape[-1]])
        
        loader_contrast = torch.utils.data.DataLoader(input, 2, shuffle=True, drop_last=True)   
        #print(f'loader: {loader}')
        print(f"After creating data loader - Memory allocated: {torch.cuda.memory_allocated()} bytes")
        #print(f'loader: {loader}')
        return loader, torch.tensor(max_length),loader_contrast
    
    

