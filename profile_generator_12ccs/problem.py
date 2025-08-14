from time import sleep
from torch.utils.data import Dataset
import torch
import os
import pickle
import numpy as np
import warnings
from sklearn.preprocessing import StandardScaler
from torch.nn.utils.rnn import pad_sequence
import re
import csv
from torch.nn.utils import rnn
import torch.nn as nn
warnings.filterwarnings('ignore')
from dtw_cal import compute_dtw
import pandas as pd
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler


PADDED = 100000

data_batch_id = '6'

def save_dtw_as_readable_txt(dtw_array, target_alo_list, save_path):

    import pandas as pd


    if len(dtw_array.shape) == 2:
        dtw_array = dtw_array[np.newaxis, ...] 

    if len(dtw_array.shape) != 3:
        print(f"[Warning] Unexpected dtw_array shape: {dtw_array.shape}, skip saving.")
        return

    num_env, M1, M2 = dtw_array.shape
    if M1 != M2 or M1 != len(target_alo_list):
        print(f"[Warning] DTW shape={dtw_array.shape}")
    
    df_rows = []
    for env_idx in range(num_env):

        env_name = env_idx
        for alo_1_idx, alo_1_name in enumerate(target_alo_list):
            for alo_2_idx, alo_2_name in enumerate(target_alo_list):
                dtw_val = dtw_array[env_idx, alo_1_idx, alo_2_idx]
                df_rows.append([env_name, alo_1_name, alo_2_name, dtw_val])

    df = pd.DataFrame(df_rows, columns=["env", "alo_1", "alo_2", "dtw"])

    df.to_csv(save_path, sep="\t", header=False, index=False)
    print(f"[Info] dtw_array (shape={dtw_array.shape}) saved to: {save_path} in tabular format.")

class Dataset(object):

    def __init__(self, batch_size,data_path,dtw_path=None,indices_to_add = [0]):
        super(Dataset, self).__init__()
        self.data_path=data_path
        self.dtw_path=dtw_path
        self.batch_size=batch_size
        self.data_position='host'

        self.max_data_len = 512      
        self.indices_to_add = indices_to_add
    
    def convert_to_seconds(self,time_str):
        time_format = "%H:%M:%S.%f"
        dt = datetime.strptime(time_str, time_format)
        total_seconds = dt.hour * 3600 + dt.minute * 60 + dt.second + dt.microsecond / 1e6
        return total_seconds
    
    def make_dataset_1209(self, batch_size, train=True, train_data_num=1):

        
        self.train_data_num = train_data_num   
        print("train_data_num: ",train_data_num)
        
        
        files = os.listdir(self.data_path)

        envs = os.listdir(self.data_path)
        
        for file in files:
            parts = file.split('_')
            rtt_part = parts[1]
            rtt_value_str = ''.join(filter(str.isdigit, rtt_part))  
            rtt_value = int(rtt_value_str)
                         
        self.env_ids=np.arange(1,len(files)+1)
        
        env_id = 0

        self.y=[]
        self.x_trace=[]
        x_len=[]
        x_env=[]
        self.x_env_id=[]
        x_index=[]
        max_length=0
       
        alo_dic={'htcp': 0, 'bbr': 1,  'vegas': 2, 'westwood': 3, 'scalable': 4, 'highspeed': 5, 
                 'veno': 6, 'reno':7, 'yeah': 8, 'illinois': 9 ,'bic': 10, 'cubic': 11}

        for env in envs:   
            y_tmp={}    
            x_trace_tmp={}
            x_env_tmp={}
            x_len_tmp={}
            max_length_env=0

            env_id+=1

            alos=os.listdir(os.path.join(self.data_path,env))
            
            index_alo=-1    
            print(f"env: {env} =================================================================")
            for alo in alos:

                numbers = re.findall(r'\d+', alo)[0]

                
                if 'host' in alo:
                    alo_name = re.search(r'_(\w+)_', alo).group(1)
                    alo_idx = re.search(r'(\d+)(\..*)$', alo).group(1) 
                    if alo_name not in alo_dic.keys():
                        # print(f"alo_name: {alo_name} not found in alo_dic, skipping.")
                        continue
                    if alo_name not in y_tmp.keys():
                        y_tmp[alo_name] = []
                        x_env_tmp[alo_name] = []
                        x_len_tmp[alo_name] = []
                        x_trace_tmp[alo_name] = []
                        
                    label_i=[alo_dic[alo_name]]

                    
                    capture_path = os.path.join(self.data_path, env, alo.replace("host", "capture"))  
                    host_path = os.path.join(self.data_path, env, alo)   
                    
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

                    
                    y_tmp[alo_name].append(label_i) 
                    x_env_tmp[alo_name].append(env_numbers)
                    
                    x_len_tmp[alo_name].append([len(data_x_trace)])
                    x_trace_tmp[alo_name].append(torch.Tensor(data_x_trace))
            

            x_trace_tmp = dict(sorted(x_trace_tmp.items()))
            for alo_name, data_list in x_trace_tmp.items(): 

                if not data_list:
                    # print(f"No data found for {alo_name} at env: {env}, skipping.")
                    continue
                last_column_values = [data[-1][0] for data in data_list]
     
                sorted_indices = np.argsort(last_column_values)
                
                median_pos = len(last_column_values) // 2
                
                if len(sorted_indices) == 1: 
                    indices_to_add = [sorted_indices[0]]
                else:
                    if train_data_num % 2 == 0:
                        raise ValueError("num must be an odd number")

                    neighbors_count = (train_data_num - 1) // 2

                    if train:  
                        indices_to_add = sorted_indices[median_pos - neighbors_count : median_pos + neighbors_count + 1]
                        

                    else: 
                        indices_to_add = np.concatenate((sorted_indices[:median_pos - neighbors_count], sorted_indices[median_pos + neighbors_count + 1:9]))

                        indices_to_add = np.array(self.indices_to_add)

                        
                        
                    
                self.input_data_num=len(indices_to_add)
                for index in indices_to_add:
                    print(alo_name)
                    data_x_trace_tmp = x_trace_tmp[alo_name][index]
                    index_alo+=1
                    scaler = MinMaxScaler()
                    data_x_trace_tmp = torch.tensor(scaler.fit_transform(data_x_trace_tmp), dtype=torch.float32)
                    
                    self.y.append(y_tmp[alo_name][index])
                    x_env.append(x_env_tmp[alo_name][index])
                    x_len.append(x_len_tmp[alo_name][index])
                    self.x_trace.append(data_x_trace_tmp)
                    self.x_env_id.append([int(env_id)])
                    x_index.append([index_alo])
                   

                    if len(data_x_trace_tmp) > max_length:
                        max_length = len(data_x_trace_tmp)
                               
        self.num_alos=len(alo_dic)
        data_trace = rnn.pad_sequence(self.x_trace, batch_first=True, padding_value=0)
        data_trace=torch.reshape(data_trace,[len(self.y),-1])
        
        self.input=torch.cat([torch.Tensor(x_len),torch.Tensor(self.x_env_id),torch.Tensor(x_env),torch.Tensor(self.y),torch.Tensor(x_index),data_trace],dim=1)

        loader = torch.utils.data.DataLoader(self.input, batch_size, shuffle=True, drop_last=True)
        
        max_length = torch.tensor(max_length)
        
        return loader,max_length
    
    def get_merged_df(self, host_path, capture_path):

        host_df = pd.read_csv(host_path, sep=',', quotechar='"', 
                            names=['_ws.col.Time', 'frame.len', 'ip.src', 'tcp.srcport', 'ip.dst', 'tcp.dstport', 'tcp.len', 'tcp.seq_raw', 'tcp.ack_raw', 'tcp.options.timestamp.tsval', 'tcp.options.timestamp.tsecr'],
                            skiprows=1)  
        

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
        
        merged_df['Time_host'] = pd.to_datetime(merged_df['Time_host'], format='%H:%M:%S.%f')
        merged_df['Time_seq'] = pd.to_datetime(merged_df['Time_seq'], format='%H:%M:%S.%f')
        
        base_time = merged_df['Time_seq'].iloc[0]
        return merged_df, base_time
    
    
    def custom_fillna(self,x):
        
        for col in x.index:
            for row in x.columns:
                if pd.isna(x.loc[col, row]) and not pd.isna(x.loc[row, col]):
                    x.loc[col, row] = x.loc[row, col]
        return x
        
    def load_action_dtw(self):
        
        dtw_result=None
 
        target_alo_list=['htcp','bbr', 'vegas', 'westwood', 'scalable', 'highspeed', 'veno', 'reno', 'yeah', 'illinois', 'bic', 'cubic']
        if self.dtw_path is None:
            
            dtw_result=compute_dtw([self.x_trace,self.y,self.x_env_id],data_path=None)
            

        else:
            print("dtw_path: ",self.dtw_path)
            
            data = pd.read_csv(self.dtw_path, sep="\t", header=None, names=["env", "alo_1", "alo_2", "dtw"])

            data["dtw"] = (data['dtw'].values.reshape(-1, 1))
            envs = data["env"].unique()
            
            for env in envs:
                
                env_data = data[data["env"] == env]

                swapped_env_data = env_data[["env", "alo_2", "alo_1", "dtw"]].copy()
                swapped_env_data.columns = ["env", "alo_1", "alo_2", "dtw"]
                env_data = pd.concat([env_data, swapped_env_data], ignore_index=True)
                

                heatmap_data = pd.pivot_table(env_data, values="dtw", index="alo_1", columns="alo_2", aggfunc=np.mean)
                
                heatmap_data = self.custom_fillna(heatmap_data)  
                matrix_dtw = heatmap_data.fillna(0)  
                matrix_dtw=matrix_dtw.values
                if matrix_dtw.shape[0]==matrix_dtw.shape[1] and matrix_dtw.shape[0]==len(target_alo_list):
                    if dtw_result is not None:
                        dtw_result=np.concatenate([dtw_result,np.expand_dims(matrix_dtw,0)],axis=0)
                    else:
                        dtw_result=np.expand_dims(matrix_dtw,0)
        
        return dtw_result
    
    def get_trace(self,action,alo):

        envs=torch.Tensor(self.env_ids).unsqueeze(0).repeat(len(action),1)     
        
        matches=(envs==action.unsqueeze(-1).repeat(1,envs.shape[1]).detach().cpu()).int()  

        index=matches.argmax(dim=1)  
        index=index*self.num_alos*self.input_data_num
        index=index.int()+alo.int()
        data=self.input[index.tolist(),:]
        return data
