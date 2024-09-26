from torch.utils.data import Dataset
import torch
import os
import pickle
import numpy as np
import warnings
from sklearn.preprocessing import StandardScaler
import re
import csv
from torch.nn.utils import rnn
warnings.filterwarnings('ignore')
from dtw_cal import compute_dtw
import pandas as pd
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler

#torch.cuda.set_device(1)
PADDED = 100000

class Dataset(object):

    def __init__(self, batch_size,data_path,dtw_path=None):
        super(Dataset, self).__init__()
        self.data_path=data_path
        self.dtw_path=dtw_path
        self.batch_size=batch_size
        self.data_position='host'

    
    def convert_to_seconds(self,time_str):
        time_format = "%H:%M:%S.%f"
        dt = datetime.strptime(time_str, time_format)
        total_seconds = dt.hour * 3600 + dt.minute * 60 + dt.second + dt.microsecond / 1e6
        return total_seconds
    

    def make_dataset_arrivetime(self,batch_size,train=True, train_data_num=5):
        
        self.train_data_num = train_data_num  
        envs=os.listdir(self.data_path)
        self.env_ids=np.arange(1,len(envs)+1)
        
        
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
            for alo in alos:
                
                alo_name = re.search(r'_(\w+)_', alo).group(1)
                if alo_name=='dctcp':
                    continue

                #alo_name= re.search(r'([^_]+)_', alo).group(1)
                if alo_name not in y_tmp.keys():
                    y_tmp[alo_name] = []
                    x_env_tmp[alo_name] = []
                    x_len_tmp[alo_name] = []
                    x_trace_tmp[alo_name] = []
                    
                label_i=[alo_dic[alo_name]]

                if self.data_position=='host' and (self.data_position in alo):
                    
                    str_index = alo.find("host")
                    capture_filename = alo[:str_index] + "capture" + alo[str_index + len("host"):]

                    f=open(os.path.join(self.data_path,env)+f'/{capture_filename}','r')
                    lines=f.readlines()
                    f.close()
                    data_capture=[]
                    previous_seconds = None
                    day_in_seconds = 24 * 3600
                    for line in lines:
                        line=line.strip()
                        line=line.split(' ')
                        if line[0]=='seq':
                            total_seconds = self.convert_to_seconds(line[1])
                            if previous_seconds is not None and total_seconds < previous_seconds:
                                total_seconds += day_in_seconds 
                            previous_seconds = total_seconds
                            data_capture.append([total_seconds,int(line[-1])])
                    data_capture = np.array(data_capture)
                    
                    if len(data_capture) == 0:
                        continue
                    
                    f=open(os.path.join(self.data_path,env)+'/'+alo,'r')  #host
                    lines=f.readlines()
                    lines = lines[2:]
                    f.close()
                    previous_seconds = None

                    packet_history_dic = {}
                    data_x_trace = []
                    for line in lines:
                        line=line.strip()
                        line=line.split(' ')
                        packet_seq=int(line[-1])
                        if packet_seq in data_capture[:,-1]:
                            if packet_seq not in packet_history_dic:
                                packet_history_dic[packet_seq] = 0
                            index_array = np.where(data_capture[:,-1]==packet_seq)[0]   
                            index=index_array[min(packet_history_dic[packet_seq], len(index_array)-1)] 
                            time_packet_capture=data_capture[index][0]
                            total_seconds = self.convert_to_seconds(line[0])
                            if previous_seconds is not None and total_seconds < previous_seconds:
                                total_seconds += day_in_seconds  
                            previous_seconds = total_seconds
                            delay_packet = total_seconds - time_packet_capture
                            if delay_packet < 0:
                                continue
                            packet_history_dic[packet_seq] += 1 
                            data_x_trace.append([time_packet_capture, delay_packet])

                    data_x_trace = np.array(data_x_trace)
                    data_x_trace = data_x_trace[np.argsort(data_x_trace[:, 0])]
                    first_value = data_x_trace[0,0]
                    data_x_trace[:, 0] = data_x_trace[:, 0] - first_value
                    numbers = re.findall(r'\d+', env)
                    env_numbers = list(map(int, numbers[:2])) 
                    y_tmp[alo_name].append(label_i) 
                    x_env_tmp[alo_name].append(env_numbers)
                    x_len_tmp[alo_name].append([len(data_x_trace)])
                    x_trace_tmp[alo_name].append(torch.Tensor(data_x_trace))

            for alo_name, data_list in x_trace_tmp.items(): 
                last_column_values = [data[-1][0] for data in data_list]
                sorted_indices = np.argsort(last_column_values)
                median_pos = len(last_column_values) // 2
                median_index = sorted_indices[median_pos]

                neighbors_count = (train_data_num - 1) // 2
                if train: 

                    indices_to_add = sorted_indices[median_pos - neighbors_count : median_pos + neighbors_count + 1]
                    self.train_data_num=len(indices_to_add)

                else: 
                    
                    indices_to_add = np.concatenate((sorted_indices[:median_pos - neighbors_count-1], sorted_indices[median_pos + neighbors_count+1:]))
                    self.train_data_num=len(indices_to_add)
                    
                
                for index in indices_to_add:
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

        loader = torch.utils.data.DataLoader(self.input, batch_size, shuffle=False, drop_last=True)
        
        return loader,max_length
    
        
    
    def custom_fillna(self,x):
        for col in x.index:
            for row in x.columns:
                if pd.isna(x.loc[col, row]) and not pd.isna(x.loc[row, col]):
                    x.loc[col, row] = x.loc[row, col]
        return x
        
    def load_action_dtw(self):
        dtw_result=None
        target_alo_list=['htcp','bbr', 'dctcp', 'vegas', 'westwood', 'scalable', 'highspeed', 'veno', 'reno', 'yeah', 'illinois', 'bic', 'cubic']
        if self.dtw_path is None:
            
            dtw_result=compute_dtw([self.x_trace,self.y,self.x_env_id],data_path=None)
        
        else:
            # matrix_dtw=self.loadtxt(self.dtw_path)
            data = pd.read_csv(self.dtw_path, sep="\t", header=None, names=["env", "alo_1", "alo_2", "dtw"])
            # scaler = StandardScaler()
            # data["dtw"] = scaler.fit_transform(data["dtw"].values.reshape(-1, 1)).flatten()   
            # a=max(data["dtw"].values.reshape(-1, 1))
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
        
        envs=torch.Tensor(self.env_ids).unsqueeze(0).repeat(len(action),1)     #[batch,num_envs]
    
        matches=(envs==action.unsqueeze(-1).repeat(1,envs.shape[1]).detach().cpu()).int()  #[batch,num_envs]
        index=matches.argmax(dim=1)  
        index=index*self.num_alos*self.train_data_num  
        index=index.int()+alo.int()
        data=self.input[index.tolist(),:]
        return data
        
        
        
        
        

    
