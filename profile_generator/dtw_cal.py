from dtw import *
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.interpolate import interp1d
import math
import re

from sklearn.preprocessing import MinMaxScaler

current_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(current_dir, 'data', 'train_data')

def load_data_arrivetime(data):
    trace=data[0]
    label=data[1]
    env=data[2]
    dic_envs={}
    dic_alos={}
    envs=[]
    alos=[]
    for i in range(len(trace)):
        envname=env[i][0]
        if envname not in dic_envs:
            envs.append(envname)
            dic_envs[envname]=[]
        dic_envs[envname].append(trace[i])
        if label[i][0] not in dic_alos:
            alos.append(label[i][0])
            dic_alos[label[i][0]]=[]
        dic_alos[label[i][0]].append(trace[i])
    return dic_envs,dic_alos,envs,label
    
    
    

def load_data(path):
    dic_envs={}
    dic_alos={}
    envs = os.listdir(data_path)

    for env in envs:
        alo_path=os.path.join(data_path,env)
        alos=os.listdir(alo_path)

        for alo in alos:
            trace=np.loadtxt(os.path.join(alo_path,alo))
            if trace.shape[0]>10000:
                trace=trace[:10000,:]
            if env not in dic_envs:
                dic_envs[env]=[]
            dic_envs[env].append(trace)  
            if alo not in dic_alos:
                dic_alos[alo]=[]
            dic_alos[alo].append(trace)  
    return dic_envs,dic_alos,envs,alos  
            
def dtw_similarity(data1,data2):

    alignment = dtw(data1,data2,dist_method='euclidean',keep_internals = True)

    dis=alignment.distance/(max(len(data1),len(data2)))
    return dis

def resample(data1, data2, alpha=0.75):
    # Re-sample the data using linear interpolation with a sampling interval of 5 ms
    time1 = data1[:,0]
    time2 = data2[:,0]
    y1=data1[:,1]
    y2=data2[:,1]
    stride=min(np.min(time1[1:] - time1[:-1]),np.min(time2[1:] - time2[:-1]))
    stride = round(stride, 3)
    new_time1 = np.arange(0, np.max(time1), stride)    
    new_time2 = np.arange(0, np.max(time2), stride)
    new_y_1 = np.interp(new_time1, time1, y1)
    new_y_2 = np.interp(new_time2, time2, y2)
    return new_y_1, new_y_2
        

def compute_dtw(data,data_path=None):
    
    if data_path is not None:
        dic_envs,dic_alos,envs,alos=load_data(data_path)
    else:
        dic_envs,dic_alos,envs,alos=load_data_arrivetime(data)
        
    dtw_path = os.path.join(current_dir, 'dtw_matrix.txt')
    f = open(dtw_path,'w')
    row_data=[]
    num_alos=len(dic_alos)
    num_envs=len(dic_envs)
    
    matrix_dtw=np.zeros((num_envs,num_alos,num_alos))        
    for env in dic_envs:    
        id_env=list(dic_envs.keys()).index(env)
        for i in range(len(dic_alos)):
            for j in range(i,len(dic_alos)):
                
                data1=dic_envs[env][i]
                data2=dic_envs[env][j]


                dis=dtw_similarity(data1[:,1],data2[:,1])
                matrix_dtw[id_env,i,j]=dis
                matrix_dtw[id_env,j,i]=dis

                f.write(f"{env}\t{i}\t{j}\t{dis}\n")  
    # f.close()            
    return matrix_dtw
    
                
                
                
if __name__ == '__main__':
    compute_dtw(data_path)
    
            
