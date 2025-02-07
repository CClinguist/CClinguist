

from os.path import join, basename, dirname, exists
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import pprint as pp
import os
import numpy as np
import torchmetrics
import csv
from torch.nn.utils import rnn
import re
from sklearn.preprocessing import MinMaxScaler
import math
import wandb
import numpy.linalg as LA
import matplotlib.pyplot as plt
import shutil
import pandas as pd
from tqdm import tqdm
# from utils import torch_load_cpu
from datetime import datetime, timedelta
from torch.nn.utils.rnn import pad_sequence
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


class Classifier(nn.Module):
    def __init__(self, num_classify, dim_hidden, num_layers, num_heads,dim_input, dropout_rate, margin, weight_verf):
        super(Classifier, self).__init__()
        
        self.num_classify = num_classify
        print("num_classify:", num_classify)
        self.encoder_model = seq_model_transformer(dim_hidden, num_layers, num_heads, dim_input, dropout_rate)
        
        self.fc = nn.Sequential(
            nn.Linear(3*(dim_input*num_heads)+dim_hidden, dim_hidden),#, device='cuda'),  
            nn.ReLU(),
            #nn.BatchNorm1d(dim_hidden, device='cuda'),
            nn.LayerNorm(dim_hidden),#, device='cuda'),
            nn.Linear(dim_hidden, int(dim_hidden/2)),#, device='cuda'),
            nn.ReLU(),
            #nn.BatchNorm1d(int(dim_hidden/2), device='cuda'),
            nn.LayerNorm(int(dim_hidden/2)),#, device='cuda'),
            nn.Linear(int(dim_hidden/2), num_classify),#, device='cuda')
        )
        
        self.margin=nn.Parameter(torch.tensor(margin), requires_grad=True)
        self.weight_verf = weight_verf

    def forward(self, x, length, x_env, max_length):
        self.feature_encoded = self.encoder_model(x, length, x_env, max_length)
        
        self.output_classify = self.fc(self.feature_encoded)
        return self.output_classify
    
    def get_loss(self,label,batch_size):
        
        label_input=label[0:batch_size+self.num_classify]     
        label_constrast=label[batch_size+self.num_classify:]  
        y_label=label_input==label_constrast
        y_label = torch.where(y_label, torch.tensor(1), torch.tensor(-1))
        idx_1=torch.where(y_label==1)
        idx_0=torch.where(y_label==-1)
        f_output=self.output_classify[0:batch_size+self.num_classify,:]
        f_output_constrast=self.output_classify[batch_size+self.num_classify:,:]
        
        # f_input=self.feature_encoded[0:batch_size+self.num_classify,:]
        # f_constrast=self.feature_encoded[batch_size+self.num_classify:,:]
        
        
        loss_dis_out=F.pairwise_distance(f_output[idx_0], f_output_constrast[idx_0], p=2)   
        loss_dis_in=F.pairwise_distance(f_output[idx_1], f_output_constrast[idx_1], p=2)
        # a = torch.pow(loss_dis_in.mean(),2)
        # c = torch.pow(loss_dis_out,2)
        # b = torch.pow(torch.max(torch.tensor(0),(torch.abs(self.margin)-torch.pow(loss_dis_out,2)).mean()),2)
        
        # if  loss_dis_in.size()[0]==0:
        #     loss_verf=0.5*torch.pow(torch.max(torch.tensor(0),(torch.abs(self.margin)-torch.pow(loss_dis_out,2)).mean()),2)
        # else:    
        a=torch.abs(self.margin)-loss_dis_out
        b=torch.max(torch.tensor(0),a)
        # loss_verf=0.5* torch.pow(loss_dis_in.mean(),2)+0.5*torch.pow(torch.max(torch.tensor(0),(torch.abs(self.margin)-loss_dis_out.mean())),2)
        loss_verf=0.5* torch.pow(loss_dis_in.mean(),2)+0.5*torch.pow(torch.max(torch.tensor(0),(torch.abs(self.margin)-torch.pow(loss_dis_out.mean(),2))),2)
            
    
        dis_out=loss_dis_out
        dis_in= loss_dis_in  
        #if test:
        # with open('cluster_results/test_dis_in_1.txt', 'a') as f:
        # 
        #     np.set_printoptions(suppress=True)
        #     np.savetxt(f, dis_in.cpu().detach().numpy(), fmt='%f')
        # f.close()
        # with open('cluster_results/test_dis_out_1.txt', 'a') as f:
        #     np.set_printoptions(suppress=True)
        #     np.savetxt(f, dis_out.cpu().detach().numpy(), fmt='%f')
        # f.close()
        
        criterion = nn.CrossEntropyLoss()
        loss_ident_1 = criterion(f_output,label_input)
        loss_ident_2 = criterion(f_output_constrast,label_constrast)
            
        loss = (loss_ident_1 + loss_ident_2+ self.weight_verf * loss_verf)

        return loss,self.margin
    
class seq_model_transformer(nn.Module):
    def __init__(self, dim_hidden, num_layers, num_heads,dim_input, dropout_rate):
        super().__init__()
        self.num_heads = num_heads   
        self.dim_input = dim_input
        self.dim_hidden = dim_hidden
        self.num_layers = num_layers
        self.dropout_rate = dropout_rate
        self.input_encoder = nn.Linear(self.dim_input, self.num_heads*self.dim_input)#, device='cuda')
        self.transformer_layer = nn.TransformerEncoderLayer(d_model=self.dim_input*self.num_heads, nhead=self.num_heads, dim_feedforward=self.dim_hidden, dropout=self.dropout_rate ,batch_first=True)#, device='cuda')
        self.transformer_encoder = nn.TransformerEncoder(self.transformer_layer, num_layers=self.num_layers)
        
        self.env_linear = nn.Linear(2, self.dim_hidden)#, device='cuda')  
    
    def forward(self, data_trace, data_length, x_env, max_length):   
        device = data_trace.device
        
        batch_size = len(data_length)
        # print("data_trace device:", data_trace.device)
        # print("data_length device:", data_length.device)
        # print("x_env device:", x_env.device)
        # if isinstance(max_length, torch.Tensor):
        #     print("max_length device:", max_length.device)
        # else:
        #     print("max_length is not a tensor")
        
        
        env_numbers = x_env#.cuda()  
        input = data_trace  
        
       
        input = self.input_encoder(input)#.cuda())
        #print("input device:", input.device)
        
        env_transformed = self.env_linear(env_numbers)
        
        
       
        positional_encoding = PositionalEncoding(d_model=self.dim_input*self.num_heads, max_length=max_length)   # max_len是单元素tensor
        positional_encoding.to(device)
       
        src_key_padding_mask = torch.arange(max_length.item(), device=device).expand(batch_size, -1) >= data_length.clone().detach().to(device).unsqueeze(1)
        #src_key_padding_mask = torch.arange(max_length.item(), device=device).expand(batch_size, -1) >= data_length.clone().detach().unsqueeze(1)
        #src_key_padding_mask = torch.arange(max_length.item()).expand(batch_size, -1) >= torch.tensor(data_length).unsqueeze(1)
        src = positional_encoding(input)#.cuda()
        
        
        output = self.transformer_encoder(src=src, src_key_padding_mask=src_key_padding_mask)#.cuda())
        
        
        seq_feature = torch.cat([output[:, -1, :], torch.mean(output, dim=1), torch.max(output, dim=1)[0]], dim=-1)
        
        
        feature = torch.cat((seq_feature, env_transformed), dim=-1)
        
        return feature.to(torch.float32)


class PositionalEncoding(nn.Module):
    "Implement the PE function."

    def __init__(self, d_model, max_length):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_length.item(), d_model)   
        position = torch.arange(0, max_length.item()).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)#.cuda()
       
        self.register_buffer("pe", pe)

    def forward(self, x):
        
        x = x + self.pe[:, : x.size(1)].requires_grad_(False)
        return x
    