from os.path import join, basename, dirname, exists
import argparse
import shutil
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

# from utils import torch_load_cpu
from datetime import datetime, timedelta
currentDateAndTime = datetime.now()
TIME=str(currentDateAndTime.month)+'_'+str(currentDateAndTime.day)+'_'+str(currentDateAndTime.hour)+'_'+str(currentDateAndTime.minute)
torch.set_printoptions(sci_mode=False)


parser = argparse.ArgumentParser(description='Classifier_4_CcIdentify_wiz_env&receiveTime_0716"')

parser.add_argument('--epochs', default=10, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--lr', '--learning-rate', default=0.001, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--batch_size', default=12, type=int)
parser.add_argument('--seed', default=3402, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--num-cluster', default=12, type=int)
parser.add_argument('--dim_input', default=2, type=int)
# parser.add_argument('--dim_hidden', default=128, type=int)
parser.add_argument('--dim_hidden', default=128, type=int)
parser.add_argument('--num_layers', default=16, type=int)
parser.add_argument('--num_heads', default=16, type=int)
parser.add_argument('--exp-dir', default='classify_model_arriveTime/'+TIME, type=str,
                    help='experiment directory')
# parser.add_argument('--train_path', default='data/time_seqNum_07_13_3alos', type=str,
#                     help='experiment directory')
parser.add_argument('--train_path', default='data/07_16_dp_80_400KB_stride_20', type=str,
                    help='experiment directory')
parser.add_argument('--test_path', default='data/07_16_dp_80_400KB_stride_20', type=str,
                    help='experiment directory')
parser.add_argument('--train', default=False, type=bool)
parser.add_argument('--load_model_path', default='checkpoint_model_classify/checkpoint_2500.pth.tar', type=str,
                    help='experiment directory')



class Classifier(nn.Module):
    def __init__(self, num_classify, dim_hidden, num_layers, num_heads,dim_input):
        super(Classifier, self).__init__()
        
        self.encoder_model = seq_model_transformer(dim_hidden, num_layers, num_heads,dim_input)
        
        self.fc = nn.Sequential(
            nn.Linear(3*(dim_input*num_heads)+dim_hidden, dim_hidden, device='cuda'), 
            nn.ReLU(),
            #nn.BatchNorm1d(dim_hidden, device='cuda'),
            nn.LayerNorm(dim_hidden, device='cuda'),
            nn.Linear(dim_hidden, int(dim_hidden/2), device='cuda'),
            nn.ReLU(),
            #nn.BatchNorm1d(int(dim_hidden/2), device='cuda'),
            nn.LayerNorm(int(dim_hidden/2), device='cuda'),
            nn.Linear(int(dim_hidden/2), num_classify, device='cuda')
        )

    def forward(self, x, length, x_env, max_length):
        feature_encoded = self.encoder_model(x, length, x_env, max_length)
        
        output_classify = self.fc(feature_encoded)
        return output_classify
    
class seq_model_transformer(nn.Module):
    def __init__(self, dim_hidden, num_layers, num_heads,dim_input):
        super().__init__()
        self.num_heads = num_heads   
        self.dim_input = dim_input
        self.dim_hidden = dim_hidden
        self.num_layers = num_layers
        self.input_encoder = nn.Linear(self.dim_input, self.num_heads*self.dim_input, device='cuda')
        self.transformer_layer = nn.TransformerEncoderLayer(d_model=self.dim_input*self.num_heads, nhead=self.num_heads, dim_feedforward=self.dim_hidden, batch_first=True, device='cuda')
        self.transformer_encoder = nn.TransformerEncoder(self.transformer_layer, num_layers=self.num_layers)
        
        self.env_linear = nn.Linear(2, self.dim_hidden, device='cuda')  
    
    def forward(self, data_trace, data_length, x_env, max_length):
        batch_size = len(data_length)
        
        env_numbers = x_env.cuda() 
        input = data_trace  
        
        input = self.input_encoder(input.cuda())
        
        env_transformed = self.env_linear(env_numbers)
        
        
        positional_encoding = PositionalEncoding(d_model=self.dim_input*self.num_heads, max_len=max_length)
        src_key_padding_mask = torch.arange(max_length).expand(batch_size, -1) >= data_length.clone().detach().unsqueeze(1)
        src = positional_encoding(input).cuda()
        
        output = self.transformer_encoder(src=src, src_key_padding_mask=src_key_padding_mask.cuda())
        
        seq_feature = torch.cat([output[:, -1, :], torch.mean(output, dim=1), torch.max(output, dim=1)[0]], dim=-1)
        
        feature = torch.cat((seq_feature, env_transformed), dim=-1)
        
        return feature.to(torch.float32)

class PositionalEncoding(nn.Module):
    "Implement the PE function."

    def __init__(self, d_model, max_len):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)   
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).cuda()
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[:, : x.size(1)].requires_grad_(False)
        return x
    



