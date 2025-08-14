
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
# import wandb
import numpy.linalg as LA
import matplotlib.pyplot as plt
import shutil
import pandas as pd
from tqdm import tqdm
from datetime import datetime, timedelta
from torch.nn.utils.rnn import pad_sequence
from preprocess_newdata_learning_basedcc import Dataset_build
from model_unknown import Classifier
import pickle
import os
import argparse


from torch.cuda.amp import autocast, GradScaler
# wandb.init(mode='offline') 

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

#currentDateAndTime = datetime.now()
#TIME=str(currentDateAndTime.month)+'_'+str(currentDateAndTime.day)+'_'+str(currentDateAndTime.hour)+'_'+str(currentDateAndTime.minute)
torch.set_printoptions(sci_mode=False)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


    
    
def save_checkpoint(state, is_best, filename, save_type=None):
    if is_best:
        dir_name = os.path.dirname(filename)
        best_model_filename = f'best_model_{save_type}.pth.tar'
        best_model_path = os.path.join(dir_name, best_model_filename)
        torch.save(state, best_model_path)
        print(f'The best checkpoint has been saved as {best_model_path}')
    else:
        torch.save(state, filename)




def build_model(config=None):
    
    # with wandb.init(config=config):
        # config = wandb.config
        config = config
        # exp_dir = f'checkpoint_0501/{wandb.run.name}'
        if config.seed is not None:
            random.seed(config.seed)
            torch.manual_seed(config.seed)
            np.random.seed(config.seed)
        # if not os.path.exists(exp_dir):
        #     os.makedirs(exp_dir)
            
        
        dataset=Dataset_build(config).cuda() 
        
        
        
        # Create the classify model
        model = Classifier(config.num_cluster,config.dim_hidden,config.num_layers,config.num_heads,config.dim_input, config.dropout_rate,config.margin, config.weight_verf)
        print(f'model has been created!')
        model.to(device)
        model.eval()
        
        
        scaler = GradScaler()
        criterion = nn.CrossEntropyLoss().cuda()
        losses=0
        # DDP
        accuracy = torchmetrics.Accuracy(task='multiclass',num_classes=config.num_cluster,top_k=1).cuda()
        #accuracy = torchmetrics.Accuracy(task='multiclass',num_classes=config.num_cluster,top_k=1).to(device)
        if config.load_model_path is not None:
            checkpoint=torch.load(config.load_model_path)
            model.load_state_dict(checkpoint['state_dict'])
            print(model.margin)
            

            
        
        min_test_loss = float('inf')
        min_test_accuracy = 0
        if os.path.exists(f'dis_log/{config.num_cluster}') == False:
            os.makedirs(f'dis_log/{config.num_cluster}')
        
        f=open(f'dis_log/{config.num_cluster}/{config.unknown_cca}'+'.txt', 'a+')
        
        alo_dic={'htcp': 0, 'bbr': 1, 'vegas': 2, 'westwood': 3, 'scalable': 4, 'highspeed': 5, 'veno': 6, 
                 'reno': 7, 'yeah': 8, 'illinois': 9, 'bic': 10, 'cubic': 11, 'pccLatency': 12, 'astraea': 13, 'pccLoss': 14}
        # alo_dic={'htcp': 0, 'bbr': 1, 'vegas': 2, 'westwood': 3, 'scalable': 4, 'highspeed': 5, 'veno': 6, 
        #          'reno': 7, 'yeah': 8, 'illinois': 9, 'bic': 10, 'cubic': 11, 'pcc': 12}
        
        for epoch in tqdm(range(config.epochs)):
            
            #---------------------------------------------test-------------------------------------------------------
            losses=0
            test_all_label=[]
            test_all_label_pre=[]
            model.eval() 
            eval_input, max_length_eval =dataset.forward(path=config.test_path, profile = config.profile, target_cca=config.unknown_cca)
            label=eval_input[:,3].to(torch.int64)
            length=eval_input[:,0].to(torch.int64).cuda()
            x_env=eval_input[:,1:3].cuda()
            data_trace=torch.reshape(eval_input[:,4:],[-1,max_length_eval,config.dim_input]).cuda()
            # data_trace_1 = data_trace[0:1,:,:]
            with torch.no_grad():
                output_classify=model.forward(data_trace,length,x_env,max_length_eval)
                # output_classify = model.forward(data_trace_1,length[0:1],x_env[0:1,:],max_length_eval)
            softmax = torch.nn.Softmax(dim=1).cuda()
            probs = softmax(output_classify)
            pre_label = torch.argmax(softmax(output_classify),dim=1)
            pre_label = pre_label.tolist()
            f.write('data_path: '+config.test_path+'\n')
            f.write(str(config.unknown_cca)+' '+str(config.profile)+'\n')
            
            
            for i in range(len(pre_label)):
                pre_label_i = pre_label[i]
                alo_name = list(alo_dic.keys())[list(alo_dic.values()).index(pre_label_i)]
                constrast_input, max_length_constrast = dataset.forward(path=config.constrast_path, profile = config.profile, target_cca=alo_name, max_data_len=250)
                length=constrast_input[:,0].to(torch.int64).cuda()
                x_env=constrast_input[:,1:3].cuda()
                data_trace=torch.reshape(constrast_input[:,4:],[-1,max_length_constrast,config.dim_input]).cuda()
                with torch.no_grad():
                    output_classify_constrast=model.forward(data_trace,length,x_env,max_length_constrast)
                    
                pre_feature = output_classify[i,:].expand_as(output_classify_constrast)
                loss_dis_out=F.pairwise_distance(pre_feature, output_classify_constrast, p=2)
                # np.set_printoptions(suppress=True)
                f.write(str(config.unknown_cca)+' '+str(alo_name) + ' ' + str(loss_dis_out.mean()) +' ')
                
                np.set_printoptions(suppress=True)
                np.savetxt(f,probs[i,:].cpu().numpy().reshape([1,-1]),fmt='%.2f')
                # f.write('\n')
                
            f.close()
            torch.cuda.empty_cache()
           
        

    #destroy_process_group()


def convert_to_seconds(time_str):
    time_format = "%H:%M:%S.%f"
    dt = datetime.strptime(time_str, time_format)
    total_seconds = dt.hour * 3600 + dt.minute * 60 + dt.second + dt.microsecond / 1e6
    return total_seconds

# def build_model_wrapper(config=None):
#     #rank, local_rank, world_size = ddp_setup()
#     #print(f'rank: {rank}, local_rank: {local_rank}, world_size: {world_size}')
#     build_model(0, 1, config)

def parse_arguments():
    parser = argparse.ArgumentParser(description="Model Configuration")

    parser.add_argument('--dim_input', type=int, default=2, help='Input dimension')
    parser.add_argument('--dim_hidden', type=int, default=128, help='Hidden layer dimension')
    parser.add_argument('--num_layers', type=int, default=8, help='Number of layers')
    parser.add_argument('--num_heads', type=int, default=8, help='Number of attention heads')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='Learning rate')
    parser.add_argument('--epochs', type=int, default=1, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=10, help='Batch size')
    parser.add_argument('--num_cluster', type=int, default=12, help='Number of clusters')
    parser.add_argument('--data_position', type=str, default='host', help='Data position')
    parser.add_argument('--seed', type=int, default=3402, help='Random seed')
    parser.add_argument('--test_path', type=str, default='data_upgrading/simulation_new_CCs_test', help='Test data path')
    parser.add_argument('--constrast_path', type=str, default='data_upgrading/simulation_15CCs_train', help='Contrast data path')
    parser.add_argument('--train', type=bool, default=False, help='Training mode')
    parser.add_argument('--load_model_path', type=str, default='/home/ml4net/LJH/EFAAA/1209_cclinguist/models/good-sweep-1/best_model_loss.pth.tar', help='Model load path')
    parser.add_argument('--optimizer', type=str, default='adam', help='Optimizer type')
    parser.add_argument('--scheduler', type=str, default=None, help='Scheduler type')
    parser.add_argument('--train_data_num', type=int, default=7, help='Number of training data points')
    parser.add_argument('--dropout_rate', type=float, default=0.2, help='Dropout rate')
    parser.add_argument('--update_layers', type=str, default='fc', help='Layers to update')
    parser.add_argument('--max_len', type=int, default=250, help='Maximum sequence length')
    parser.add_argument('--margin', type=float, default=2.5, help='Margin for optimization')
    parser.add_argument('--weight_verf', type=float, default=0.06297875727280886, help='Weight verification')
    parser.add_argument('--profile', type=str, default='rtt_160ms_bdw_600Kbps', help='Profile setting')
    parser.add_argument('--unknown_cca', type=str, default='pccLatency', help='Unknown CCA type')

    return parser.parse_args()

if __name__ == "__main__":
   
    online_flag = False  #skip wandb online
    
    if online_flag:
        wandb = True # use your wandb online account
        # sweep_id = wandb.sweep(sweep_config, project="Classifier_cclinguist_unknown")
        # wandb.init(mode="offline")
        # wandb.agent(sweep_id, build_model, count=1)
    else:
        config = parse_arguments()
        unknown_ccas = ['pccLatency', 'astraea', 'pccLoss']  # pccLatency, astraea, pccLoss
        for unknown_cca in unknown_ccas:
            config.unknown_cca = unknown_cca
            print(f'unknown_cca: {config.unknown_cca}')
            num_clusters = [12,13,14,15]  # 12 knownccas, 13 knownccas:pccLatency, 14 knownccas:pccLatency + astraea, 15 knownccas:pccLoss
            dic_cluster_model = {
                12: 'checkpoint/checkpoint_12ccas.pth.tar',
                13: 'checkpoint/checkpoint_13ccas.pth.tar',
                14: 'checkpoint/checkpoint_14ccas.pth.tar',
                15: 'checkpoint/checkpoint_15ccas.pth.tar'
            }
            dic_culster_test_path = {
                12: 'data_upgrading/simulation_new_CCs_test',
                13: 'data_upgrading/simulation_new_CCs_test',
                14: 'data_upgrading/simulation_new_CCs_test',
                15: 'data_upgrading/simulation_new_CCs_test',
            }
            dic_cluster_profile = {  #the best profile configed by profile-tree
                12: 'rtt_160ms_bdw_600Kbps',          
                13: 'rtt_200ms_bdw_600Kbps',
                14: 'rtt_160ms_bdw_600Kbps',
                15: 'rtt_400ms_bdw_400Kbps'
            }

            for num_cluster in num_clusters:
                config.num_cluster = num_cluster
                config.load_model_path = dic_cluster_model[num_cluster]
                config.test_path = dic_culster_test_path[num_cluster]
                config.profile = dic_cluster_profile[num_cluster]
                config.constrast_path = 'data_upgrading/simulation_15CCs_train'
                print(f'num_cluster: {num_cluster}, load_model_path: {config.load_model_path}, test_path: {config.test_path}')
                
                # run the model with the current configuration
                build_model(config)




        
        
        
        