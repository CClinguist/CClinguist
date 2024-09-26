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
from tqdm import tqdm

from datetime import datetime, timedelta
torch.set_printoptions(sci_mode=False)

wandb.login(key="")


sweep_config = {
    #'method': 'bayes',   #Need at least one searchable parameter
    'method': 'random', 
    'metric': {
        'name': 'test_accuracy',
        'goal': 'maximize'
    },
    'parameters': {
        'dim_input': {
            'value': 2
        },
        'dim_hidden': {
            'value': 256
        },
        'num_layers': {
            'value': 8
        },
        'num_heads': {
            'value': 16
        },
        'learning_rate': {
            'value': 0.002021127440586036  # fanciful-sweep-1_chosen4incremental
        },
        'epochs':{
            'value':15000
        },
        'batch_size':{
            'value':32
        },
        'num_cluster':{
            'value':13
        },
        'data_position':{
            'value':'host'
        },
        'seed':{
            'value':3402
        },
        'train_path':{
            'value': 'data/60-300_200-1000'
        },
        'test_path':{
            'value': 'data/60-300_200-1000'
        },
        'train':{
            'value': True
        },
        'load_model_path':{
            'value': './checkpoints/incremental_training/BaseModel.pth.tar'
        },
        'optimizer':{
            #'values': ['sgd', 'adam']
            'value':'adam'
        },
        'scheduler':{
            #'values': ['ReduceLROnPlateau', 'CosineAnnealingLR', None]
            'value': None
        },
        'min_lr':{
            'value': 1e-6
        },
        'train_data_num':{
            'value':5
        },
        'dropout_rate':{
            'value': 0.3
        },
        'update_layers':{
            'value': 'fc'
        }
    }
}



pp.pprint(sweep_config)
sweep_id = wandb.sweep(sweep_config, project="")




class Classifier(nn.Module):
    def __init__(self, num_classify, dim_hidden, num_layers, num_heads,dim_input, dropout_rate):
        super(Classifier, self).__init__()
        
        self.encoder_model = seq_model_transformer(dim_hidden, num_layers, num_heads,dim_input, dropout_rate)
        
        self.fc = nn.Sequential(
            nn.Linear(3*(dim_input*num_heads)+dim_hidden, dim_hidden, device='cuda'), 
            nn.ReLU(),
            nn.LayerNorm(dim_hidden, device='cuda'),
            nn.Linear(dim_hidden, int(dim_hidden/2), device='cuda'),
            nn.ReLU(),
            nn.LayerNorm(int(dim_hidden/2), device='cuda'),
            nn.Linear(int(dim_hidden/2), num_classify, device='cuda')
        )

    def forward(self, x, length, x_env, max_length):
        feature_encoded = self.encoder_model(x, length, x_env, max_length)
        
        output_classify = self.fc(feature_encoded)
        return output_classify
    
class seq_model_transformer(nn.Module):
    def __init__(self, dim_hidden, num_layers, num_heads,dim_input, dropout_rate):
        super().__init__()
        self.num_heads = num_heads  
        self.dim_input = dim_input
        self.dim_hidden = dim_hidden
        self.num_layers = num_layers
        self.dropout_rate = dropout_rate
        self.input_encoder = nn.Linear(self.dim_input, self.num_heads*self.dim_input, device='cuda')
        self.transformer_layer = nn.TransformerEncoderLayer(d_model=self.dim_input*self.num_heads, nhead=self.num_heads, dim_feedforward=self.dim_hidden, dropout=self.dropout_rate ,batch_first=True, device='cuda')
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
    with wandb.init(config=config, mode='disabled'):
        config = wandb.config
        exp_dir = f'./checkpoints/incremental_training'
        if config.seed is not None:
            random.seed(config.seed)
            torch.manual_seed(config.seed)
            np.random.seed(config.seed)
        if not os.path.exists(exp_dir):
            os.makedirs(exp_dir)
        
        
        dataset=dataset_build(config)
        train_loader,max_length_train=dataset.forward(path=config.train_path, train=config.train, train_data_num=config.train_data_num)
        eval_loader,max_length_eval=dataset.forward(path=config.test_path, train=False, train_data_num=config.train_data_num)
        model = Classifier(config.num_cluster,config.dim_hidden,config.num_layers,config.num_heads,config.dim_input, config.dropout_rate)
        # Define the optimizer and loss function
        
        if config.optimizer == 'adam':
            optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
        elif config.optimizer == 'sgd':
            optimizer = torch.optim.SGD(model.parameters(), lr=config.learning_rate, momentum=0.9)
        
        
        if config.scheduler == 'ReduceLROnPlateau':
            #scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.95, patience=20, verbose=True)
            #scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.95, patience=10, verbose=True)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.95, patience=5, verbose=True)
        elif config.scheduler == 'CosineAnnealingLR':
            #scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50, eta_min=config.min_lr)
            #scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20, eta_min=config.min_lr)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50, eta_min=config.min_lr)

        elif config.scheduler == None and config.load_model_path is None:
            optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate) 
        
        criterion = nn.CrossEntropyLoss()
        losses=0
        pre_epoch = 0
        accuracy = torchmetrics.Accuracy(task='multiclass',num_classes=config.num_cluster,top_k=1)
        if config.load_model_path is not None:
            checkpoint=torch.load(config.load_model_path)
            pre_epoch = checkpoint['epoch']
            load_data = torch.load(config.load_model_path, map_location=lambda storage, loc: storage)
            
            dim_hidden_half = model.fc[6].in_features
            model.fc[6] = nn.Linear(dim_hidden_half, config.num_cluster, device='cuda')
            del checkpoint['state_dict']['fc.6.weight']
            del checkpoint['state_dict']['fc.6.bias']


            model.load_state_dict(checkpoint['state_dict'], strict=False)
            model.to('cuda')
            if config.update_layers:
                # Freeze the encoder_model part
                for name, param in model.named_parameters():
                    if not name.startswith(config.update_layers): 
                        param.requires_grad = False
                if config.update_layers == 'fc':
                    optimizer = torch.optim.Adam(model.fc.parameters(), lr=config.learning_rate)
                elif config.update_layers == 'fc.6':
                    optimizer = torch.optim.Adam(model.fc[6].parameters(), lr=config.learning_rate)
        
        min_test_loss = float('inf')
        min_test_accuracy = 0
        
        for epoch in tqdm(range(pre_epoch, pre_epoch+config.epochs)):
            losses=0
            if config.train:
                model.train()
                for batch_id, input in enumerate(train_loader):  
                    label=input[:,3].to(torch.int64)
                    length=input[:,0].to(torch.int64)
                    x_env=input[:,1:3]
                    data_trace=torch.reshape(input[:,4:],[-1,max_length_train,config.dim_input])
                    output_classify=model.forward(data_trace,length,x_env,max_length_train)
                    softmax=torch.nn.Softmax(dim=1)
                    batch_acc = accuracy(torch.argmax(softmax(output_classify),dim=1).cpu(), label)
                    loss=criterion(output_classify,label.cuda())
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    losses+=loss.item()
                
                total_train_accuracy = accuracy.compute()
                if config.scheduler:
                    if config.scheduler == 'ReduceLROnPlateau':
                        scheduler.step(losses)
                    else:
                        scheduler.step()
                    for param_group in optimizer.param_groups: 
                        if param_group['lr'] < config.min_lr:
                            param_group['lr'] = config.min_lr
                            print(f"Learning rate adjusted to minimum value {config.min_lr}")
                wandb.log({"train_loss": losses, "train_accuracy": total_train_accuracy, "epoch": epoch})
                accuracy.reset()
                if epoch == pre_epoch + 500 and total_train_accuracy < 0.3:
                    break
            
            losses=0
            test_all_label=[]
            test_all_label_pre=[]
            test_accuracy = torchmetrics.Accuracy(task='multiclass',num_classes=config.num_cluster,top_k=1)
            model.eval()
            for i,input in enumerate(eval_loader):
                label=input[:,3].to(torch.int64)
                length=input[:,0].to(torch.int64)
                x_env=input[:,1:3]
                data_trace=torch.reshape(input[:,4:],[-1,max_length_eval,config.dim_input])
                with torch.no_grad():
                    output_classify=model.forward(data_trace,length,x_env,max_length_eval)
                softmax=torch.nn.Softmax(dim=1)
                x=softmax(output_classify)

                batch_acc = test_accuracy(torch.argmax(softmax(output_classify),dim=1).cpu(), label)
                loss=criterion(output_classify,label.cuda())
                losses+=loss.item()
                
                
            total_test_accuracy = test_accuracy.compute()
            wandb.log({"test_loss": losses, "test_accuracy": total_test_accuracy, "epoch": epoch})
            
            filename = f'{exp_dir}/checkpoint_{epoch+1:04d}.pth.tar'
            if (epoch+1) % 2000 == 0: 
                save_checkpoint({
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'optimizer' : optimizer.state_dict(),
                }, is_best=False, filename=filename)
            
            is_best_accuracy = total_test_accuracy > min_test_accuracy and total_test_accuracy > 0.7
            is_best_loss = losses < min_test_loss and losses < 45

            if is_best_accuracy or is_best_loss: 
                if is_best_accuracy:
                    save_type = 'accuracy'
                    min_test_accuracy = total_test_accuracy
                else:
                    save_type = 'loss'
                    min_test_loss = losses
                    
                save_checkpoint({
                    'epoch': epoch,
                    'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                }, is_best=True, filename=filename, save_type=save_type)
            torch.cuda.empty_cache()

def convert_to_seconds(time_str):
    time_format = "%H:%M:%S.%f"
    dt = datetime.strptime(time_str, time_format)
    total_seconds = dt.hour * 3600 + dt.minute * 60 + dt.second + dt.microsecond / 1e6
    return total_seconds


class dataset_build(nn.Module):
    def __init__(self, config):
        super(dataset_build, self).__init__()
        self.batch_size=config.batch_size
        self.data_position=config.data_position
        
        
    def forward(self,path,train=True,train_data_num=2):
        envs=os.listdir(path)
        alo_id=0
        
        y=[]
        x_trace=[]
        x_env=[]
        x_len=[]
        max_length=0
        alo_dic={'htcp': 0, 'bbr': 1, 'vegas': 2, 'westwood': 3, 'scalable': 4, 'highspeed': 5, 'veno': 6, 'reno': 7, 'yeah': 8, 'illinois': 9, 'bic': 10, 'cubic': 11, 'dctcp': 12}

        for env in envs:
            y_tmp={}  
            x_trace_tmp={}
            x_env_tmp={}
            x_len_tmp={}
            max_length_env=0
            
            
            alos=os.listdir(os.path.join(path,env))
            #plt.figure(figsize=(10, 6))
            
            for alo in alos:
                #label_i=[alo_dic[alo]]
                
                #alo_name= re.search( r'_(.+)\.', alo).group(1)
                alo_name = re.search(r'_(\w+)_', alo).group(1)

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

                    f=open(os.path.join(path,env)+f'/{capture_filename}','r')
                    lines=f.readlines()
                    f.close()
                    data_capture=[]
                    previous_seconds = None
                    day_in_seconds = 24 * 3600
                    for line in lines:
                        line=line.strip()
                        line=line.split(' ')
                        if line[0]=='seq':
                            total_seconds = convert_to_seconds(line[1])
                            if previous_seconds is not None and total_seconds < previous_seconds:
                                total_seconds += day_in_seconds 
                            previous_seconds = total_seconds
                            data_capture.append([total_seconds,int(line[-1])])
                    data_capture = np.array(data_capture)
                    
                    if len(data_capture) == 0:
                        continue
                    
                    f=open(os.path.join(path,env)+'/'+alo,'r')  #host
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
                            total_seconds = convert_to_seconds(line[0])
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

                if train: 
                    if train_data_num % 2 == 0:
                        raise ValueError("num must be an odd number")

                    neighbors_count = (train_data_num - 1) // 2
                    indices_to_add = sorted_indices[median_pos - neighbors_count : median_pos + neighbors_count + 1]

                else: 
                    indices_to_add = [sorted_indices[0]]
                    
                for index in indices_to_add:
                    data_x_trace_tmp = x_trace_tmp[alo_name][index]
                    scaler = MinMaxScaler()
                    data_x_trace_tmp = torch.tensor(scaler.fit_transform(data_x_trace_tmp), dtype=torch.float32)
                    y.append(y_tmp[alo_name][index])
                    x_env.append(x_env_tmp[alo_name][index])
                    x_len.append(x_len_tmp[alo_name][index])
                    x_trace.append(data_x_trace_tmp)

                    if len(data_x_trace_tmp) > max_length:
                        max_length = len(data_x_trace_tmp)
                    
            
            
            
        data_trace = rnn.pad_sequence(x_trace, batch_first=True, padding_value=0)
        data_trace=torch.reshape(data_trace,[len(y),-1])
        input=torch.cat([torch.Tensor(x_len),torch.Tensor(x_env),torch.Tensor(y),data_trace],dim=1)
        loader = torch.utils.data.DataLoader(input, self.batch_size, shuffle=True, drop_last=True)
        return loader,max_length
                        
            
            
        

if __name__ == "__main__":
    wandb.agent(sweep_id, build_model, count=1)