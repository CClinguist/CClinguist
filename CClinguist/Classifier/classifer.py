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
import pickle
# from utils import torch_load_cpu
from datetime import datetime, timedelta


currentDateAndTime = datetime.now()
TIME=str(currentDateAndTime.month)+'_'+str(currentDateAndTime.day)+'_'+str(currentDateAndTime.hour)+'_'+str(currentDateAndTime.minute)
torch.set_printoptions(sci_mode=False)

wandb.login(key="")
sweep_config = {
    'method': 'bayes',   #Need at least one searchable parameter
    #'method': 'random', 
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
            'value': 0.0003
        },
        'epochs':{
            #'value':5000
            'value': 1
        },
        'batch_size':{
            #'value':60  
            'value': 60
        },
        'num_cluster':{
            'value':12
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
            'value':'data/07_27_dp_40_100KB' 
        },
        'train':{
            'value': False
        },
        'load_model_path':{
            'value': 'CClinguist/Classifier/checkpoints/training/best_model_accuracy.pth.tar'
            
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
        'margin':{
            'value': 2.5
        },
        'weight_verf':{
            'value': 0.1        
        },
        'target_envs':{
            'value': None
        }
}}




sweep_id = wandb.sweep(sweep_config, project="Classifier_unknown")

feature = [[0] * 12 for _ in range(12)]
cnt = [0] * 12


class Classifier(nn.Module):
    def __init__(self, num_classify, dim_hidden, num_layers, num_heads,dim_input, dropout_rate):
        super(Classifier, self).__init__()
        
        self.encoder_model = seq_model_transformer(dim_hidden, num_layers, num_heads,dim_input, dropout_rate)
        
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
        
        self.fc_constrast = nn.Sequential(
            nn.Linear(3*(dim_input*num_heads)+dim_hidden, dim_hidden, device='cuda'), 
            nn.ReLU(),
            #nn.BatchNorm1d(dim_hidden, device='cuda'),
            nn.LayerNorm(dim_hidden, device='cuda'),
            nn.Linear(dim_hidden, int(dim_hidden/2), device='cuda'),
            nn.ReLU(),
            #nn.BatchNorm1d(int(dim_hidden/2), device='cuda'),
            nn.LayerNorm(int(dim_hidden/2), device='cuda'),
            nn.Linear(int(dim_hidden/2), 32, device='cuda')
        )
        
        # margin = torch.rand(size=(1,), requires_grad=True,device='cuda')   
        # self.margin=nn.Parameter(torch.tensor(margin, requires_grad=True))

    def forward(self, x, length, x_env, max_length):
        feature_encoded = self.encoder_model(x, length, x_env, max_length)
        self.feature_constrast=self.fc_constrast(feature_encoded)
        
        self.output_classify = self.fc(feature_encoded)
        return self.output_classify
    
    def get_loss(self,label,batch_size,weight_verf,margin):
        
        # print(f'---------- get_loss --------------')
        label_input=label[0:batch_size]    
        label_constrast=label[batch_size:] 
        y_label=label_input==label_constrast
        y_label = torch.where(y_label, torch.tensor(1), torch.tensor(-1))
        idx_1=torch.where(y_label==1)
        idx_0=torch.where(y_label==-1)
        
        f_input=self.output_classify[0:batch_size,:]
        f_constrast=self.output_classify[batch_size:,:]


        for i in label_constrast.tolist():
            cnt[i] += 1
        for item in zip(f_constrast.tolist(), label_constrast.tolist()):
            idx = item[1]
            for i in range(12):
                feature[idx][i] += item[0][i]
        
        for i in label_input.tolist() + label_constrast.tolist():
            cnt[i] += 1
            print(i)
        for item in zip(f_input.tolist() + f_constrast.tolist(), label_input.tolist() + label_constrast.tolist()):
            idx = item[1]
            for i in range(len(item[0])):
                print(i)
                print(idx)
                feature[idx][i] += item[0][i]
        
        loss_dis_out=F.pairwise_distance(f_input[idx_0], f_constrast[idx_0], p=2)
        loss_dis_in=F.pairwise_distance(f_input[idx_1], f_constrast[idx_1], p=2)
        loss_verf=0.5* torch.pow(loss_dis_in.mean(),2)+0.5*torch.max(torch.tensor(0),(margin-torch.pow(loss_dis_out,2)).mean())
        
        criterion = nn.CrossEntropyLoss()
        loss_ident_1 = criterion(f_input,label_input)
        loss_ident_2 = criterion(f_constrast,label_constrast)
            
        loss = loss_ident_1 + loss_ident_2+ weight_verf * loss_verf

        return loss
    
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
    with wandb.init(config=config):
        config = wandb.config
        exp_dir = f'classify_unknown/{wandb.run.name}'
        if config.seed is not None:
            random.seed(config.seed)
            torch.manual_seed(config.seed)
            np.random.seed(config.seed)
        if not os.path.exists(exp_dir):
            os.makedirs(exp_dir)
        
        
        dataset=dataset_build(config)
        train_loader,max_length_train=dataset.forward(path=config.train_path, train=config.train, train_data_num=config.train_data_num, target_envs=config.target_envs)
        eval_loader,max_length_eval=dataset.forward(path=config.test_path, train=False, train_data_num=config.train_data_num, target_envs=config.target_envs)
        
        if len(eval_loader) == 0:
            print("eval_loader is empty.")
        else:
            print("eval_loader is not empty.")
        
        model = Classifier(config.num_cluster,config.dim_hidden,config.num_layers,config.num_heads,config.dim_input, config.dropout_rate)
       
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
        
        losses=0
        
        accuracy = torchmetrics.Accuracy(task='multiclass',num_classes=config.num_cluster,top_k=1)
        if config.load_model_path is not None:
            checkpoint=torch.load(config.load_model_path)
            load_data = torch.load(config.load_model_path, map_location=lambda storage, loc: storage)
            
            model.load_state_dict(checkpoint['state_dict'], strict=False)
            model.to('cuda')
            
            optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
            
            optimizer_state = load_data['optimizer']


            
            new_optimizer_state = optimizer.state_dict()
            for key in optimizer_state['state'].keys():
                if key in new_optimizer_state['state']:
                    new_optimizer_state['state'][key] = optimizer_state['state'][key]
                    
            new_optimizer_state['param_groups'][0]['lr'] = optimizer_state['param_groups'][0]['lr']
            new_optimizer_state['param_groups'][0]['betas'] = optimizer_state['param_groups'][0]['betas']
            new_optimizer_state['param_groups'][0]['eps'] = optimizer_state['param_groups'][0]['eps']
            new_optimizer_state['param_groups'][0]['weight_decay'] = optimizer_state['param_groups'][0]['weight_decay']

           
            optimizer.load_state_dict(new_optimizer_state)
            
        min_test_loss = float('inf')
        min_test_accuracy = 0
        
        for epoch in tqdm(range(0, config.epochs)):
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
                    loss=model.get_loss(label.cuda(),config.batch_size,config.weight_verf,config.margin)
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

                print(f"Training acc for epoch {epoch}: {total_train_accuracy}")
                wandb.log({"train_loss": losses, "train_accuracy": total_train_accuracy, "epoch": epoch})
                accuracy.reset()
                
            
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
                    loss=model.get_loss(label.cuda(),config.batch_size,config.weight_verf,config.margin)
                    
                if output_classify.size(0) == 1:
                    output_classify = output_classify.unsqueeze(0)
                softmax=torch.nn.Softmax(dim=1)
                x=softmax(output_classify)

                batch_acc = test_accuracy(torch.argmax(softmax(output_classify),dim=1).cpu(), label)
                
                losses+=loss.item()
                
                
            total_test_accuracy = test_accuracy.compute()
            print(f"Testeing acc for epoch {epoch}: {total_test_accuracy}")
            wandb.log({"test_loss": losses, "test_accuracy": total_test_accuracy, "epoch": epoch})
            
            filename = f'{exp_dir}/checkpoint_{epoch+1:04d}.pth.tar'
            if (epoch+1) % 1000 == 0 and config.train: 
                save_checkpoint({
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'optimizer' : optimizer.state_dict(),
                }, is_best=False, filename=filename)
            
            is_best_accuracy = total_test_accuracy > min_test_accuracy and total_test_accuracy > 0.75
            is_best_loss = losses < min_test_loss and losses < 40

            if config.train and (is_best_accuracy or is_best_loss):
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
                print(f'The best model has been saved and the test accuracy is: {total_test_accuracy}\n')
            torch.cuda.empty_cache()
            f.close()



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
        
        
    def forward(self,path,train=True,train_data_num=2, target_envs=None):
        envs=os.listdir(path)
        print(envs)
        alo_id=0
        
        y=[]
        x_trace=[]
        x_env=[]
        x_len=[]
        max_length=0
        alo_dic={'htcp': 0, 'bbr': 1, 'vegas': 2, 'westwood': 3, 'scalable': 4, 'highspeed': 5, 'veno': 6, 'reno': 7, 'yeah': 8, 'illinois': 9, 'bic': 10, 'cubic': 11, 'unknown': 12}

        for env in envs:
            
            numbers = re.findall(r'\d+', env)
            env_numbers = list(map(int, numbers[:2])) 
            
            y_tmp={} 
            x_trace_tmp={}
            x_env_tmp={}
            x_len_tmp={}
            max_length_env=0
            
            
            alos=os.listdir(os.path.join(path,env))
            
            for alo in alos:
    
                alo_name = re.search(r'_(\w+)_', alo).group(1)
                
                if alo_name == 'dctcp':
                    continue
                if alo_name not in y_tmp.keys():
                    y_tmp[alo_name] = []
                    x_env_tmp[alo_name] = []
                    x_len_tmp[alo_name] = []
                    x_trace_tmp[alo_name] = []
                
                if alo_name in alo_dic:
                    label_i=[alo_dic[alo_name]]
                else:
                    label_i=[len(alo_dic)]

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
                        if len(line)<=1:
                            continue
                        print(line)
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

                if train:  # train_loader
                    if train_data_num % 2 == 0:
                        raise ValueError("num must be an odd number")

                    neighbors_count = (train_data_num - 1) // 2
                    indices_to_add = sorted_indices[median_pos - neighbors_count : median_pos + neighbors_count + 1]

                else: # test_loader
                    indices_to_add = np.arange(len(data_list))
                    
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
        loader = torch.utils.data.DataLoader(input, self.batch_size*2, shuffle=True, drop_last=True)
        return loader,max_length
                            
def read_pickle_file(file_path):
    
    try:
        with open(file_path, 'rb') as file:
            data = pickle.load(file)
            print(len(data[0]), data)
    except Exception as e:
        print(f"Error reading pickle file: {e}")

                        
                

if __name__ == "__main__":
    
    wandb.agent(sweep_id, build_model, count=1)
    
    result_feature = [[feat[i] / (c+0.00001) for i in range(len(feat))] for feat, c in zip(feature, cnt)]
    with open('result_feature.pkl', 'wb') as f:
        pickle.dump(result_feature, f)