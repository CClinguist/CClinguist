
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
from preprocess_newdata import delayDataset_build
from model_unknown import Classifier


from torch.cuda.amp import autocast, GradScaler

# DDP
import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group, broadcast
from config_module import Config


os.environ["CUDA_VISIBLE_DEVICES"] = "0"

#currentDateAndTime = datetime.now()
#TIME=str(currentDateAndTime.month)+'_'+str(currentDateAndTime.day)+'_'+str(currentDateAndTime.hour)+'_'+str(currentDateAndTime.minute)
torch.set_printoptions(sci_mode=False)

wandb.login(key="")


# smooth-sweep-3_best 

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
            #'values': [128, 256]
            'value': 128
        },
        'num_layers': {
            #'values': [4, 8, 16]
            'value': 8
        },
        'num_heads': {
            #'values': [4, 8, 16]
            'value': 8
        },
        'learning_rate': {
            # 'distribution': 'uniform',
            # 'min': 5e-4,
            # 'max': 1e-3
            'value': 1e-4  # smooth-sweep-3_best 
        },
        'epochs':{
            'value':1
        },
        'batch_size':{
            'value':32
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
            'value': 'data/simulation_hk2pk'
        },
        'test_path':{
            'value': 'data/simulation_hk2jp_1201to1203'
        },
        'train':{
            'value': True
        },
        'load_model_path':{
            'value': None
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
            'value':10
        },
        'dropout_rate':{
            'values':[0.1, 0.2, 0.3]
            # 'values': [0.0, 0.1, 0.2]
            #'value': 0.1
        },
        'update_layers':{
            #'values':['fc', 'fc.6']
            'value': None
        },
        'max_len':{
            #'values':[-1, 256, 512, 1024]
            'value': 512
        },

        'margin':{
            'values': [2.5]   
            
        },
        'weight_verf':{
            'values':[0.00485668544506156]  
            # 'distribution': 'uniform',
            # 'min': 1e-3, 
            # #'max': 0.0002
            # 'max': 5e-3
            # 'values': [0.1]   
            
        },
    }
}



def ddp_setup():
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    init_process_group(backend="nccl")
    
    
    
def save_checkpoint(state, is_best, filename, save_type=None):
    if is_best:
        dir_name = os.path.dirname(filename)
        best_model_filename = f'best_model_{save_type}.pth.tar'
        best_model_path = os.path.join(dir_name, best_model_filename)
        torch.save(state, best_model_path)
        print(f'The best checkpoint has been saved as {best_model_path}')
    else:
        torch.save(state, filename)

def broadcast_model(model, rank):
    for param in model.parameters():
        broadcast(param.data, src=0)

#def build_model(world_size, config=None):  
def build_model(config=None):
    # torchrun
    #rank = int(os.environ['RANK'])
    #local_rank = int(os.environ['LOCAL_RANK'])
    
    # ddp 
    #torch.cuda.set_device(local_rank)  
    #ddp_setup() 
    
    # without torchrun
    rank = 0
    local_rank = 0
    
    #print(f'global_rank: {rank}, local_rank: {local_rank}')
    
    with wandb.init(config=config):
        config = wandb.config
        exp_dir = f'checkpoint_1201_newdata/{wandb.run.name}'
        if config.seed is not None:
            random.seed(config.seed)
            torch.manual_seed(config.seed)
            np.random.seed(config.seed)
        if not os.path.exists(exp_dir):
            os.makedirs(exp_dir)
            
        
        dataset=delayDataset_build(config).cuda() 
        
        # DDP
        # train_loader,max_length_train=dataset.forward(path=config.train_path, train=config.train, train_data_num=config.train_data_num, rank=rank)
        # eval_loader,max_length_eval=dataset.forward(path=config.test_path, train=False, train_data_num=config.train_data_num, rank=rank)
        
        print(f'train_loader has been created!')
        if config.train:
            train_loader,max_length_train,train_loader_contrast=dataset.forward(path=config.train_path, train=config.train, train_data_num=config.train_data_num)
        eval_loader,max_length_eval,eval_loader_contrast=dataset.forward(path=config.test_path, train=False, train_data_num=config.train_data_num)
        
        # Create the classify model
        model = Classifier(config.num_cluster,config.dim_hidden,config.num_layers,config.num_heads,config.dim_input, config.dropout_rate,config.margin, config.weight_verf)
        # model = Classifier(config.num_cluster,config.dim_hidden,config.num_layers,config.num_heads,config.dim_input, config.dropout_rate,config.margin, config.weight_verf)
        print(f'model has been created!')
        #model.to(device)
        model.cuda() 
        #print(f'model has been moved to cuda:{local_rank}!')
        # if rank == 0:
        #     broadcast_model(model, rank)
        #     print(f'broadcast_model has been finished!')
        
        #model = DDP(model, device_ids=[local_rank])
        #print(f'DDP passed')
        
        scaler = GradScaler()
        
        
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
            #optimizer = torch.optim.Adam(model.parameters(), lr=0.00017508698028190282) 
            optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate) 
        
        criterion = nn.CrossEntropyLoss().cuda()
        losses=0
        pre_epoch = 0
        # DDP
        accuracy = torchmetrics.Accuracy(task='multiclass',num_classes=config.num_cluster,top_k=1).cuda()
        #accuracy = torchmetrics.Accuracy(task='multiclass',num_classes=config.num_cluster,top_k=1).to(device)

        if config.load_model_path is not None:
            checkpoint=torch.load(config.load_model_path)
            pre_epoch = checkpoint['epoch']
            
            # dim_hidden_half = model.fc[6].in_features
            
            # model.fc[6] = nn.Linear(dim_hidden_half, config.num_cluster, device='cuda')
            
            # del checkpoint['state_dict']['fc.6.weight']
            # del checkpoint['state_dict']['fc.6.bias']

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
            # f=open(f'{exp_dir}/result_classify.txt','a+')
            # f.write(f'-------------------------------start epoch {epoch}---------------------------\n')
            if config.train:
                model.train()
                
                # DDP
                #train_loader.sampler.set_epoch(epoch)
                #print(f'train_loader.sampler has been finished!')
                
                for batch_id, input in enumerate(train_loader):  
                    #print(f"Batch {batch_id} - Before processing: Memory allocated: {torch.cuda.memory_allocated()} bytes")#\n In detail: {torch.cuda.memory_summary(device=0, abbreviated=False)}")
                    optimizer.zero_grad()
                    
                    
                    with torch.autograd.profiler.profile(enabled=True, use_cuda=True) as prof:
                        cca_different_collect=next(iter(train_loader_contrast))   
                        input_a=torch.cat((input[0:config.batch_size,:],cca_different_collect[0]),dim=0)
                        input_b=torch.cat((input[config.batch_size:,:],cca_different_collect[1]),dim=0)
                        input=torch.cat((input_a,input_b),dim=0).cuda()
                        label=input[:,3].to(torch.int64)
                        length=input[:,0].to(torch.int64)
                        x_env=input[:,1:3]
                        data_trace=torch.reshape(input[:,4:],[-1,max_length_train,config.dim_input])
                        #print(f'label: {label}, length:{length}, x_env:{x_env}, data_x_trace: {data_trace}')
                        #print(f"Batch {batch_id} - After loading data: Memory allocated: {torch.cuda.memory_allocated()} bytes")
                        
                        # output_classify=model.forward(data_trace,length,x_env,max_length_train)
                        # #DDP
                        # softmax = torch.nn.Softmax(dim=1).cuda()
                        # #softmax=torch.nn.Softmax(dim=1).to(device)
                        
                        # loss=criterion(output_classify,label)#.cuda())
                        # loss.backward()
                        # print(f"Batch {batch_id} - After backward: Memory allocated: {torch.cuda.memory_allocated()} bytes")#\n In detail: {torch.cuda.memory_summary(device=0, abbreviated=False)}")
                        # optimizer.step()
                        # print(f"Batch {batch_id} - After optimization: Memory allocated: {torch.cuda.memory_allocated()} bytes")#\n In detail: {torch.cuda.memory_summary(device=0, abbreviated=False)}")
    
                        with autocast():
                            output_classify = model.forward(data_trace, length, x_env, max_length_train)
                            softmax = torch.nn.Softmax(dim=1).cuda()
                            loss,margin=model.get_loss(label.cuda(),config.batch_size)
                        scaler.scale(loss).backward()
                        scaler.step(optimizer)
                        scaler.update()
                    #print(prof.key_averages().table(sort_by="self_cuda_time_total"))
                    #print(f'Memory summary: {torch.cuda.memory_summary(device=0, abbreviated=False)}')
                    batch_acc = accuracy(torch.argmax(softmax(output_classify),dim=1), label)
                    losses+=loss.item()
                    torch.cuda.empty_cache()
                    #print(f"Epoch {epoch}, Batch {batch_id}, Memory allocated: {torch.cuda.memory_allocated()} bytes")
                
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
                # print(losses)
                print(f"Training acc for epoch {epoch}: {total_train_accuracy}")
                #f.write(f'train loss: {losses} train_accuracy: {total_train_accuracy}    ')
                margin_value=margin.data.item()
                wandb.log({"train_loss": losses, "train_accuracy": total_train_accuracy, "epoch": epoch, "margin": margin_value})
                accuracy.reset()
                # if epoch == pre_epoch + 100 and total_train_accuracy < 0.1:
                #     break
            
            #---------------------------------------------test-------------------------------------------------------
            losses=0
            test_all_label=[]
            test_all_label_pre=[]
            #DDP
            test_accuracy = torchmetrics.Accuracy(task='multiclass',num_classes=config.num_cluster,top_k=1).cuda()
            
            #test_accuracy = torchmetrics.Accuracy(task='multiclass',num_classes=config.num_cluster,top_k=1).to(device)
            model.eval()
            for i,input in enumerate(eval_loader):
                cca_different_collect=next(iter(eval_loader_contrast))  
                input_a=torch.cat((input[0:config.batch_size,:],cca_different_collect[0]),dim=0)
                input_b=torch.cat((input[config.batch_size:,:],cca_different_collect[1]),dim=0)
                input=torch.cat((input_a,input_b),dim=0).cuda()
                label=input[:,3].to(torch.int64)
                length=input[:,0].to(torch.int64)
                x_env=input[:,1:3]
                data_trace=torch.reshape(input[:,4:],[-1,max_length_eval,config.dim_input])
                
               
                with torch.no_grad():
                    output_classify=model.forward(data_trace,length,x_env,max_length_eval)
                softmax = torch.nn.Softmax(dim=1).cuda()
                #softmax=torch.nn.Softmax(dim=1).to(device)
                x=torch.argmax(softmax(output_classify),dim=1)

                batch_acc = test_accuracy(torch.argmax(softmax(output_classify),dim=1), label)
                loss,margin=model.get_loss(label.cuda(),config.batch_size)
                losses+=loss.item()
                torch.cuda.empty_cache()
                #print(f"Test Batch {i}, Memory allocated: {torch.cuda.memory_allocated()} bytes")

                
                # probabilities_list = x.tolist()
                # labels_list = label.tolist()
                #print(f"Batch {i + 1} - Predicted Probabilities: {probabilities_list}, True Labels: {labels_list}\n")
                #f.write(f"Epoch: {epoch+1}, Batch {i + 1} - Predicted Probabilities: {probabilities_list}, True Labels: {labels_list}\n")
                
                
            total_test_accuracy = test_accuracy.compute()
            # print(losses)
            #f.write(f'test loss: {losses} test_accuracy: {total_test_accuracy}\n')\
            print(f"Test acc for epoch {epoch}: {total_test_accuracy}")
            wandb.log({"test_loss": losses, "test_accuracy": total_test_accuracy, "epoch": epoch, "margin": margin.data.item()})
            
            filename = f'{exp_dir}/checkpoint_{epoch+1:04d}.pth.tar'
            # DDP
            if (epoch+1) % 100 == 0 and rank == 0: 
            #if (epoch+1) % 1000 == 0: 
                save_checkpoint({
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'optimizer' : optimizer.state_dict(),
                }, is_best=False, filename=filename)
            
            is_best_accuracy = total_test_accuracy > min_test_accuracy and total_test_accuracy > 0.7
            is_best_loss = losses < min_test_loss and losses < 50

            # DDP
            if (is_best_accuracy or is_best_loss) and rank == 0:
           
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
                #f.write(f'The best model has been saved and the test accuracy is: {total_test_accuracy}\n')
            torch.cuda.empty_cache()
            #f.close()

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



if __name__ == "__main__":
   
    # DDP
    #world_size = 1
    online_flag =True  
    
    
    if online_flag:
        sweep_id = wandb.sweep(sweep_config, project="Classifier_cclinguist_newdata_seqLen_512_unknown_1205")
        #DDP
        #wandb.agent(sweep_id, build_model_wrapper, count=5)
        wandb.agent(sweep_id, build_model, count=10)
    else:
        #config_dict = {key: value['value'] for key, value in sweep_config['parameters'].items()}
        # config_dict = {
        #     key: (
        #         random.uniform(value['min'], value['max']) if key == 'learning_rate' else value.get('value', 0.1)
        #     )
        #     for key, value in sweep_config['parameters'].items()
        # }
        config_dict = {
            key: (
                random.uniform(value['min'], value['max']) if key == 'learning_rate' else
                random.choice(value['values']) if 'values' in value else
                value.get('value', 0.1)
            )
            for key, value in sweep_config['parameters'].items()
        }
        
        config = Config(config_dict)
        print(f'config:\n{config_dict}')
        build_model(config)
        
        #DDP
        
        #mp.spawn(build_model, args=(world_size,config), nprocs=1, join=True)
        
        
        