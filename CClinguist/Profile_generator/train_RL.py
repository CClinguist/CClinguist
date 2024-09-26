import numpy as np
import os
import time
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
from torch.nn import DataParallel
import torch.nn.functional as F
import datetime
import math
import warnings
from RL import PPO
warnings.filterwarnings('ignore')


# result output
output_dir = 'Profile_generator/results/state_output'
os.makedirs(output_dir, exist_ok=True)
current_time = datetime.datetime.now().strftime('%m%d-%H%M-%S')
filename = f'state_{current_time}.txt'
filepath = os.path.join(output_dir, filename)


def train_epoch(RL_model, Classify_model, train_data_loader, train_dtw, train_dataset, max_length_train,epoch, opts):
    
    print("Start train epoch {}, for run {}".format(epoch,  opts.run_name))
    
    Classify_model.eval()
    print('start train')
    softmax=torch.nn.Softmax(dim=1)
    num_envs=len(os.listdir(opts.train_data_path))
    losses_ppo=0
    losses_critic=0
    batch_num=0
    num_all_batches=0
    all_steps=0
    cannot_search=0
    # reward_scaling = RewardScaling(shape=1, gamma=opts.gamma)
    # reward_scaling.reset()
    # state_norm = Normalization(shape=opts.num_cluster)  # Trick:state normalization
    
    with open(opts.log_file,'a+') as log_f:
        log_f.write('----------------------train at epoch {}-----------------------------'.format(epoch)+ '\n')
        log_f.close()
        
    state_file = open(filepath, 'w')    
        
    for batch_id, (batch_input) in tqdm(enumerate(train_data_loader)):
        #-----------------load the data collected under the init porfile, and start exploring--------------------------------------------------------
        batch_label=batch_input[:,4].to(torch.int64)  
        print(batch_label)
        batch_env_id=batch_input[:,1].to(torch.int64)       
        init_env_id=batch_env_id
        batch_env=batch_input[:,2:4]
        batch_length=batch_input[:,0].to(torch.int64)
        batch_index=batch_input[:,5].to(torch.int64)
        batch_trace=torch.reshape(batch_input[:,6:],[-1,max_length_train,opts.dim_input])
        # print(batch_trace)
        log_predicted_alo=[]
        t=0
        current_ep_reward=0
        last_continue_index=torch.ones_like(batch_label).to(opts.device).bool()
        last_action=torch.zeros([opts.batch_size,num_envs]).to(opts.device)
        state_mask=torch.ones([opts.batch_size,opts.num_cluster]).to(opts.device)  
        RL_model.buffer.is_terminals.append(last_continue_index.to(device=opts.device))
        
        #-----------------start exploring the environment, and record the trace data--------------------------------------------------------
        while t <= opts.max_deepth:
           
            if t==0:   
                #---------------t==0, without classifier，state=uniform distribution，action= init profile--------------------------------------------
                state=(torch.ones([opts.batch_size,opts.num_cluster])/opts.num_cluster).to(opts.device)
                action=RL_model.select_action(state,opts.batch_size,train_dtw,epoch,last_action,batch_env_id.to(opts.device))  #按照默认action强制选择
            else: 
                #---------------t>0, with classifier，state=softmax distribution，action=greedy--------------------------------------------
                action=RL_model.select_action(state,opts.batch_size,train_dtw,epoch,last_action)
            
            #-----------------record the last action, avoid walking back---------------------------------------------------
            last_action=torch.zeros([opts.batch_size,num_envs]).to(opts.device)   
            last_action[torch.arange(0,opts.batch_size),action-1]=1
            
            #-----------------get the trace data under next profile---------------------------------------------------------------------
            next_data=train_dataset.get_trace(action,batch_index)   
            batch_label=next_data[:,4].to(torch.int64)
            print(batch_label)
            batch_env_id=next_data[:,1]
            batch_env=next_data[:,2:4]
            batch_length=next_data[:,0].to(torch.int64)
            
            batch_index=batch_input[:,5].to(torch.int64)
            batch_trace=torch.reshape(next_data[:,6:],[-1,max_length_train,opts.dim_input])
            
            # ----------------the output of the classifier as the next state--------------------------------------------------------------
            with torch.no_grad():
                next_state = Classify_model(batch_trace,batch_length,batch_env,max_length_train)
                next_state=softmax(next_state).to(opts.device)
                # print(next_state)
                state_file.write(f"{next_state}\n")
                
                
            #-----------------determine whether the data collected under the new action meets the termination condition. If it does, the exploration ends. Otherwise, continue the exploration. 
            # Add the CCAs that have been excluded to the blacklist (probability set to 0), and recalculate the distribution of possible CCAs----------------------------------------------
            flag_final,next_state,continue_index,alo_acc_flag,state_mask=RL_model.is_final(next_state,last_continue_index,batch_label.to(opts.device),state_mask)   #is_final时基于未采取action之前的state判断的
            
            reward=RL_model.cal_reward(action,next_state,train_dtw,continue_index,t,opts.max_deepth,alo_acc_flag)
            # reward = reward_scaling(reward)
            state=next_state   
            
            # saving reward and is_terminals
            RL_model.buffer.rewards.append(reward)
            if t!=opts.max_deepth and flag_final==False:
                RL_model.buffer.is_terminals.append(continue_index.to(device=opts.device))
            last_continue_index=continue_index
            current_ep_reward += reward
            t+=1
            batch_num=batch_id+1
            
            #-----------------end the exploration when all batches confidently find the target CCAs or exceed the maximum depth----------------------------------------------
            if flag_final:     
                break
            
        # update PPO agent
        loss_ppo,loss_critic,actions,logprobs,states,state_values,probs,rewards,predicted=RL_model.update(train_dtw,epoch)
        # print(predicted)
        # print(rewards)
        
        
        log_rewards=rewards.reshape([-1]).cpu().detach().tolist()
        log_actions=actions.reshape([-1]).cpu().detach().tolist()
        losses_ppo+=loss_ppo
        losses_critic+=loss_critic
        
        # ---------------------------------log output----------------------------------------------------------------------------------
        # log_predicted_alo=torch.squeeze(torch.stack(log_predicted_alo, dim=0)) 
        if actions.dim()==1:  
            actions=actions.unsqueeze(0)
            predicted=predicted.unsqueeze(0)
        with open(opts.log_file,'a+') as log_f:
            for i in range(opts.batch_size):
                log_f.write('{}, {}, {}, {}, {}, {}\n'.format(
                    batch_label[i].item(),
                    init_env_id[i].item(),
                    actions[:,i][rewards[:,i]!=0].detach().cpu().numpy(),
                    predicted[:,i][rewards[:,i]!=0].detach().cpu().numpy(),
                    rewards[:,i][rewards[:,i]!=0].detach().cpu().numpy(),
                    len(rewards[:,i][rewards[:,i]!=0])
                ))
                steps=len(rewards[:,i][rewards[:,i]!=0])
                all_steps+=steps
                num_all_batches+=1
                if len(rewards[:,i][rewards[:,i]!=0])==opts.max_deepth+1:
                    cannot_search+=1
                        
            log_f.write('\n')
            log_f.close()
    
    state_file.close()
    
    # save model weights
    torch.save(
                        {
                            'model': RL_model.state_dict(),
                            'optimizer_critic': RL_model.optimizer_critic.state_dict(),
                            'optimizer_actor': RL_model.optimizer_actor.state_dict()
                        },
                        os.path.join(opts.save_path, 'epoch-{}.pt'.format(epoch))
                    )
    print(losses_critic/batch_num,losses_ppo/batch_num)
       

def test_epoch(RL_model, Classify_model, test_data_loader, test_dtw, test_dataset, max_length_test, epoch, opts):
    
    Classify_model.eval()
    print('start test')
    
    softmax=torch.nn.Softmax(dim=1)
    num_envs=len(os.listdir(opts.train_data_path))
    batch_num=0
    num_all_batches=0
    all_steps=0
    cannot_search=0
    acc_batch=0
    with open(opts.log_file,'a+') as log_f:
        log_f.write('----------------------test at epoch {}-----------------------------'.format(epoch)+ '\n')
        log_f.close()
    for batch_id, (batch_input) in tqdm(enumerate(test_data_loader)):
        
        batch_label=batch_input[:,4].to(torch.int64)
        print(batch_label)
        batch_env_id=batch_input[:,1].to(torch.int64)      
        init_env_id=batch_env_id
        batch_env=batch_input[:,2:4]
        batch_length=batch_input[:,0].to(torch.int64)
        batch_index=batch_input[:,5].to(torch.int64)
        batch_trace=torch.reshape(batch_input[:,6:],[-1,max_length_test,opts.dim_input])
        log_predicted_alo=[]
        t=0
        current_ep_reward=0
        last_continue_index=torch.ones_like(batch_label).to(opts.device).bool()
        last_action=torch.zeros([opts.batch_size,num_envs]).to(opts.device)
        state_mask=torch.ones([opts.batch_size,opts.num_cluster]).to(opts.device)   
        RL_model.buffer.is_terminals.append(last_continue_index.to(device=opts.device))
        while t <= opts.max_deepth:
            if t==0:  
                state=(torch.ones([opts.batch_size,opts.num_cluster])/opts.num_cluster).to(opts.device)   
                action=RL_model.select_action(state,opts.batch_size,test_dtw,epoch,last_action,batch_env_id.to(opts.device),train=False) 
            else: 
                action=RL_model.select_action(state,opts.batch_size,test_dtw,epoch,last_action,init_action=None,train=False)   
            
            last_action=torch.zeros([opts.batch_size,num_envs]).to(opts.device)   
            last_action[torch.arange(0,opts.batch_size),action-1]=1
            
            
            next_data=test_dataset.get_trace(action,batch_index)   
            batch_label=next_data[:,4].to(torch.int64)  
            print(batch_label)
            batch_env_id=next_data[:,1]       
            batch_env=next_data[:,2:4]
            batch_length=next_data[:,0].to(torch.int64)
            batch_index=batch_input[:,5].to(torch.int64)
            batch_trace=torch.reshape(next_data[:,6:],[-1,max_length_test,opts.dim_input])
            
            with torch.no_grad():
                next_state = Classify_model(batch_trace,batch_length,batch_env,max_length_test)
                next_state=softmax(next_state).to(opts.device)
                
            
            flag_final,next_state,continue_index,state_mask=RL_model.is_final_test(next_state,last_continue_index,batch_label.to(opts.device),state_mask)   #is_final时基于未采取action之前的state判断的
            reward=RL_model.cal_reward_test(action,next_state,test_dtw,continue_index,t,opts.max_deepth)
            state=next_state   
            
            # saving reward and is_terminals
            RL_model.buffer.rewards.append(reward)
            if t!=opts.max_deepth and flag_final==False:
                RL_model.buffer.is_terminals.append(continue_index.to(device=opts.device))
            last_continue_index=continue_index
            current_ep_reward += reward
            t+=1
            batch_num=batch_id+1
            
            if flag_final:    
                break
        
        actions,states,probs,rewards,predicted=RL_model.test_output()
        
       
        if states.dim() == 2:
            states = states.unsqueeze(0) 
        elif states.dim() == 3 and state.dim() == 2:
            state = state.unsqueeze(0)  
        
        states=torch.cat((states,torch.ones([1,state.shape[1],state.shape[-1]]).to(opts.device)),dim=0)
        np.set_printoptions(suppress=True, precision=4)

        if actions.dim()==1:  
            actions=actions.unsqueeze(0)
            predicted=predicted.unsqueeze(0)
        
        with open(filepath,'a+') as log_state:
            for i in range(opts.batch_size):
                log_state.write('{}, {}, {},{}, {},{}\n'.format(
                    batch_label[i].item(),
                    init_env_id[i].item(),
                    actions[:,i][rewards[:,i]!=0].detach().cpu().numpy(),
                    states[1:,i,:][(rewards[:,i]!=0),:].detach().cpu().numpy(),
                    predicted[:,i][rewards[:,i]!=0].detach().cpu().numpy(),
                    len(rewards[:,i][rewards[:,i]!=0])                   
                ))
            
        with open(opts.log_file_test,'a+') as log_f:
            for i in range(opts.batch_size):
                log_f.write('{}, {}, {},{}, {},{}\n'.format(
                    batch_label[i].item(),
                    init_env_id[i].item(),
                    actions[:,i][rewards[:,i]!=0].detach().cpu().numpy(),
                    predicted[:,i][rewards[:,i]!=0].detach().cpu().numpy(),
                    rewards[:,i][rewards[:,i]!=0].detach().cpu().numpy(),
                    len(rewards[:,i][rewards[:,i]!=0])                   
                ))
                steps=len(rewards[:,i][rewards[:,i]!=0])
                all_steps+=steps
                num_all_batches+=1
                if len(rewards[:,i][rewards[:,i]!=0])==opts.max_deepth+1:
                    cannot_search+=1
                a=predicted[:,i][rewards[:,i]!=0][-1]
                b=batch_label[i].item()
                if a==b:
                    acc_batch+=1
                        
            log_f.write('\n')
            log_f.close()
        