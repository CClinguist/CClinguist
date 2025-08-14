import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal
from torch.distributions import Categorical
import math
import numpy as np
# from normalization import RewardScaling
import torch.nn.functional as F
torch.backends.cudnn.benchmark=False
# torch.backends.cudnn.benchmark = True
# CUDA_LAUNCH_BLOCKING=1

################################## set device ##################################
print("============================================================================================") 

device = torch.device('cuda:0')
print("============================================================================================")



################################## PPO Policy ##################################
class RolloutBuffer:
    
    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.state_values = []
        self.is_terminals = []
        self.action_probs = []
        self.predicted=[]
    
    def clear(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.state_values[:]
        del self.is_terminals[:]
        del self.action_probs [:]
        del self.predicted[:]
        
    def convert(self,batch_size):
        
        is_terminal=torch.squeeze(torch.stack(self.is_terminals, dim=0))  #[step,batch]
        if is_terminal.dim()==1:
                is_terminal=is_terminal.unsqueeze(0).reshape([-1,batch_size])
        old_actions=(torch.squeeze(torch.stack(self.actions, dim=0))).int()
        old_states = torch.squeeze(torch.stack(self.states, dim=0))  #[step,batch,num_alos,num_alos]
        old_logprobs = torch.squeeze(torch.stack(self.logprobs, dim=0)) #[step,batch] 
        old_state_values = torch.squeeze(torch.stack(self.state_values, dim=0)) #[step,batch]
        if old_state_values.dim()==1:
                old_state_values=old_state_values.unsqueeze(0).reshape([-1,batch_size])
        old_probs = torch.squeeze(torch.stack(self.action_probs, dim=0))
        predicted=torch.squeeze(torch.stack(self.predicted, dim=0))
        return old_actions,old_logprobs,old_states,old_state_values,old_probs,is_terminal,predicted


class ActorCritic(nn.Module):
    def __init__(self, state_dim,out_cnn_dim,opts):
        super(ActorCritic, self).__init__()

        self.train=opts.train
        self.num_conv=math.ceil(math.log2(state_dim))-2
        self.out_cnn_dim=out_cnn_dim
        self.ss=nn.Conv2d(in_channels=1,out_channels=self.out_cnn_dim,kernel_size=opts.kernel_size,stride=opts.stride,device=device)
        

        self.cnn_layer_actor=nn.Sequential(nn.Conv2d(in_channels=1,out_channels=self.out_cnn_dim,kernel_size=opts.kernel_size,stride=opts.stride,device=device))
        for i in range(self.num_conv):
            self.cnn_layer_actor.add_module('CNN_'+str(i),nn.Conv2d(in_channels=self.out_cnn_dim,out_channels=self.out_cnn_dim,kernel_size=opts.kernel_size,stride=opts.stride,device=device))
        

        self.actor= nn.Sequential(
                        nn.Linear(out_cnn_dim, opts.actor_h1),
                        nn.LeakyReLU(),
                        nn.Linear(opts.actor_h1, opts.actor_h1),
                        nn.LeakyReLU(),
                        nn.Linear(opts.actor_h1, opts.actor_h2),
                        nn.LeakyReLU(),
                        nn.Linear(opts.actor_h2,1),
                        torch.nn.Softmax(dim=1)   #[batch,action,1]
                    )
        
        # critic
        self.cnn_layer_critic=nn.Sequential(nn.Conv2d(in_channels=1,out_channels=self.out_cnn_dim,kernel_size=opts.kernel_size,stride=opts.stride,device=device))
        for i in range(self.num_conv):
            self.cnn_layer_critic.add_module('criticcnn_'+str(i),nn.Conv2d(in_channels=self.out_cnn_dim,out_channels=self.out_cnn_dim,kernel_size=opts.kernel_size,stride=opts.stride,device=device))
        self.critic= nn.Sequential(
                        nn.Linear(out_cnn_dim, opts.actor_h1),
                        nn.LeakyReLU(),
                        nn.Linear(opts.actor_h1, opts.actor_h1),
                        nn.LeakyReLU(),
                        nn.Linear(opts.actor_h1, opts.actor_h2),
                        nn.LeakyReLU(),
                        nn.Linear(opts.actor_h2,1),   #[batch,1]
                        nn.LeakyReLU(),
                        nn.Linear(1,1)
                    ) 
        
    

    def forward(self):
        raise NotImplementedError
    
    def act(self, state,action,dtw_matrix,last_action,init_action,epoch,train=True):

        dtw_matrix=torch.tensor(dtw_matrix).to(device=device)

        
        dtw_cancidate_action=torch.index_select(dtw_matrix,dim=0,index=action.reshape(-1)-1)  #[batch*env_num, num_alos,num_alos]
        state=state.unsqueeze(-1).repeat(1,1,state.shape[-1])  #[batch,num_alo,num_alo]
        
        
        dtw_cancidate_action=dtw_cancidate_action.reshape(state.shape[0],-1,state.shape[-1],state.shape[-1])  # [batch,action,num_alos, num_alos]

        state=state.unsqueeze(1).repeat(1,action.shape[-1],1,1)   #[batch,envs,num_alos,num_alos]
        state_dtw=torch.einsum('bijk,bikm->bijm',dtw_cancidate_action.float(),state.float())    
        state_dtw=state_dtw.reshape([-1,1,state_dtw.shape[-2],state_dtw.shape[-1]]).to(dtype=torch.float32)

        action_probs = self.actor(self.cnn_layer_actor(state_dtw).reshape([state.shape[0],state.shape[1],-1])).squeeze(-1)
        action_logprob=torch.log(action_probs)    #[batch,num_envs]
        if init_action is not None:
            action_opt = init_action
        else:
            
            print(f"before reshape,last_action shape: {last_action.shape}")
            last_action = last_action[:, :43]
            print(f"last_action shape: {last_action.shape}")
            print(f"action_probs shape: {action_probs.shape}")

            action_probs[last_action==1]=0.00001   
            # print(action_probs)
            dist = Categorical(action_probs)
            if train:    
                action_opt=dist.sample()+1
                # if any(last_action[torch.arange(0,action_opt.shape[0]),action_opt-1])==1:
                #     index=torch.where(last_action[torch.arange(0,action_opt.shape[0]),action_opt-1]==1)[0]
                #     action_opt[index]=action_top2[1,index]
            else:
                
                action_opt=torch.argmax(action_probs,dim=1)+1

        action_logprob=action_logprob[torch.arange(0,action_opt.shape[0]),action_opt-1]   #[batch]
        # a=self.cnn_layer_critic(state_dtw)
        state_val = self.critic(self.cnn_layer_critic(state_dtw).reshape([state.shape[0],state.shape[1],-1])).squeeze(-1)   
        state_val=state_val.mean(dim=1)  
         

        return action_opt.detach(), action_logprob.detach(), state_val.detach(),action_probs.detach()
    
    def evaluate(self, state, action, opt_action, dtw_matrix):
        try:

            dtw_matrix = torch.Tensor(dtw_matrix).to(device=device)
            action = action[0, :].repeat(state.shape[0] * state.shape[1], 1)
            
            dtw_cancidate_action = torch.index_select(dtw_matrix, dim=0, index=action.reshape(-1) - 1)
            dtw_cancidate_action = dtw_cancidate_action.reshape(action.shape[0], -1, state.shape[-1], state.shape[-1])
            

            if torch.isnan(dtw_cancidate_action).any():
                print("Error: dtw_cancidate_action contains NaNs")
            

            state = state.reshape([-1, state.shape[-1]]).unsqueeze(-1).repeat(1, 1, state.shape[-1])
            state = state.unsqueeze(1).repeat(1, action.shape[-1], 1, 1)
            state_dtw = torch.einsum('bijk,bikm->bijm', dtw_cancidate_action, state)
            
            if torch.isnan(state_dtw).any():
                print("Error: state_dtw contains NaNs")
            
            state_dtw = state_dtw.reshape([-1, 1, state_dtw.shape[-2], state_dtw.shape[-1]]).to(dtype=torch.float32)
            

            cnn_output = self.cnn_layer_actor(state_dtw).reshape([state.shape[0], state.shape[1], -1])
            
            if torch.isnan(cnn_output).any():
                print("Error: CNN output contains NaNs")
            
            action_probs = self.actor(cnn_output).squeeze(-1)
            

            if torch.isnan(action_probs).any():
                print("Error: action_probs contains NaNs")
            
            if not torch.all((action_probs >= 0) & (action_probs <= 1)):
                print("Warning: action_probs values are out of [0, 1] range")
            
            dist = Categorical(action_probs)
            dist_entropy = dist.entropy()
            

            action_logprob = torch.log(action_probs).reshape(-1, action_probs.shape[-1])
            opt_logprob = action_logprob[torch.arange(0, action_logprob.shape[0]), opt_action.reshape([-1]) - 1]
            

            cnn_critic_output = self.cnn_layer_critic(state_dtw).reshape([state.shape[0], state.shape[1], -1])
            
            if torch.isnan(cnn_critic_output).any():
                print("Error: CNN critic output contains NaNs")
            
            state_val = self.critic(cnn_critic_output).squeeze(-1).sum(dim=-1)
            
            return opt_logprob, state_val, dist_entropy

        except Exception as e:
            print(f"An error occurred: {e}")
            raise
    

class PPO(nn.Module):
    def __init__(self, opts):
        super(PPO, self).__init__()
        self.gamma = opts.gamma   
        self.batch_size=opts.batch_size
        self.eps_clip = opts.eps_clip
        self.K_epochs = opts.K_epochs
        self.base_penalty=opts.base_penalty
        self.base_acc_reward=opts.base_acc_reward
        self.buffer = RolloutBuffer()
        self.train_dtw=opts.train_dtw
        self.num_envs=opts.num_envs
        self.policy = ActorCritic(opts.num_cluster,opts.outdim_cnn,opts).to(opts.device)

        self.opts=opts
        self.MaeLoss=torch.nn.L1Loss()
        self.max_epochs=opts.n_epochs
        self.use_lr_decay=True
        self.use_adv_norm=True
        self.num_cluster=opts.num_cluster
        self.device=opts.device
        self.rl_search_penalty=opts.rl_search_penalty
        self.base_obvious_reward=opts.base_obvious_reward
        self.direction_penalty=opts.direction_penalty
        self.optimizer_actor= torch.optim.Adagrad(self.policy.actor.parameters(), lr=opts.lr_actor,eps=1e-5)
        self.optimizer_critic= torch.optim.Adagrad(self.policy.critic.parameters(), lr=opts.lr_critic,eps=1e-5)
        self.entropy=opts.entropy
        self.dtw_weight=opts.dtw_weight
        self.error_threhold=opts.error_threhold
        self.removed_prob = opts.removed_prob

        self.coef_dtw = opts.coef_dtw
        self.coef_search_penalty = opts.coef_search_penalty
        self.coef_base_penalty = opts.coef_base_penalty
        self.coef_acc_reward = opts.coef_acc_reward
        self.coef_obvious_reward = opts.coef_obvious_reward
        self.coef_direction_penalty = opts.coef_direction_penalty

        
        
    def select_action(self, state,batch_size,dtw_matrix,epoch,last_action,init_action=None,train=True):

        
        num_envs=dtw_matrix.shape[0]
        print("num_envs in select_action:",num_envs)
        self.batch_size=batch_size
        self.candidate_action=torch.arange(1,num_envs+1).unsqueeze(0).repeat(batch_size,1).to(self.opts.device) 
        with torch.no_grad(): 
            action, action_logprob, state_val ,action_probs= self.policy.act(state,self.candidate_action,dtw_matrix,last_action,init_action,epoch,train)
        
        
        self.buffer.states.append(state)
        self.buffer.actions.append(action.int())
        self.buffer.logprobs.append(action_logprob)
        self.buffer.state_values.append(state_val)
        self.buffer.action_probs.append(action_probs)

        return action
    
    def is_final_test(self,state,continue_index_laststep,label,state_mask):

        

        state=state*state_mask

        sum_nozero=torch.sum(state,dim=1)
        state=state/sum_nozero.unsqueeze(-1)
        continue_flag=torch.max(state,dim=1).values<(1-self.error_threhold/2)
        
        predicted=torch.argmax(state,dim=1)
        self.buffer.predicted.append(predicted)
        
        continue_flag=continue_index_laststep&continue_flag  
        index_continue=torch.where(continue_flag==True)
        num_continue=index_continue[0].size(0)
        if num_continue==0:
            flag_final=True
        else:
            flag_final=False

            loss_continue=state[index_continue[0],:]
   
            black_list=torch.where(loss_continue<=self.removed_prob/self.num_cluster)        
             

            state_continue=state[index_continue[0],:]
            state_continue[black_list[0],black_list[1]]=0

            sum_nozero_state=torch.sum(state_continue,dim=1)
            state_continue=state_continue/sum_nozero_state.unsqueeze(-1).repeat(1,self.num_cluster)
            state[index_continue[0],:]=state_continue

            state_mask=(state!=0).int()

         
        return flag_final,state,continue_flag,state_mask
    
    def cal_reward_test(self,action,state,dtw_matrix,continue_flag,t,max_deepth):

        dtw_matrix=torch.Tensor(dtw_matrix).to(device=device)*self.dtw_weight
        dtw_action=torch.index_select(dtw_matrix,0,action.int()-1)  #[batch,num_alos,num_alos]
        reward=torch.einsum('bij,bjk->bik',state.unsqueeze(1),dtw_action) #[batch,1,3]*[batch,3,3]->[batch,1,3]
        reward=torch.mean(reward,-1) #[batch,1,1]
        reward=reward.squeeze(-1).squeeze(-1)  
  
        search_steps=continue_flag.int()*t*self.rl_search_penalty

        if t==max_deepth:
            penalty=continue_flag*self.base_penalty
        else:
            penalty=torch.zeros_like(reward)   

        obvious=(~continue_flag).int()*self.base_obvious_reward
        
        # reward=reward-search_steps-penalty+obvious
        
        reward = (
            self.coef_dtw * reward
            - self.coef_search_penalty * search_steps   
            - self.coef_base_penalty * penalty  
            + self.coef_obvious_reward * obvious    
        )
        return reward
            
    
    def is_final(self,state,continue_index_laststep,label,state_mask):

        state=state*state_mask

        sum_nozero=torch.sum(state,dim=1)
        state=state/sum_nozero.unsqueeze(-1)


        label_one_hot=F.one_hot(label,num_classes=self.num_cluster).float()
        loss_pre=torch.abs(label_one_hot-state)
        continue_flag=loss_pre.sum(dim=1)>self.error_threhold    

        predicted=torch.argmax(state,dim=1)
        self.buffer.predicted.append(predicted)
        # print(predicted)  
        alo_acc_flag=predicted==label
        if any(continue_flag==0):   
            continue_flag=continue_flag|(~alo_acc_flag)
            
        continue_flag=continue_index_laststep&continue_flag 
        index_continue=torch.where(continue_flag==True)
        num_continue=index_continue[0].size(0)
        if num_continue==0:
            flag_final=True
        else:
            flag_final=False   

            loss_continue=loss_pre[index_continue[0],:]

            black_list=torch.where(loss_continue<=((self.removed_prob)/self.num_cluster))        

            state_continue=state[index_continue[0],:]
            state_continue[black_list[0],black_list[1]]=0
            

            sum_nozero_state=torch.sum(state_continue,dim=1)
            state_continue=state_continue/sum_nozero_state.unsqueeze(-1).repeat(1,self.num_cluster)
            state[index_continue[0],:]=state_continue

            state_mask=(state!=0).int()

            
        return flag_final,state,continue_flag,alo_acc_flag,state_mask
 
      
    def cal_reward(self,action,state,dtw_matrix,continue_flag,t,max_deepth,alo_acc_flag):

        dtw_matrix=torch.Tensor(dtw_matrix).to(device=device)*self.dtw_weight

        dtw_action=torch.index_select(dtw_matrix,0,action.int()-1)  #[batch,num_alos,num_alos]
        reward=torch.einsum('bij,bjk->bik',state.unsqueeze(1),dtw_action) #[batch,1,3]*[batch,3,3]->[batch,1,3]
        reward=torch.mean(reward,-1) #[batch,1,1]
        reward=reward.squeeze(-1).squeeze(-1) 

        if t==0:
            penalty_direction=(~alo_acc_flag).int()*self.direction_penalty
        else:
            penalty_direction=torch.zeros_like(reward)

        search_steps=continue_flag.int()*t*self.rl_search_penalty

        if t==max_deepth:
            penalty=continue_flag*self.base_penalty
        else:
            penalty=torch.zeros_like(reward)

        acc=alo_acc_flag.int()*self.base_acc_reward   

        obvious=(~continue_flag).int()*self.base_obvious_reward
        
        # reward=reward-search_steps-penalty+acc+obvious-penalty_direction

        reward = (
            self.coef_dtw * reward
            - self.coef_search_penalty * search_steps   
            - self.coef_base_penalty * penalty  
            + self.coef_acc_reward * acc    
            + self.coef_obvious_reward * obvious    
            - self.coef_direction_penalty * penalty_direction  
        )    
        return reward
    
    
    def test_output(self):

        rewards = None
        discounted_reward = torch.zeros(self.batch_size).to(device=device)
        for reward, is_terminal in zip(reversed(self.buffer.rewards), reversed(self.buffer.is_terminals)):
            
            discounted_reward = torch.mul(discounted_reward,is_terminal)  
            reward=torch.mul(reward,is_terminal)  
            discounted_reward = reward + (self.gamma * discounted_reward)
            if rewards is None:
                rewards=discounted_reward.unsqueeze(0)
            else:
                rewards=torch.cat([discounted_reward.unsqueeze(0),rewards],0)
            

        rewards = rewards.to(self.opts.device)  #[steps,batch]

        old_actions,old_logprobs,old_states,old_state_values,old_probs,old_is_termial,predicted=self.buffer.convert(self.batch_size)

        

        advantages = rewards - old_state_values  #[step,batch]
        old_logprobs_v=old_logprobs.reshape([-1,self.batch_size])[old_is_termial!=0]
        rewards_valid=rewards[old_is_termial!=0]
        self.buffer.clear()
        
        return old_actions,old_states,old_probs,rewards,predicted
        
        
             
    def update(self,dtw_matrix,epoch):

        
        rewards = None        
        
        discounted_reward = torch.zeros(self.batch_size).to(device=device)

        
        for reward, is_terminal in zip(reversed(self.buffer.rewards), reversed(self.buffer.is_terminals)):
            
            discounted_reward = torch.mul(discounted_reward,is_terminal)  
            reward=torch.mul(reward,is_terminal)   
            discounted_reward = reward + (self.gamma * discounted_reward)
            if rewards is None:
                rewards=discounted_reward.unsqueeze(0)
            else:
                rewards=torch.cat([discounted_reward.unsqueeze(0),rewards],0)
            

        rewards = rewards.to(self.opts.device)  #[steps,batch]
        

        old_actions,old_logprobs,old_states,old_state_values,old_probs,old_is_termial,predicted=self.buffer.convert(self.batch_size)

        
        
        ##split valid data
        advantages = rewards - old_state_values  #[step,batch]
        old_logprobs_v=old_logprobs.reshape([-1,self.batch_size])[old_is_termial!=0]
        rewards_valid=rewards[old_is_termial!=0]
        advantages=advantages.reshape([-1,self.batch_size])[old_is_termial!=0]

        
        
        for _ in range(self.K_epochs):
            
            logprobs, state_values, dist_entropy = self.policy.evaluate(old_states.reshape([-1,old_states.shape[-2],old_states.shape[-1]]), self.candidate_action,old_actions,dtw_matrix)   #logprobs [step*batch,num_actions]  state_values[step*batch]
           
                
            #split valid data    
            state_values_v=state_values.reshape([-1,self.batch_size])[old_is_termial!=0]
            dist_entropy_v=dist_entropy.reshape([-1,self.batch_size])[old_is_termial!=0]
            logprobs_v=logprobs.reshape([-1,self.batch_size])[old_is_termial!=0]
            

            ratios = torch.exp(logprobs_v - old_logprobs_v.detach())  

            # Finding Surrogate Loss  
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1-self.eps_clip, 1+self.eps_clip) * advantages

            loss_actor=-torch.min(surr1, surr2).mean()-self.entropy*dist_entropy_v 
            
    
            # take gradient step
            self.optimizer_actor.zero_grad()
            loss_actor.mean().backward(retain_graph=True)
            self.optimizer_actor.step()
            
            
            
            loss_critic=self.MaeLoss((state_values_v).reshape([-1]), rewards_valid.reshape([-1]))  
            # print(loss_critic)
            self.optimizer_critic.zero_grad()
            loss_critic.backward(retain_graph=True)
            self.optimizer_critic.step()
            
        if self.use_lr_decay:  # Trick:learning rate Decay
            lr_a_now = self.optimizer_actor.param_groups[0]['lr'] * (1 - epoch / self.max_epochs)
            lr_c_now = self.optimizer_critic.param_groups[0]['lr'] * (1 - epoch / self.max_epochs)
            for p in self.optimizer_actor.param_groups:
                p['lr'] = lr_a_now
            for p in self.optimizer_critic.param_groups:
                p['lr'] = lr_c_now
        
        # clear buffer  
        self.buffer.clear()
        # print(rewards)
        return loss_actor.mean(),loss_critic,old_actions,old_logprobs,old_states,old_state_values,old_probs,rewards,predicted
    

    
    def save(self, checkpoint_path):
        torch.save(self.policy_old.state_dict(), checkpoint_path)
   
    def load(self, checkpoint_path):
        self.policy_old.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))
        self.policy.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))
 