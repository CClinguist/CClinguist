import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal
from torch.distributions import Categorical
import math
import numpy as np
import torch.nn.functional as F
torch.backends.cudnn.benchmark=False
# torch.backends.cudnn.benchmark = True
# CUDA_LAUNCH_BLOCKING=1

################################## set device ##################################
print("============================================================================================") 
if(torch.cuda.is_available()): 
    device = torch.device('cuda:0') 
    torch.cuda.empty_cache()
    print("Device set to : " + str(torch.cuda.get_device_name(device)))
else:
    print("Device set to : cpu")

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
        '''
        state: state as input,(dim equals to the num of alos)
        state:[batch,action,num_features,num_features]  
        cnn capture the feature of profiles, then use dnn to predict the action probability
        select action based on the action probability
        '''
        self.train=opts.train
        self.num_conv=math.ceil(math.log2(state_dim))-2
        self.out_cnn_dim=out_cnn_dim
        self.ss=nn.Conv2d(in_channels=1,out_channels=self.out_cnn_dim,kernel_size=opts.kernel_size,stride=opts.stride,device=device)
        
        #---------------------------feature extraction--------------------------------
        self.cnn_layer_actor=nn.Sequential(nn.Conv2d(in_channels=1,out_channels=self.out_cnn_dim,kernel_size=opts.kernel_size,stride=opts.stride,device=device))
        for i in range(self.num_conv):
            self.cnn_layer_actor.add_module('CNN_'+str(i),nn.Conv2d(in_channels=self.out_cnn_dim,out_channels=self.out_cnn_dim,kernel_size=opts.kernel_size,stride=opts.stride,device=device))
        

        #---------------------------------output the action probability--------------------------------
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

        '''
        state: [batch,num_alos,num_alos]
        action:[batch,env_nums] 
        dtw_matrix:[all_action_nums,num_alos,num_alos]
        '''
        
        #-------------------get dtw matrix for candidate action-------------------------------------------------------
        dtw_matrix=torch.tensor(dtw_matrix).to(device=device)
        dtw_cancidate_action=torch.index_select(dtw_matrix,dim=0,index=action.reshape(-1)-1)  #[batch*env_num, num_alos,num_alos]
        state=state.unsqueeze(-1).repeat(1,1,state.shape[-1])  #[batch,num_alo,num_alo]
        dtw_cancidate_action=dtw_cancidate_action.reshape(state.shape[0],-1,state.shape[-1],state.shape[-1])  # [batch,action,num_alos, num_alos]
        
        
        #----------------state*dtw use dtw matrix to help select the best action---------------------------------------
        state=state.unsqueeze(1).repeat(1,action.shape[-1],1,1)   #[batch,envs,num_alos,num_alos]
        state_dtw=torch.einsum('bijk,bikm->bijm',dtw_cancidate_action.float(),state.float())
        state_dtw=state_dtw.reshape([-1,1,state_dtw.shape[-2],state_dtw.shape[-1]]).to(dtype=torch.float32)
        

        #----------------calculate the action probability--------------------------------------------------------------
        action_probs = self.actor(self.cnn_layer_actor(state_dtw).reshape([state.shape[0],state.shape[1],-1])).squeeze(-1)
        action_logprob=torch.log(action_probs)    #[batch,num_envs]
        if init_action is not None:
            action_opt = init_action
        else:
            action_probs[last_action==1]=0.00001   # avoid loop
            # print(action_probs)
            dist = Categorical(action_probs)
            if train:  
                # train, sample action based on the distribution
                action_opt=dist.sample()+1
                
            else:
                # test, select action with the maximum probability
                action_opt=torch.argmax(action_probs,dim=1)+1
                # print(action_probs)
        action_logprob=action_logprob[torch.arange(0,action_opt.shape[0]),action_opt-1]   
        state_val = self.critic(self.cnn_layer_critic(state_dtw).reshape([state.shape[0],state.shape[1],-1])).squeeze(-1)   
        state_val=state_val.mean(dim=1)  
         
        return action_opt.detach(), action_logprob.detach(), state_val.detach(),action_probs.detach()
    
    def evaluate(self, state, action,opt_action,dtw_matrix):
        '''
        action:[batch,env_nums] -->[step,batch,env_num]
        opt_action:[step,batch]--->[step*batch] 
        state:[step,batch,num_alos]---->[step*batch,num_alos,num_alos]-->[step*batch,num_alos,num_alos]
        
        '''
        

        dtw_matrix=torch.Tensor(dtw_matrix).to(device=device)
        action=action[0,:].repeat(state.shape[0]*state.shape[1],1)  #[step*batch,env_nums]
        dtw_cancidate_action=torch.index_select(dtw_matrix,dim=0,index=action.reshape(-1)-1)  #[step*batch*env_num, num_alos,num_alos]
        dtw_cancidate_action=dtw_cancidate_action.reshape(action.shape[0],-1,state.shape[-1],state.shape[-1])  # [step*batch,num_envs,num_alos, num_alos]
        
        
        state=state.reshape([-1,state.shape[-1]]).unsqueeze(-1).repeat(1,1,state.shape[-1])   #[step*batch,num_alos,num_alos]
        state=state.unsqueeze(1).repeat(1,action.shape[-1],1,1)   #[step*batch,num_envs,num_alos,num_alos]
        state_dtw=torch.einsum('bijk,bikm->bijm',dtw_cancidate_action,state)
        state_dtw=state_dtw.reshape([-1,1,state_dtw.shape[-2],state_dtw.shape[-1]]).to(dtype=torch.float32)
        
        
        action_probs = self.actor(self.cnn_layer_actor(state_dtw).reshape([state.shape[0],state.shape[1],-1])).squeeze(-1)
        dist = Categorical(action_probs)
        dist_entropy = dist.entropy()
        action_logprob=torch.log(action_probs).reshape(-1,action_probs.shape[-1])    #[step*batch,num_envs]
        
        #-------------------extract the probability of the optimal action----------------------------------------------
        opt_logprob=action_logprob[torch.arange(0,action_logprob.shape[0]),opt_action.reshape([-1])-1]
        
        #-----------------critic calculate the state value----------------------------------------------------------------
        state_val = self.critic(self.cnn_layer_critic(state_dtw).reshape([state.shape[0],state.shape[1],-1])).squeeze(-1)
        state_val=state_val.sum(dim=-1)
        
        return opt_logprob, state_val, dist_entropy


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

        
        
    def select_action(self, state,batch_size,dtw_matrix,epoch,last_action,init_action=None,train=True):
        '''
        state: [batch, num_alo]  
        dtw_matrix:[env_nums,num_alos,num_alos]
        last_action: action taken in the last step
        '''
        num_envs=dtw_matrix.shape[0]
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
        '''
        when test, only use the confidence of the probability distribution to determine whether to terminate
        '''
        
        #----------------for the alo that has been excluded last time, the probability distribution is set to 0---------
        state=state*state_mask
        
        #-------------------normalize the non-zero items to make the sum equal to 1---------------------------------------
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
        '''
        when test, the reward is calculated based on the confidence of the probability distribution
        '''
        dtw_matrix=torch.Tensor(dtw_matrix).to(device=device)*self.dtw_weight
        dtw_action=torch.index_select(dtw_matrix,0,action.int()-1)  #[batch,num_alos,num_alos]
        reward=torch.einsum('bij,bjk->bik',state.unsqueeze(1),dtw_action) #[batch,1,3]*[batch,3,3]->[batch,1,3]
        reward=torch.mean(reward,-1) #[batch,1,1]
        reward=reward.squeeze(-1).squeeze(-1)  #dtw
        search_steps=continue_flag.int()*t*self.rl_search_penalty
        if t==max_deepth:
            penalty=continue_flag*self.base_penalty
        else:
            penalty=torch.zeros_like(reward)   
        obvious=(~continue_flag).int()*self.base_obvious_reward
        reward=-search_steps-penalty+obvious
        return reward
        
                     
    
    def is_final(self,state,continue_index_laststep,label,state_mask):
        '''
        use the difference between the predicted distribution and the true distribution to determine whether to terminate
        '''
        
        
        state=state*state_mask
        sum_nozero=torch.sum(state,dim=1)
        state=state/sum_nozero.unsqueeze(-1)
        # print(state.shape)
       
        
        #------------------deterimine whether to continue --------------------------------------------------------------
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
            # print(state_mask)
            
        return flag_final,state,continue_flag,alo_acc_flag,state_mask
    
       
      
    def cal_reward(self,action,state,dtw_matrix,continue_flag,t,max_deepth,alo_acc_flag):
        '''
        the big reward the better, ideal reward: confident(obvious)-search_steps+acc
        
        '''
        
        dtw_matrix=torch.Tensor(dtw_matrix).to(device=device)*self.dtw_weight
        dtw_action=torch.index_select(dtw_matrix,0,action.int()-1)  #[batch,num_alos,num_alos]
        reward=torch.einsum('bij,bjk->bik',state.unsqueeze(1),dtw_action) #[batch,1,3]*[batch,3,3]->[batch,1,3]
        reward=torch.mean(reward,-1) #[batch,1,1]
        reward=reward.squeeze(-1).squeeze(-1) 
        #penalty for the root of the action
        if t==0:
            penalty_direction=(~alo_acc_flag).int()*self.direction_penalty
        else:
            penalty_direction=torch.zeros_like(reward)
        search_steps=continue_flag.int()*t*self.rl_search_penalty
        #penalty for the search steps
        if t==max_deepth:
            penalty=continue_flag*self.base_penalty
        else:
            penalty=torch.zeros_like(reward)
        #reward for the accuracy
        acc=alo_acc_flag.int()*self.base_acc_reward   
        #reward for the confidence of the probability distribution
        obvious=(~continue_flag).int()*self.base_obvious_reward
        
        reward=-search_steps-penalty+acc+obvious-penalty_direction
        return reward
    
    
    def test_output(self):
       
        #----------------------reward of each step when path is terminated----------------------------------------------
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
        
        # convert list to tensor
        old_actions,old_logprobs,old_states,old_state_values,old_probs,old_is_termial,predicted=self.buffer.convert(self.batch_size)
        
        
        ##split valid data
        advantages = rewards - old_state_values  #[step,batch]
        old_logprobs_v=old_logprobs.reshape([-1,self.batch_size])[old_is_termial!=0]
        rewards_valid=rewards[old_is_termial!=0]
        self.buffer.clear()
        
        return old_actions,old_states,old_probs,rewards,predicted
        
        
             
    def update(self,dtw_matrix,epoch):
        '''
        ppo update with batch
        '''
        
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
        # if self.use_adv_norm:  # Trick:advantage normalization 
        #         advantages = ((advantages - advantages.mean()) / (advantages.std() + 1e-5))
        
        
        for _ in range(self.K_epochs):
            
            logprobs, state_values, dist_entropy = self.policy.evaluate(old_states.reshape([-1,old_states.shape[-2],old_states.shape[-1]]), self.candidate_action,old_actions,dtw_matrix)   #logprobs [step*batch,num_actions]  state_values[step*batch]
           
                
            #split valid data    
            state_values_v=state_values.reshape([-1,self.batch_size])[old_is_termial!=0]
            dist_entropy_v=dist_entropy.reshape([-1,self.batch_size])[old_is_termial!=0]
            logprobs_v=logprobs.reshape([-1,self.batch_size])[old_is_termial!=0]
            
            
            # importance sampling
            ratios = torch.exp(logprobs_v - old_logprobs_v.detach())  

            # Finding Surrogate Loss  
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1-self.eps_clip, 1+self.eps_clip) * advantages
            # loss_actor=torch.min(surr1, surr2)-self.entropy*dist_entropy_v  ##dist_entropy正则化，提高泛化性  [valid_batch]
            # loss_actor=loss_actor.mean()
            loss_actor=-torch.min(surr1, surr2).mean()-self.entropy*dist_entropy_v 
            # loss_actor=loss_actor.mean()
            # print(loss_actor.mean())
            
            
           
            # take gradient step
            self.optimizer_actor.zero_grad()
            loss_actor.mean().backward(retain_graph=True)
            self.optimizer_actor.step()
            
            
            
            loss_critic=self.MaeLoss((state_values_v).reshape([-1]), rewards_valid.reshape([-1]))  
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
        
    
        
        
       