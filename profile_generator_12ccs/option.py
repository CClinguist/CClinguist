import json
import os
import sys
import time
import argparse
import torch
times=time.strftime("%m%d-%H%M-%S")

print("[option.py loaded from]", os.path.abspath(__file__))
flag = 0 #test=0,train=1

def get_options(args=None):
    parser = argparse.ArgumentParser(
        description="RL model for adaptive TCP identification")
    
    #run
    parser.add_argument('--seed', type=int, default=3402, help='Random seed')
    parser.add_argument('--run_name', type=str, default='seach for env')
    parser.add_argument('--use_cuda', action='store_true', default=True, help='Use cuda')
    
    base_dir = os.path.dirname(os.path.abspath(__file__))

    # '''
    # for 12 known CCA in linux
    # '''

    if flag==0: 
        #  test
        parser.add_argument('--batch-size', type=int, default=36, help='Batch size')  
        parser.add_argument('--train_data_path', type=str, default=os.path.join(base_dir, 'data_12ccs', 'simulation_hk2jp_1201to1203'), 
                            help='Path to training data')
        parser.add_argument('--test_data_path', type=str, default=os.path.join(base_dir, 'data_12ccs', 'simulation_pk2hk_12_20'), 
                            help='Path to test data')
        parser.add_argument('--train', type=bool, default=False, help='train or test')
        parser.add_argument('--eval_only', action='store_true', default = True, help='Perform evaluation only') 
        parser.add_argument('--load_path', type=str, default = os.path.join(base_dir, 'model_rl', '1209-1916-29', 'epoch-7.pt'))
        parser.add_argument('--load_path_classifier', type=str, 
                            default=os.path.join(base_dir, 'models', 'best_model_accuracy.pth.tar'),
                            help='Path to load classifier model from')     
        parser.add_argument('--num_cluster', type=int, default=12 , help='Number of clusters')  
        parser.add_argument('--dtw_path', type=str, default= os.path.join(base_dir, 'dtw_matrix.txt'), help='Path to DTW data')
    

    if flag==1: 
        #  train
        parser.add_argument('--batch-size', type=int, default=36, help='Batch size')  
        parser.add_argument('--train_data_path', type=str, default=os.path.join(base_dir, 'data', 'simulation_hk2jp_1201to1203'), 
                            help='Path to training data')
        parser.add_argument('--test_data_path', type=str, default=os.path.join(base_dir, 'data', 'simulation_pk2hk_12_20'), 
                            help='Path to test data')
        parser.add_argument('--train', type=bool, default=True, help='train or test')
        parser.add_argument('--eval_only', action='store_true', default = False, help='Perform evaluation only') 
        parser.add_argument('--load_path', type=str, default = None)
        parser.add_argument('--load_path_classifier', type=str, 
                            default=os.path.join(base_dir, 'models', 'best_model_accuracy.pth.tar'),
                            help='Path to load classifier model from')     
        parser.add_argument('--num_cluster', type=int, default=12 , help='Number of clusters')    
        parser.add_argument('--dtw_path', type=str, default=os.path.join(base_dir, 'dtw_matrix.txt'), help='Path to DTW data')
    
    

    
    parser.add_argument('--resume', type=str, default=None, help='Resume training from checkpoint')
    parser.add_argument('--n_epochs', type=int, default=6, help='Number of training epochs')   
    parser.add_argument('--lr_critic', type=float, default=0.02, help='Learning rate for critic,last=0.002')
    parser.add_argument('--lr_actor', type=float, default=0.05, help='Learning rate for actor')
    parser.add_argument('--lr_classify', type=float, default=0.1, help='Learning rate for model,last value is 0.1')
    

    parser.add_argument('--dim_input', default=2, type=int)
    parser.add_argument('--dim_hidden', default=128, type=int)   
    parser.add_argument('--num_layers', default=8, type=int)    
    parser.add_argument('--dim_mlp', default=256, type=int)
    parser.add_argument('--num_heads', default=8, type=int)    
    parser.add_argument('--margin', default = 1.2905981663049348, type=float) 
    parser.add_argument('--weight_verf', default=0.00485668544506156, type=float) 
    parser.add_argument('--dropout_rate', default=0.2, type=float) 
    
        
    #RL train
    parser.add_argument('--entropy', type=float, default=0.01, help='Entropy coefficient')
    parser.add_argument('--update_timestep', type=int, default=1, help='Update PPO agent after every n timesteps')
    parser.add_argument('--log_freq', type=int, default=10, help='Logging frequency')
    parser.add_argument('--print_freq', type=int, default=100, help='Printing frequency')
    parser.add_argument('--save_model_freq', type=int, default=10, help='Model saving frequency')
    parser.add_argument('--checkpoint_path', type=str, default=None, help='Path to save model checkpoint')
    parser.add_argument('--error_threhold', type=float, default=0.6, help='Model checkpoint frequency')   
    parser.add_argument('--removed_prob', type=float, default=0.3, help='The probability of removed alos,need to be divided by num_cluster')
    
    #env_split
    parser.add_argument('--update_env_steps', type=int, default=5, help='Update env after every n epochs')
    
    #PPO init
    parser.add_argument('--gamma', type=float, default=0.6, help='Discount factor')
    parser.add_argument('--eps-clip', type=float, default=0.2, help='Clip parameter for PPO')
    parser.add_argument('--K-epochs', type=int, default=10, help='K epochs to update policy')
    parser.add_argument('--num-envs', type=int, default=20, help='Number of environments') 
    parser.add_argument('--device', type=str, default='cuda:0', help='Device to run the model on')
    parser.add_argument('--train_dtw', type=bool, default=True)
    parser.add_argument('--outdim_cnn', type=int, default=32, help='kernel number of cnn')
    parser.add_argument('--kernel_size', type=int, default=2)
    parser.add_argument('--stride', type=int, default=2)
    parser.add_argument('--actor_h1', type=int, default=128)
    parser.add_argument('--actor_h2', type=int, default=64)


    parser.add_argument('--dtw_weight', type=float, default=1, help='reward of dtw')
    parser.add_argument('--base_acc_reward', type=float, default=1, help='reward of find the accuracy opt alo')
    parser.add_argument('--base_obvious_reward', type=float, default=1, help='reward of obvious reward')   
    parser.add_argument('--rl_search_penalty', type=float, default=1, help='penalty of search steps')   
    
    parser.add_argument('--base_penalty', type=float, default=15, help='penalty of can not search reward')
    parser.add_argument('--direction_penalty', type=float, default=30, help='penalty of first direction')   
    parser.add_argument('--max_deepth', type=int, default=6, help='Maximum depth of exploration')  
    
    # reward term coefficients (to evaluate sensitivity)
    parser.add_argument('--coef_dtw', type=float, default=1.0, help='Coefficient for dtw reward')   
    parser.add_argument('--coef_base_penalty', type=float, default=1.0, help='Coefficient for base penalty at max depth')
    #w1,w2,w3
    parser.add_argument('--coef_acc_reward', type=float, default=1.0, help='Coefficient for correct classification reward')
    parser.add_argument('--coef_obvious_reward', type=float, default=1.0,help='Coefficient for obvious confidence reward')
    parser.add_argument('--coef_search_penalty', type=float, default=1.0, help='Coefficient for search step penalty')
    
    parser.add_argument('--coef_direction_penalty', type=float, default=1.0, help='Coefficient for wrong direction penalty')

    parser.add_argument('--indices_to_add',type=int,nargs='+',default = [0],help='Indices of input data to add (e.g., --indices_to_add 0 1 2)')
    
    #log
    log_output_dir = os.path.join(base_dir, 'results_12CCs_coef', 'log_output')
    train_log_dir = os.path.join(log_output_dir, 'train_log')
    test_log_dir = os.path.join(log_output_dir, 'test_log')
    model_dir = os.path.join(base_dir, 'model_rl')
    
    log_steps_dir = os.path.join(base_dir, 'results_12CCs_coef', 'log_steps')
    log_steps_train_dir = os.path.join(log_steps_dir, 'train_log')
    log_steps_test_dir = os.path.join(log_steps_dir, 'test_log')
    
    # log_acc_dir = os.path.join(base_dir, 'results_12CCs_coef', 'log_acc')
    # log_acc_train_dir = os.path.join(log_acc_dir, 'train_log')
    # log_acc_test_dir = os.path.join(log_acc_dir, 'test_log')

    result_dir = os.path.join(base_dir, 'results_12CCs_coef', 'analyze')

    for d in [train_log_dir, test_log_dir, model_dir,
              log_steps_train_dir, log_steps_test_dir, result_dir]:
        os.makedirs(d, exist_ok=True)
    
    parser.add_argument('--log_file', type=str, default=os.path.join(train_log_dir, f'train_{times}.txt'), help='Path to save results')
    parser.add_argument('--log_file_test', type=str, default=os.path.join(test_log_dir, f'test_{times}.txt'), help='Path to save results, redirected to a new directory')
    parser.add_argument('--save_path', type=str, default=model_dir, help='Path to save results')
    parser.add_argument('--log_file_steps', type=str, default=os.path.join(log_steps_train_dir, f'train_{times}.txt'), help='Path to save results')
    # parser.add_argument('--log_file_acc', type=str, default=os.path.join(log_acc_train_dir, f'train_{times}.txt'), help='Path to save results')
    parser.add_argument('--log_file_steps_test', type=str, default=os.path.join(log_steps_test_dir, f'test_{times}.txt'), help='Path to save results')
    # parser.add_argument('--log_file_acc_test', type=str, default=os.path.join(log_acc_test_dir, f'test_{times}.txt'), help='Path to save results')

    opts = parser.parse_args(args)
        
    opts.run_name = "{}".format(time.strftime("%m%d-%H%M-%S"))
    opts.save_path = os.path.join(
        opts.save_path,
        opts.run_name
    )
    
    if not os.path.exists(opts.save_path):
        os.makedirs(opts.save_path)
        
    
    return opts
