import os
import time
import argparse
import torch
times=time.strftime("%m%d-%H%M-%S")



def get_options(args=None):
    parser = argparse.ArgumentParser(
        description="RL model for adaptive TCP identification")

    #run
    parser.add_argument('--seed', type=int, default=3402, help='Random seed')
    parser.add_argument('--train', type=bool, default=False, help='train or test')
    parser.add_argument('--run_name', type=str, default='seach for env')
    parser.add_argument('--use_cuda', action='store_true', default=True, help='Use cuda')
    parser.add_argument('--load_path', type=str, default='Profile_generator/checkpoints/rl-epoch-5.pt')


    parser.add_argument('--resume', type=str, default=None, help='Resume training from checkpoint')
    
    parser.add_argument('--load_path_classifier', type=str, 
                        default='Profile_generator/checkpoints/best_model_accuracy.pth.tar',
                        help='Path to load classifier model from')
    

    parser.add_argument('--n_epochs', type=int, default=1000, help='Number of training epochs')   
    parser.add_argument('--eval_only', action='store_true', default=True, help='Perform evaluation only')
    parser.add_argument('--lr_critic', type=float, default=0.002, help='Learning rate for critic')
    parser.add_argument('--lr_actor', type=float, default=0.01, help='Learning rate for actor')
    parser.add_argument('--lr_classify', type=float, default=0.1, help='Learning rate for model')
    
    parser.add_argument('--train_data_path', type=str, default='DataCollection/data_example/60-300_200-1000', 
                         help='Path to training data')
    
    parser.add_argument('--test_data_path', type=str, default='DataCollection/data_example/60-300_200-1000', 
                         help='Path to testing data')
       
    
    parser.add_argument('--dtw_path', type=str, default=None, help='Path to DTW data')
    parser.add_argument('--num_cluster', type=int, default=12 , help='Number of clusters')
    
    
    parser.add_argument('--dim_input', default=2, type=int)
    parser.add_argument('--dim_hidden', default=256, type=int)   #256
    parser.add_argument('--num_layers', default=8, type=int)    #8
    parser.add_argument('--dim_mlp', default=256, type=int)
    parser.add_argument('--num_heads', default=16, type=int)    #16
        
    #RL train
    parser.add_argument('--entropy', type=float, default=0.01, help='Entropy coefficient')
    parser.add_argument('--update_timestep', type=int, default=1, help='Update PPO agent after every n timesteps')
    parser.add_argument('--log_freq', type=int, default=10, help='Logging frequency')
    parser.add_argument('--print_freq', type=int, default=100, help='Printing frequency')
    parser.add_argument('--save_model_freq', type=int, default=10, help='Model saving frequency')
    parser.add_argument('--checkpoint_path', type=str, default=None, help='Path to save model checkpoint')
    parser.add_argument('--error_threhold', type=float, default=0.2, help='Model checkpoint frequency')   
    parser.add_argument('--removed_prob', type=float, default=0.3, help='The probability of removed alos,need to be divided by num_cluster')
    
    #env_split
    parser.add_argument('--update_env_steps', type=int, default=5, help='Update env after every n epochs')
    
    #PPO init
    parser.add_argument('--gamma', type=float, default=0.6, help='Discount factor')
    parser.add_argument('--eps-clip', type=float, default=0.2, help='Clip parameter for PPO')
    parser.add_argument('--K-epochs', type=int, default=10, help='K epochs to update policy')
    parser.add_argument('--num-envs', type=int, default=20, help='Number of environments')
    parser.add_argument('--batch-size', type=int, default=12, help='Batch size')    
    parser.add_argument('--device', type=str, default='cuda', help='Device to run the model on')
    parser.add_argument('--train_dtw', type=bool, default=True)
    parser.add_argument('--outdim_cnn', type=int, default=32, help='kernel number of cnn')
    parser.add_argument('--kernel_size', type=int, default=2)
    parser.add_argument('--stride', type=int, default=2)
    parser.add_argument('--actor_h1', type=int, default=128)
    parser.add_argument('--actor_h2', type=int, default=64)
    
    #reward
    parser.add_argument('--dtw_weight', type=float, default=1, help='reward of dtw')
    parser.add_argument('--base_penalty', type=float, default=15, help='penalty of can not search reward')
    parser.add_argument('--base_acc_reward', type=float, default=10, help='reward of find the accuracy opt alo')
    parser.add_argument('--base_obvious_reward', type=float, default=20, help='reward of obvious reward')   
    parser.add_argument('--rl_search_penalty', type=float, default=5, help='penalty of search steps') 
    parser.add_argument('--direction_penalty', type=float, default=30, help='penalty of first direction') 
    parser.add_argument('--max_deepth', type=int, default=6, help='Maximum depth of exploration')
    
    
    #log
    parser.add_argument('--log_dir_train', type=str, default='Profile_generator/results/log_output/train_log', help='Path to save logs')
    parser.add_argument('--log_dir_test', type=str, default='Profile_generator/results/log_output/test_log', help='Path to save logs')
    parser.add_argument('--log_file', type=str, default='Profile_generator/results/log_output/train_log/train_'+times+'.txt', help='Path to save results')
    parser.add_argument('--log_file_test', type=str, default='Profile_generator/results/log_output/test_log/test_'+times+'.txt', help='Path to save results,redirected to a new directory')
    parser.add_argument('--save_path', type=str, default='Profile_generator/model_rl_save/', help='Path to save results')
    parser.add_argument('--log_file_steps', type=str, default='Profile_generator/results/log_steps/train_log/train_'+times+'.txt', help='Path to save results')
    parser.add_argument('--log_file_acc', type=str, default='Profile_generator/results/log_acc/train_log/train_'+times+'.txt', help='Path to save results')
    
    
    opts = parser.parse_args(args)
    if not os.path.exists(opts.log_dir_train):
        os.makedirs(opts.log_dir_train)
        
    if not os.path.exists(opts.log_dir_test):
        os.makedirs(opts.log_dir_test)
    
        
        
    opts.use_cuda = torch.cuda.is_available()
    opts.run_name = "{}".format(time.strftime("%m%d-%H%M-%S"))
    opts.save_path = os.path.join(
        opts.save_path,
        opts.run_name
    )
    if not os.path.exists(opts.save_path):
        os.makedirs(opts.save_path)
        
    
    return opts
