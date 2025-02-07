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
    parser.add_argument('--run_name', type=str, default='seach for env')
    parser.add_argument('--use_cuda', action='store_true', default=True, help='Use cuda')

    ##  test

    current_dir = os.path.dirname(os.path.abspath(__file__))
    train_data_path = os.path.join(current_dir, 'data', 'train_data')
    test_data_path = os.path.join(current_dir, 'data', 'test_data')
    load_path = os.path.join(current_dir, 'model_rl', 'best_epoch.pt')
    load_path_classifier = os.path.join(current_dir, 'models', 'good-sweep-1', 'best_model_accuracy.pth.tar')
    dtw_path = os.path.join(current_dir, 'dtw_matrix.txt')
    
    parser.add_argument('--train_data_path', type=str, default=train_data_path, help='Path to training data')
    parser.add_argument('--test_data_path', type=str, default=test_data_path, help='Path to test data')
    parser.add_argument('--batch-size', type=int, default=12, help='Batch size')  
    parser.add_argument('--train', type=bool, default=False, help='train or test')
    parser.add_argument('--eval_only', action='store_true', default = True, help='Perform evaluation only') 
    parser.add_argument('--load_path', type=str, default = load_path)
    parser.add_argument('--load_path_classifier', type=str, 
                        default=load_path_classifier,
                        help='Path to load classifier model from')     # no pcc and test set from other dataset
    parser.add_argument('--num_cluster', type=int, default=12 , help='Number of clusters')   
    parser.add_argument('--dtw_path', type=str, default=dtw_path, help='Path to DTW data')
    
    ##  train
    # parser.add_argument('--batch-size', type=int, default=12, help='Batch size') 
    # parser.add_argument('--train_data_path', type=str, default=train_data_path, 
    #                     help='Path to training data')
    # parser.add_argument('--test_data_path', type=str, default=test_data_path, 
    #                     help='Path to test data')
    # parser.add_argument('--train', type=bool, default=True, help='train or test')
    # parser.add_argument('--eval_only', action='store_true', default=False, help='Perform evaluation only') 
    # parser.add_argument('--load_path', type=str, default = None)
    # parser.add_argument('--load_path_classifier', type=str, 
    #                     default=load_path_classifier,
    #                     help='Path to load classifier model from')     # no pcc and test set from other dataset
    # parser.add_argument('--num_cluster', type=int, default=12 , help='Number of clusters')    #去掉dctcp
    # parser.add_argument('--dtw_path', type=str, default=None, help='Path to DTW data')
    
    

    parser.add_argument('--resume', type=str, default=None, help='Resume training from checkpoint')
    parser.add_argument('--n_epochs', type=int, default=20, help='Number of training epochs')   
    parser.add_argument('--lr_critic', type=float, default=0.02, help='Learning rate for critic,last=0.002')
    parser.add_argument('--lr_actor', type=float, default=0.05, help='Learning rate for actor')
    parser.add_argument('--lr_classify', type=float, default=0.1, help='Learning rate for model,last value is 0.1')
    
    parser.add_argument('--dim_input', default=2, type=int)
    parser.add_argument('--dim_hidden', default=128, type=int)   
    parser.add_argument('--num_layers', default=8, type=int)    
    parser.add_argument('--dim_mlp', default=256, type=int)
    parser.add_argument('--num_heads', default=8, type=int)   
    parser.add_argument('--margin', default = 2.45141, type=float) 
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
    
    #reward
    parser.add_argument('--dtw_weight', type=float, default=1, help='reward of dtw')
    parser.add_argument('--base_penalty', type=float, default=15, help='penalty of can not search reward')
    parser.add_argument('--base_acc_reward', type=float, default=13, help='reward of find the accuracy opt alo')
    parser.add_argument('--base_obvious_reward', type=float, default=20, help='reward of obvious reward')   
    parser.add_argument('--rl_search_penalty', type=float, default=4, help='penalty of search steps')  
    parser.add_argument('--direction_penalty', type=float, default=30, help='penalty of first direction')   
    parser.add_argument('--max_deepth', type=int, default=6, help='Maximum depth of exploration')  
    
    
    #log
    base_dir = os.path.dirname(os.path.abspath(__file__))

    log_output_dir = os.path.join(base_dir, 'results', 'log_output')
    train_log_dir = os.path.join(log_output_dir, 'train_log')
    test_log_dir = os.path.join(log_output_dir, 'test_log')
    model_dir = os.path.join(base_dir, 'model_rl')
    
    log_steps_dir = os.path.join(base_dir, 'results', 'log_steps')
    log_steps_train_dir = os.path.join(log_steps_dir, 'train_log')
    log_steps_test_dir = os.path.join(log_steps_dir, 'test_log')
    
    log_acc_dir = os.path.join(base_dir, 'results', 'log_acc')
    log_acc_train_dir = os.path.join(log_acc_dir, 'train_log')
    log_acc_test_dir = os.path.join(log_acc_dir, 'test_log')

    os.makedirs(train_log_dir, exist_ok=True)
    os.makedirs(test_log_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(log_steps_train_dir, exist_ok=True)
    os.makedirs(log_steps_test_dir, exist_ok=True)
    os.makedirs(log_acc_train_dir, exist_ok=True)
    os.makedirs(log_acc_test_dir, exist_ok=True)
    
    parser.add_argument('--log_file', type=str, default=os.path.join(train_log_dir, f'train_{times}.txt'), help='Path to save results')
    parser.add_argument('--log_file_test', type=str, default=os.path.join(test_log_dir, f'test_{times}.txt'), help='Path to save results, redirected to a new directory')
    parser.add_argument('--save_path', type=str, default=model_dir, help='Path to save results')
    parser.add_argument('--log_file_steps', type=str, default=os.path.join(log_steps_train_dir, f'train_{times}.txt'), help='Path to save results')
    parser.add_argument('--log_file_acc', type=str, default=os.path.join(log_acc_train_dir, f'train_{times}.txt'), help='Path to save results')
    parser.add_argument('--log_file_steps_test', type=str, default=os.path.join(log_steps_test_dir, f'test_{times}.txt'), help='Path to save results')
    parser.add_argument('--log_file_acc_test', type=str, default=os.path.join(log_acc_test_dir, f'test_{times}.txt'), help='Path to save results')
        
        
    
    
    

    opts = parser.parse_args(args)
        
    opts.run_name = "{}".format(time.strftime("%m%d-%H%M-%S"))
    opts.save_path = os.path.join(
        opts.save_path,
        opts.run_name
    )
    if not os.path.exists(opts.save_path):
        os.makedirs(opts.save_path)
        
    
    return opts
