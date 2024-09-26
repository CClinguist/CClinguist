#!/usr/bin/env python

import os
import json
import pprint as pp
import time
import torch
import torch.optim as optim
from problem import Dataset
from option import get_options
from train_RL import train_epoch,test_epoch
import numpy as np
from classifier_structure import Classifier
from RL import PPO 
import warnings
warnings.filterwarnings('ignore')


def run(opts):
    
    pp.pprint(vars(opts))
    torch.manual_seed(opts.seed)

    # -------------------------------------Load data from load_path---------------------------------------------------
    load_data = {}
    assert opts.load_path is None or opts.resume is None, "Only one of load path and resume can be given"
    load_path = opts.load_path if opts.load_path is not None else opts.resume
    

    # Initialize model
    classify_model=Classifier(opts.num_cluster,opts.dim_hidden,opts.num_layers,opts.num_heads,opts.dim_input)
    classify_model.eval()
    RL_model=PPO(opts)  

    #dataset
    train_dataset = Dataset(opts.batch_size,opts.train_data_path,opts.dtw_path)
    train_data_loader,max_length_train=train_dataset.make_dataset_arrivetime(opts.batch_size,train = True,train_data_num=5)  
    train_dtw=train_dataset.load_action_dtw()
    
    
    #-----------------------------------------Load optimizer state-------------------------------------------
    if load_path is not None:
        print('  [*] Loading model from {}'.format(load_path))
        load_data = torch.load(load_path, map_location=lambda storage, loc: storage) 
        model_ = RL_model
        model_.load_state_dict(torch.load(load_path)['model'])
        if 'optimizer_critic' in load_data:
            RL_model.optimizer_critic.load_state_dict(load_data['optimizer_critic'])
            for state in RL_model.optimizer_critic.state.values():
                for k, v in state.items():
                    if torch.is_tensor(v):
                        state[k] = v.to(opts.device)
        if 'optimizer_actor' in load_data:
            RL_model.optimizer_actor.load_state_dict(load_data['optimizer_actor'])
            for state in RL_model.optimizer_actor.state.values():
                for k, v in state.items():
                    if torch.is_tensor(v):
                        state[k] = v.to(opts.device)         
                        
    
    load_path_classifier = opts.load_path_classifier

    if load_path_classifier is not None:
        print('  [*] Loading classifier from {}'.format(load_path_classifier))
        checkpoint=torch.load(load_path_classifier)
        classify_model.load_state_dict(checkpoint['state_dict'])
    
    
    #---------------------------------train and eval-----------------------------------------------------------
    
    if opts.eval_only:
        test_dataset = Dataset(opts.batch_size,opts.test_data_path,opts.dtw_path)
        test_data_loader,max_length_test=test_dataset.make_dataset_arrivetime(opts.batch_size,train = False,train_data_num=5)
        test_dtw=test_dataset.load_action_dtw()
        test_epoch(
            RL_model,
            classify_model,
            test_data_loader,
            test_dtw,
            test_dataset,
            max_length_test,
            0,
            opts
        )
    else:
        with open(opts.log_file,'a+') as log_f:
            log_f.write('alo_label, init_profile, actions, alo_predicted, reward_actions, steps_search' + '\n')
            log_f.close()
        for epoch in range(1,opts.n_epochs):
            
            start_time=time.time()
            
            train_epoch(
            RL_model,
            classify_model,
            train_data_loader,
            train_dtw,
            train_dataset,
            max_length_train,
            epoch,
            opts)
                

            epoch_duration = time.time() - start_time
            print("Finished epoch {}, took {} s".format(epoch, time.strftime('%H:%M:%S', time.gmtime(epoch_duration))))


if __name__ == "__main__":
    opts = get_options()

    run(opts)

    