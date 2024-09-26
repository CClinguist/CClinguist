import torch
import torch.nn as nn
import torch.utils.data
import pandas as pd
import numpy as np
import os
import re
from sklearn.preprocessing import MinMaxScaler
from torch.nn.utils.rnn import pad_sequence
from dtw import dtw
import matplotlib.pyplot as plt
import torchmetrics
from collections import Counter
from tqdm import tqdm

class queueOccuDataset_build(nn.Module): 
    def __init__(self, batch_size):
        super(queueOccuDataset_build, self).__init__()
        self.batch_size = batch_size

    def get_merged_df(self, host_path, capture_path):
        host_df = pd.read_csv(host_path, delim_whitespace=True, names=['Time', 'Length', 'DataLength', 'SeqNum'], index_col=False)
        
        
        host_df[['Length', 'DataLength']] = host_df[['Length', 'DataLength']].astype(int)
        if (host_df['DataLength'] > 1000).any():
            first_data_pkt_idx = host_df[host_df['DataLength'] > 1000].index[0]
        else:
            first_data_pkt_idx = None
            print(f"No data packets with DataLength > 1000 found in {host_path}.")
            return pd.DataFrame(), None
        
        host_df = host_df.iloc[first_data_pkt_idx:].reset_index(drop=True)

        target_length, target_datalength = host_df.loc[0, ['Length', 'DataLength']]
        
        capture_df = pd.read_csv(capture_path, delim_whitespace=True, names=['Type', 'Time', 'Length', 'DataLength', 'SeqNum'], index_col=False)
        seq_df = capture_df[capture_df['Type'] == 'seq'].reset_index(drop=True)
        

        seq_df[['Length', 'DataLength', 'SeqNum']] = seq_df[['Length', 'DataLength', 'SeqNum']].astype(int)

        matching_rows = seq_df[(seq_df['Length'] == target_length) & (seq_df['DataLength'] == target_datalength)]
        matching_idx = matching_rows.index[0]
        seq_df = seq_df.iloc[matching_idx:].reset_index(drop=True)
        
        
        host_df['Time'] = pd.to_datetime(host_df['Time'], format='%H:%M:%S.%f')
        seq_df['Time'] = pd.to_datetime(seq_df['Time'], format='%H:%M:%S.%f')
        
        #print(f'host_df: {host_df.head()}, seq_df: {seq_df.head()}')
        merged_data = []
        for _, host_row in host_df.iterrows():
            seq_num = host_row['SeqNum']
            host_time = host_row['Time']
    
            matching_seq_rows = seq_df[seq_df['SeqNum'] == seq_num]
            value_types = seq_df['SeqNum'].apply(type)
            
            if not matching_seq_rows.empty:
                #print(f'maching_seq_rows: {matching_seq_rows}')
                closest_seq_row = matching_seq_rows[matching_seq_rows['Time'] <= host_time].iloc[-1]
                merged_data.append(closest_seq_row)  
                merged_data.append(host_row)

        if not merged_data:
            return pd.DataFrame(), None

        combined_df = pd.DataFrame(merged_data).reset_index(drop=True)
        combined_df = combined_df.sort_values(by='Time').reset_index(drop=True)

        base_time = combined_df['Time'].iloc[0]
        return combined_df, base_time
        # ------------- -------------

    def forward(self, path, train_data_num, test_data_num, target_env):
        envs = os.listdir(path)
        y_train = []
        x_trace_train = []
        x_env_train = []
        x_len_train = []
        
        y_test = []
        x_trace_test = []
        x_env_test = []
        x_len_test = []
        max_train_length = 0
        max_test_length = 0
        
        alo_dic = {'htcp': 0, 'bbr': 1, 'vegas': 2, 'westwood': 3, 'scalable': 4, 'highspeed': 5, 'veno': 6, 'reno': 7, 'yeah': 8, 'illinois': 9, 'bic': 10, 'cubic': 11}
        
        
        
        for env in envs:
            if env != target_env:
                continue
            alos = os.listdir(os.path.join(path, env))
            numbers = re.findall(r'\d+', env)
            env_numbers = list(map(int, numbers[:2]))
            for alo in tqdm(alos):
                if alo.startswith("host"):
                    alo_name = re.search(r'_(\w+)_', alo).group(1)
                    alo_idx = int(re.search(r'(\d+)(\..*)', alo).group(1))
                    if alo_idx >= train_data_num + test_data_num:
                        continue
                    capture_path = os.path.join(path, env, alo.replace("host", "capture"))
                    host_path = os.path.join(path, env, alo)

                    
                    combined_df, base_time = self.get_merged_df(host_path, capture_path)
                    #print(combined_df.head())
                    if combined_df.empty:
                        continue
                    
                    label_i = [alo_dic[alo_name.lower()]]
                    
                    combined_df['RelativeTime'] = (combined_df['Time'] - base_time).dt.total_seconds()
                    combined_df = combined_df[combined_df['RelativeTime'] <= 20]
                    
                    cumulative_sum = 0
                    data_x_trace = []

                    for index, row in combined_df.iterrows():  
                        if row['Type'] == 'seq':
                            cumulative_sum += 1
                        elif pd.isna(row['Type']):
                            cumulative_sum -= 1

                        data_x_trace.append([row['RelativeTime'], cumulative_sum])
                    #print(len(data_x_trace),data_x_trace[int(len(data_x_trace)/2):int(len(data_x_trace)/2)+50])
                    scaler = MinMaxScaler()
                    data_x_trace = torch.tensor(scaler.fit_transform(data_x_trace), dtype=torch.float32)

                    
                    if alo_idx < train_data_num:
                        y_train.append(label_i)
                        x_env_train.append(env_numbers)
                        x_len_train.append([len(data_x_trace)])
                        x_trace_train.append(data_x_trace)
                        if len(data_x_trace) > max_train_length:
                            max_train_length = len(data_x_trace)
                    elif alo_idx >= train_data_num and alo_idx < train_data_num+test_data_num:
                    #else:
                        y_test.append(label_i)
                        x_env_test.append(env_numbers)
                        x_len_test.append([len(data_x_trace)])
                        x_trace_test.append(data_x_trace)
                        
                        if len(data_x_trace) > max_test_length:
                            max_test_length = len(data_x_trace)
                    
        # Process training data
        data_trace_train = pad_sequence(x_trace_train, batch_first=True, padding_value=0)
        data_trace_train = torch.reshape(data_trace_train, [len(y_train), -1])

        train_data = torch.cat([torch.Tensor(x_len_train), torch.Tensor(x_env_train), torch.Tensor(y_train), data_trace_train], dim=1)

        # Process test data
        data_trace_test = pad_sequence(x_trace_test, batch_first=True, padding_value=0)
        data_trace_test = torch.reshape(data_trace_test, [len(y_test), -1])
        #print(f'data_trace_test: {data_trace_test}')
        
        test_data = torch.cat([torch.Tensor(x_len_test), torch.Tensor(x_env_test), torch.Tensor(y_test), data_trace_test], dim=1)
        test_loader = torch.utils.data.DataLoader(test_data, self.batch_size, shuffle=True, drop_last=True)

        return train_data, test_loader, max_train_length, max_test_length
    



def dtw_similarity(data1, data2):
    alignment = dtw(data1, data2, keep_internals=True)
    distance = alignment.distance / max(len(data1), len(data2))
    return distance


def predict_with_voting(data_trace, test_labels, train_traces, train_labels, method):
    predictions = []
    
    label_groups = {}
    for idx, (trace, label) in enumerate(zip(data_trace, test_labels)):
        if label.item() not in label_groups:
            label_groups[label.item()] = []
        label_groups[label.item()].append((idx, trace))
    print(label_groups.keys())
    if method[1] == 0:
        for label, traces in label_groups.items():
            group_predictions = []
            distances_list = []
            for idx, test_trace in traces:
                distances = []
                for train_trace in train_traces:
                    distance = dtw_similarity(test_trace, train_trace)
                    #print(distance)
                    distances.append(distance)
                
                distances = np.array(distances)
                closest_label = train_labels[np.argmin(distances)]
                group_predictions.append((idx, closest_label))
                distances_list.append((idx, distances))

            vote_counts = Counter([pred for _, pred in group_predictions])
            most_common = vote_counts.most_common()
            print(f'label: {label}, group_predictions: {group_predictions}, most_common: {most_common}')
            if len(most_common) == 1 or most_common[0][1] != most_common[1][1]:
                final_label = most_common[0][0]
            else:
                min_distance = float('inf')
                final_label = None
                for idx, distances in distances_list:
                    closest_label = train_labels[np.argmin(distances)]
                    if min(distances) < min_distance:
                        min_distance = min(distances)
                        final_label = closest_label
                print(f'the final result is: {final_label}')
            for idx, _ in traces:
                predictions.append((idx, final_label))
    
    predictions.sort(key=lambda x: x[0])
    predictions = [pred for _, pred in predictions]
    predictions = torch.tensor(predictions, dtype=torch.int64)
    return predictions



def test_dtw_classify(data_path, batch_size, train_data_num, test_data_num, output_path, target_env, method):
    dataset = queueOccuDataset_build(batch_size)
    #dataset = queueSizeOnlyDataset_build(batch_size)
    train_data, test_loader, max_train_length, max_test_length = dataset.forward(path=data_path, train_data_num=train_data_num, test_data_num=test_data_num, target_env=target_env)
    # Extract training data components
    train_labels = train_data[:, 3].numpy().astype(int)
    #print(f'train_data: {train_data[0:5]}, train_label: {train_labels[0:5]}')
    if method[0] == 0:
        # with padding
        train_traces = torch.reshape(train_data[:, 4:], [-1, max_train_length, 2]).numpy()
    else:
        # without padding
        train_labels = train_data[:, 3].numpy().astype(int)
        train_lengths = train_data[:, 0].numpy().astype(int)
        train_traces = [train_data[i, 4:4+train_lengths[i]*2].reshape(-1, 2).numpy() for i in range(len(train_data))]
    
    
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    f = open(f'{output_path}/result_classify.txt', 'a+')
    # Initialize accuracy metric
    test_accuracy = torchmetrics.Accuracy(task='multiclass', num_classes=12, top_k=1)
    
    for i, input in tqdm(enumerate(test_loader)):
        if method[0] == 0:
            # with padding
            label = input[:, 3].to(torch.int64)
            length = input[:, 0].to(torch.int64)
            data_trace = torch.reshape(input[:, 4:], [-1, max_test_length, 2]).numpy()
        else:
            #without padding   
            label = input[:, 3].to(torch.int64)
            length = input[:, 0].numpy().astype(int)
            x_env = input[:, 1:3]
            data_trace = [input[j, 4:4+length[j]*2].reshape(-1, 2).numpy() for j in range(len(input))]
            
        
        predictions = []
        #print('data_trace: {data_trace}')
        predictions = predict_with_voting(data_trace, label, train_traces, train_labels, method)
        print(f'label: {label}, predictions: {predictions}')
        
        # if method[1] == 0:
        #     for test_trace in data_trace:
        #         distances = []
        #         for train_trace in train_traces:
        #             distance = dtw_similarity(test_trace, train_trace)
        #             distances.append(distance)
                
        #         distances = np.array(distances)
        #         closest_label = train_labels[np.argmin(distances)]
        #         predictions.append(closest_label)
        # else:
        #     for test_trace in data_trace:
        #         label_distances = {label: [] for label in np.unique(train_labels)}
        #         for train_trace, train_label in zip(train_traces, train_labels):
        #             distance = dtw_similarity(test_trace, train_trace)
        #             label_distances[train_label].append(distance)
                
        #         avg_distances = {label: np.mean(distances) for label, distances in label_distances.items()}
        #         closest_label = min(avg_distances, key=avg_distances.get)
        #         predictions.append(closest_label)   
        #predictions = torch.tensor(predictions, dtype=torch.int64)
        
        test_accuracy.update(predictions, label)
    
    accuracy = test_accuracy.compute().item()
    print(f"Test Accuracy: {accuracy * 100:.2f}%")
    f.write(f"Env: {target_env}, Test Accuracy: {accuracy * 100:.2f}%\n")
    f.close()

def calculate_average_accuracy(input_file):
    accuracies = []
    
    with open(input_file, 'r') as infile:
        lines = infile.readlines()
    
    for line in lines:
        if "Test Accuracy:" in line:
            accuracy_str = line.split("Test Accuracy:")[1].strip().replace('%', '')
            accuracy = float(accuracy_str)
            accuracies.append(accuracy)
    
    if accuracies:
        average_accuracy = sum(accuracies) / len(accuracies)
        return average_accuracy
    else:
        return None

def calculate_average_accuracy_high_latency(input_file):
    filtered_lines = []
    accuracies = []

    with open(input_file, 'r') as infile:
        lines = infile.readlines()
    
    for line in lines:
        if "Test Accuracy:" in line:
            # Extract RTT value
            start_idx = line.find('rtt_') + len('rtt_')
            end_idx = line.find('ms', start_idx)
            rtt_value = int(line[start_idx:end_idx])
            
            if rtt_value > 300:
                # Extract accuracy value
                accuracy_str = line.split("Test Accuracy:")[1].strip().replace('%', '')
                accuracy = float(accuracy_str)
                
                filtered_lines.append(line)
                accuracies.append(accuracy)
    
    return filtered_lines, accuracies


if __name__ == "__main__":
    
    train_path = './DataCollection/Data/xxx'
    train_data_num = 3
    test_data_num = 5
    output_path = './'
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    envs = os.listdir(train_path)
    #method_list = [[0,0], [1,0], [1,1]]
    method_list = [[0,0]]
    for method in method_list:
        method_dic = {0:'with padding', 1:'without padding', 2:'nearest label', 3:'average dis'}
        with open(f'{output_path}/results.txt', 'a+') as f:
            f.write(f'\n========== method: {method_dic[method[0]]} & {method_dic[method[1]+2]} ==========\n')
        for target_env in envs:
            test_dtw_classify(train_path, test_data_num*12, train_data_num, test_data_num, output_path, target_env, method)
    

    