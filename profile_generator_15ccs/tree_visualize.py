import os
import re
import graphviz
from datetime import datetime



def read_data_from_file(file_path):
    with open(file_path, 'r') as file:
        data_str = file.read()
    return data_str


def parse_data(data_str):
    data = {}
    lines = data_str.split('\n')
    current_env = None
    best_nodes = set()
    for line in lines:
        line = line.strip()
        if line.startswith('env:'):

            current_env = line.split()[1]
            data[current_env] = {'name': line.strip(), 'steps': {}}
        elif line.strip().startswith('['):
            parts = line.strip('[]').split(',')


            alo_str = parts[0].strip()  
            raw_env_sequence = parts[1].strip().strip('[]')  

            cleaned_env_sequence = []
            temp_num = ""
            for char in raw_env_sequence:
                if char.isdigit():
                    temp_num += char
                elif temp_num:
                    cleaned_env_sequence.append(temp_num)
                    temp_num = ""
            if temp_num:  
                cleaned_env_sequence.append(temp_num)

            predicted_alos = parts[2].strip().strip('[]').split()  
            step = parts[3].strip() 
            
            if step not in data[current_env]['steps']:
                data[current_env]['steps'][step] = []


            predicted_alo_str = predicted_alos[-1].strip()  
            
            alo_str = re.sub(r"[^\d]", "", alo_str)
            predicted_alo_str = re.sub(r"[^\d]", "", predicted_alo_str)

            
            alo = int(alo_str)
            predicted_alo = int(predicted_alo_str)

            correct = (predicted_alo == alo)

            data[current_env]['steps'][step].append((alo, cleaned_env_sequence, correct))
            
        elif line.startswith('Best Environment Parameters:'):
            # envs_info = line.split(':')
            # env_name = envs_info[1].strip()
            # best_nodes.add(env_name)
            best_nodes.add('rtt_160ms_bdw_600Kbps') 
           
    return data, best_nodes




def build_and_render_graphs(data, output_dir):
    file_name = os.path.basename(data_file_path)
    base_directory = os.path.join(output_dir, file_name.split('.')[0])
    os.makedirs(base_directory, exist_ok=True)
    data, best_nodes = data

    for env, info in data.items():
        dot = graphviz.Digraph(comment=info['name'])
        created_nodes = set()
        existing_edges = set()  
        root_node = f"Env {env} ({envs[int(env)]})"
        root_node_name = envs[int(env)]
        if root_node_name not in best_nodes:
            continue  # Skip if the root node is not in the best nodes
        dot.node(root_node, root_node, shape='box', style='filled', fillcolor='lightyellow')
        created_nodes.add(root_node)

        total_alos = len(alo_dic)
        recognized_alos = set()
        unrecognized_alos = set()

        for step, alos in info['steps'].items():
            path_nodes = {root_node}
            for alo, env_sequence, correct in alos:
                if correct:
                    recognized_alos.add(alo)
                else:
                    unrecognized_alos.add(alo)

                last_node = root_node
                for env_index in env_sequence:
                    try:
                        env_node = f"Env {env_index} ({envs[int(env_index)]})"
                    except KeyError:
                        print(f"Warning: Key {env_index} not found in envs, skipping.")
                        continue  

                    if env_node not in path_nodes:
                        if env_node not in created_nodes:
                            dot.node(env_node, env_node, shape='box', style='filled', fillcolor='lightyellow')
                            created_nodes.add(env_node)

                        edge = (last_node, env_node)
                        if edge not in existing_edges:
                            dot.edge(last_node, env_node)
                            existing_edges.add(edge)
                        path_nodes.add(env_node)
                    last_node = env_node


                alo_name = alo_dic.get(alo, f"ALO {alo}")
                alo_node = f"{env}_{alo}"
                if alo_node not in created_nodes:
                    fill_color = 'lightblue'
                    dot.node(alo_node, alo_name, shape='ellipse', style='filled', fillcolor=fill_color)
                    created_nodes.add(alo_node)

                edge = (last_node, alo_node)
                if edge not in existing_edges:
                    dot.edge(last_node, alo_node)
                    existing_edges.add(edge)

        file_path = os.path.join(base_directory, f"env_{env}.gv")
        dot.render(file_path, format='png', cleanup=True)


analyze_dir = 'results_15CCs_coef/analyze'
file_names = sorted([f for f in os.listdir(analyze_dir) if f.endswith('_analyze.txt')])

# select one file for vuisualization
for file_name in file_names:
    # Path to the input data file (including filename) 
    data_file_path = os.path.join(analyze_dir, file_name)
    # Directory containing additional data files (if any)  
    data_path = 'data_15ccs/simulation_pk2hk_15CCs'
    # Directory where results/visualizations will be saved  
    output_path = 'tree_visualization'
    

    envs_combinations = os.listdir(data_path)
    envs = {i+1: env for i, env in enumerate(envs_combinations)}


    alo_dic={0: 'htcp', 1: 'bbr', 2: 'vegas', 3: 'westwood', 4: 'scalable', 5: 'highspeed', 
         6: 'veno', 7: 'reno', 8: 'yeah', 9: 'illinois', 10: 'bic', 11: 'cubic' ,
         12:'pccLatency', 13:'astraea', 14:'pccLoss'}

    
    data_str = read_data_from_file(data_file_path)
    data = parse_data(data_str)
    build_and_render_graphs(data, output_path)




