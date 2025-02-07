
import re
import os

flag_start=False
dic_alo={}
dic_env={}


current_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(current_dir, 'data', 'test_data')
log_output_path = os.path.join(current_dir, 'results', 'log_output', 'test_log')
analyze_path = os.path.join(current_dir, 'results', 'analyze')
file_names = ['test_0207-1048-34.txt']


epoch_seq = '0'
alo_cnt = 12    
alo_dic = {
    'htcp': 0, 'bbr': 1, 'vegas': 2, 'westwood': 3, 'scalable': 4, 
    'highspeed': 5, 'veno': 6, 'reno': 7, 'yeah': 8, 'illinois': 9, 
    'bic': 10, 'cubic': 11
}

def is_complete_line(line):
    
    return line.count(',') == 5

for file_name in file_names:
    dic_alo = {}
    dic_env = {}
        
    with open(os.path.join(log_output_path, file_name), 'r') as f:
        lines = f.readlines()

    processed_lines = []
    temp_line = ''
    for line in lines:
        stripped_line = line.strip()
        if stripped_line:  
            if temp_line:
                
                combined_line = temp_line + ' ' + stripped_line
                if is_complete_line(combined_line):
                    
                    processed_lines.append(combined_line)
                    temp_line = ''  
                else:
                    temp_line = combined_line  
            else:
                
                if is_complete_line(stripped_line):
                    
                    processed_lines.append(stripped_line)
                else:
                    temp_line = stripped_line
        # print(stripped_line)


    if temp_line:
        processed_lines.append(temp_line)
 
    for line in processed_lines:
        # print(line)
        tmp = line.split(',')
        if len(tmp) >= 6:  
            alo_label = tmp[0]
            env_input = tmp[1]
            action = tmp[2]
            alo_predicted = tmp[3]
            reward_actions = tmp[4]
            steps_search = tmp[5].strip()
            if alo_label not in dic_alo:
                dic_alo[alo_label] = []
            dic_alo[alo_label].append([env_input, action, alo_predicted, steps_search])
            if env_input not in dic_env:
                dic_env[env_input] = {}
            if steps_search not in dic_env[env_input]:
                dic_env[env_input][steps_search] = []
            dic_env[env_input][steps_search].append([alo_label, action, alo_predicted, steps_search])         
                
    f.close()

    

    with open(os.path.join(analyze_path, f'{file_name}_analyze_env_epoch{epoch_seq}.txt'), 'w') as f_log:
        envs={}
        i=0
        envs_combinations = os.listdir(data_path)
        for env in envs_combinations:

            i+=1
            envs[i] = env

        min_step=10000
        flag_cannot=False
        best_env_accuracy = 0

        for key in dic_env:
            all_alo_list=dic_env[key]
            env_name=envs[int(key)]
            f_log.write('\n')
            f_log.write('env: '+key+' '+env_name+'\n')
            
            avg_step=0
            correct_predictions = 0
            for step in all_alo_list:
                info=all_alo_list[step]
                num_step=len(info)

                avg_step+=num_step*int(step)
                
                if step==' 7':
                    flag_cannot=True
                f_log.write('step: '+str(step)+' num of alos '+str(num_step)+'\n')
                for i in range(num_step):
                    if info[i][0] == info[i][2].strip(' []'):
                        correct_predictions += 1
                    f_log.write(str(info[i])+'\n')

            f_log.write('avg_step: '+str(avg_step/alo_cnt)+'\n')
        f_log.write('\n')


        envs_acc={}
        best_env_accuracy = 0
        best_env = None
        best_env_params = None




        alo_name_map = {v: k for k, v in alo_dic.items()}

        for env in dic_env:
            env_key = env.split(' ')[-1]
            f_log.write('Env Info: '+str(env_key)+' '+ envs[int(env_key)] + '\n')
            datas=dic_env[' '+env_key]
            acc_alos={}   
            num_test = {}  
            
            for step in datas:
                info=datas[step]
                num_step=len(info)
                for i in range(num_step):
                    alo=info[i][0]
                    if alo not in acc_alos:
                        acc_alos[alo]=0
                        num_test[alo] = 0
                    num_test[alo] += 1
                    # a= info[i][2].strip(' []').split(' ')[-1]
                    if info[i][0]==info[i][2].strip(' []').split(' ')[-1]:
                        acc_alos[alo]+=1
            

            acc_sum = 0
            f_log.write(f'total tests: {num_test[alo]}\n')
            for alo in acc_alos:
                if num_test[alo] > 0:
                    accuracy = acc_alos[alo]/num_test[alo]
                else:
                    accuracy = 0

                acc_sum += accuracy
                alo_name = alo_name_map[int(alo)]
                f_log.write(f'alo_index: {alo}, alo_name = {alo_name}, accuracy: {accuracy}\n')

            avg_accuracy = acc_sum/alo_cnt
            f_log.write(f'average accuracy: {avg_accuracy}'+'\n\n')

            

            if avg_accuracy > best_env_accuracy:
                best_env_accuracy = avg_accuracy
                best_env = env_key
                best_env_params = envs[int(env_key)]
            
            if env_key not in envs_acc:
                envs_acc[env_key]=[]
            envs_acc[env_key].append(avg_accuracy)


        f_log.write('--------------------------------------------------\n')
        f_log.write(f'Best Environment Accuracy: {best_env_accuracy}\n')
        f_log.write(f'Best Environment ID: {best_env}\n')
        f_log.write(f'Best Environment Parameters: {best_env_params}\n\n')



        sorted_envs_acc=sorted(envs_acc,key=lambda x:envs_acc[x][0],reverse=True)
        for env in sorted_envs_acc:
            f_log.write(env+' '+envs[int(env)]+' '+str(envs_acc[env][0])+'\n')
            

        f_log.write('\n--------------------------------------------------\n')
        f_log.write('TOP 3 Environments Algorithm Accuracy:\n')

        top_3_envs = sorted_envs_acc[:3]
        top_3_algo_acc = {alo: 0 for alo in alo_name_map.keys()} 
        top_3_algo_tests = {alo: 0 for alo in alo_name_map.keys()} 

        for env in top_3_envs:    
            current_env_accuracy = envs_acc[env][0]  
            f_log.write(f'\nEnvironment ID: {env}, Parameters: {envs[int(env)]},  Accuracy: {current_env_accuracy:.4f}\n')

            env_data = dic_env[f' {env}']

            for step, step_data in env_data.items():
                for data in step_data:
                    alo_index = int(data[0])  
                    predicted = data[2].strip(' []').split(' ')[-1]
                    top_3_algo_tests[alo_index] += 1
                    if data[0] == predicted:  
                        top_3_algo_acc[alo_index] += 1


        f_log.write('\nAverage Accuracy in TOP 3 Environments:\n')
        for alo_index, total_acc in top_3_algo_acc.items():
            total_tests = top_3_algo_tests[alo_index]
            avg_accuracy = total_acc / total_tests if total_tests > 0 else 0
            algo_name = alo_name_map[alo_index]
            f_log.write(f'Algorithm Index: {alo_index}, Name: {algo_name}, Total Tests: {total_tests}, Average Accuracy: {avg_accuracy:.4f}\n')

        f_log.write('\n--------------------------------------------------\n')
        f_log.write('Voting Mechanism in TOP 3 Environments:\n')

        
        algo_votes = {alo: 0 for alo in alo_name_map.keys()}  
        algo_env_results = {alo: {'correct': 0, 'total': 0} for alo in alo_name_map.keys()}  


        for env in top_3_envs:
            env_data = dic_env[f' {env}']
            f_log.write(f'\nEnvironment ID: {env}, Parameters: {envs[int(env)]}\n')
            
    
            for step, step_data in env_data.items():
                for data in step_data:
                    alo_index = int(data[0])  
                    predicted = data[2].strip(' []').split(' ')[-1]
                    algo_env_results[alo_index]['total'] += 1  
                    if data[0] == predicted:  
                        algo_env_results[alo_index]['correct'] += 1


        f_log.write('\nAlgorithm Voting Results:\n')
        for alo_index, results in algo_env_results.items():
            correct_count = results['correct']
            total_tests = results['total']
            accuracy = correct_count / total_tests if total_tests > 0 else 0  
            vote_result = 1 if accuracy >= 2 / 3 else 0  
            algo_votes[alo_index] = vote_result
            

            algo_name = alo_name_map[alo_index]
            f_log.write(f'Algorithm Index: {alo_index}, Name: {algo_name}, '
                        f'Correct Recognitions: {correct_count}, Total Tests: {total_tests}, '
                        f'Accuracy: {accuracy:.4f}, Voting Result: {"Pass" if vote_result else "Fail"}\n')


        total_correct_algorithms = sum(algo_votes.values())
        total_algorithms = len(algo_votes)
        final_accuracy = total_correct_algorithms / total_algorithms if total_algorithms > 0 else 0

        f_log.write('\nFinal Voting-based Accuracy:\n')
        f_log.write(f'Total Correct Algorithms: {total_correct_algorithms}\n')
        f_log.write(f'Total Algorithms: {total_algorithms}\n')
        f_log.write(f'Final Accuracy: {final_accuracy:.4f}\n')
    

        