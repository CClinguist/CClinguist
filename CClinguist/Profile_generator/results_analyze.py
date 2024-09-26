'''
find the best init profile, enable the most CCA to be recognized with the smallest step
'''
import re
flag_start=False
dic_alo={}
dic_env={}
file_name = 'test_0926-1427-30'
epoch_seq = '0'
alo_cnt = 0


def is_complete_line(line):
    return line.count(',') == 5


with open(f'Profile_generator/results/log_output/test_log/{file_name}.txt', 'r') as f:
    lines = f.readlines()
 

processed_lines = []
temp_line = ''
for line in lines:
    stripped_line = line.strip()
    if stripped_line:  
        if temp_line:
            combined_line = temp_line + ' ' + stripped_line
            if is_complete_line(combined_line):
                # print(combined_line)
                processed_lines.append(combined_line)
                temp_line = ''  
            else:
                temp_line = combined_line  
        else:
            if is_complete_line(stripped_line):
                # print(stripped_line)
                processed_lines.append(stripped_line)
            else:
                temp_line = stripped_line
   
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



f_log=open(f'Profile_generator/results/analyze/{file_name}_analyze_env_epoch{epoch_seq}.txt','w')

env_combinations =[(60, 200), (60, 400), (60, 600), (60, 800), (60, 1000), (80, 200), (80, 400), (80, 600), (80, 800), (80, 1000),
                         (100, 200), (100, 400), (100, 600), (100, 800), (100, 1000), (120, 200), (120, 400), (120, 600), (120, 800), (120, 1000),
                         (140, 200), (140, 400), (140, 600), (140, 800), (140, 1000), (160, 200), (160, 400), (160, 600), (160, 800), (160, 1000),
                         (180, 200), (180, 400), (180, 600), (180, 800), (180, 1000), (200, 200), (200, 400), (200, 600), (200, 800), (200, 1000),
                         (220, 200), (220, 400), (220, 600), (220, 800), (220, 1000), (240, 200), (240, 400), (240, 600), (240, 800), (240, 1000),
                         (260, 200), (260, 400), (260, 600), (260, 800), (260, 1000), (280, 200), (280, 400), (280, 600), (280, 800), (280, 1000),
                         (300, 200), (300, 400), (300, 600), (300, 800), (300, 1000)]
        

envs = {i+1: f'rtt_{rtt}ms_bdw_{bdw}Kbps' for i, (rtt, bdw) in enumerate(env_combinations)}

#1.find the best init profile, enable the most CCA to be recognized with the smallest step
min_step=10000
flag_cannot=False
best_env_accuracy = 0
best_envs = []
max_accuracy = 0


for key in dic_env:
    alo_cnt=0
    all_alo_list=dic_env[key]
    env_name=envs[int(key)]
    f_log.write('\n')
    f_log.write('env: '+key+' '+env_name+'\n')
    avg_step=0
    correct_predictions = 0
    for step in all_alo_list:
        info=all_alo_list[step]
        num_step=len(info)
        alo_cnt+=num_step
        avg_step+=num_step*int(step)
        if step==' 7':
            flag_cannot=True
        f_log.write('step: '+str(step)+' num of alos '+str(num_step)+'\n')
        for i in range(num_step):
            if info[i][0] == info[i][2].strip(' []'):
                correct_predictions += 1
            f_log.write(str(info[i])+'\n')
    avg_step=avg_step/alo_cnt
    accuracy = correct_predictions / alo_cnt
    
    if accuracy > max_accuracy:
        max_accuracy = accuracy
        best_envs = [(key, avg_step)]
    elif accuracy == max_accuracy:
        best_envs.append((key, avg_step))
        
    f_log.write('avg_step: '+str(avg_step)+'\n')
f_log.write('\n')

if flag_cannot:
    f_log.write('there are {} alos cannot be recognized \n'.format(num_step))
    
for best_env, min_step in best_envs:
    f_log.write('best init profile: '+best_env+' '+envs[int(best_env)]+'\n')
    f_log.write('accuracy:' +str(max_accuracy)+'\n')
    f_log.write('average step of the tree: '+str(min_step)+'\n')
    f_log.write('\n')

f_log.write('\n')
f_log.close()
    
    
            
        
    