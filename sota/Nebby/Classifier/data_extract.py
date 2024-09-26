import pandas as pd
import os
import re
import math





data_path='09_18_realNebby_jp2jp_baseRtt_0'
data_new_path='nebby_data_hk2jp_basertt_0'
if os.path.exists(data_new_path)==False:
    os.makedirs(data_new_path)




dst=['47.83.16.111','47.239.206.39','100.64.0.2']
src=['8.209.201.29','47.74.8.123','47.91.25.136']


envs=['rtt_102ms_bdw_200Kbps']

# envs=['rtt_202ms_bdw_200Kbps']

for env in envs:
    numbers = re.findall(r'\d+', env)
    env_numbers = list(map(int, numbers[:2])) 
    rtt=int(env_numbers[0])
    if rtt>100:
        base_delay=rtt/2-50
        post_delay=50
    if rtt>200:
        base_delay=rtt/2-100
        post_delay=100
    bw=env_numbers[1]
    env_path=os.path.join(data_path,env)
    files=os.listdir(env_path)
    for file in files:

        type='none'
        match = re.search(r'\d+', file)
        times = match.group()
        
        match = re.search(r'_(\w+)_',file)
        alo = match.group(1)
       
        file_name=alo+'-'+times+'-'+str(int(base_delay))+'-'+str(int(post_delay))+'-'+str(bw)+'-2.csv'
        new_data_path=data_new_path+'/'+file_name
        fnew=open(new_data_path,'w')

        csv_datas=pd.read_csv(os.path.join(env_path,file),sep=',')
        fnew.write('Type Time DataLength Raw_Seq/Ack\n')
        print((len(csv_datas)))
        for i in range(int(len(csv_datas))):
            csv_datas = csv_datas.applymap(lambda x: x.strip('"') if isinstance(x, str) else x)
            time=csv_datas.iloc[i,1]
            seq_flag=csv_datas.iloc[i,-1]
            ack_flag=csv_datas.iloc[i,-2]
            src_ip=csv_datas.iloc[i,5]
            dst_ip=csv_datas.iloc[i,7]
            if src_ip in src:
                type='seq'
                id=csv_datas.iloc[i,-2]
            elif src_ip in dst:
                type='ack'
                id=csv_datas.iloc[i,-1]
            length=csv_datas.iloc[i,-3]
            if type!='none' and str(id)!='nan' and str(length)!='nan':
                fnew.write(type +','+ str(time)+','+str(int(length))+','+str(int(id))+'\n')
    
    
    