import os
import shutil
import numpy as np
from pandas import DataFrame


algorithm_result = {'RA': {}, 'BA': {}, 'GA': {}, 'CMA': {}, 'PYS': {}}
dataset_list = []

for algorithm in algorithm_result.keys():
    path = 'Results/From_Server/' + algorithm
    Dir = os.listdir(path)
    for file in Dir:
        idx = file.split('.')[0].split('_')[-1]
        print(file)
        name = path + '/' + file

        with open(name, 'r') as result:
            for line in result:
                data_Set = line.split('\t')[0]
                value = float(line.split('\t')[1])
                if data_Set not in dataset_list:
                    dataset_list.append(data_Set)
                if idx == '0':
                    algorithm_result[algorithm][data_Set] = [value, idx]
                else:
                    if algorithm_result[algorithm][data_Set][0] < value:
                        algorithm_result[algorithm][data_Set] = [value, idx]

best_idx = {}
for dataset in dataset_list:
    r_t = []
    for k, v in algorithm_result.items():
        r_t.append(v[dataset])
    n_t = np.array(r_t)
    keys = list(algorithm_result.keys())
    max_id = n_t.argmax(axis=0)[0]
    best_idx[dataset] = [keys[max_id], n_t[max_id][1], n_t[max_id][0]]


para_path = 'KEEL_Cross_Folder_XGBoost_Para/Server/'
dataset_path = 'KEEL_Cross_Folder_npz'
dataset_dir = os.listdir(dataset_path)
for data_dir in dataset_dir:
    destination_path = para_path + '/All_use/' + data_dir
    os.makedirs(destination_path)

for key, value in best_idx.items():
    subname = key.split('_')[0]
    algorithm = value[0]
    id = value[1]
    ori_file = para_path + algorithm + '/' + algorithm + '_Validation_' + str(id) + '/' +str(subname) + '/' + key + '_' + algorithm + '.json'
    des_file = para_path + 'All_use/' + str(subname) + '/' + key + '.json'
    shutil.copyfile(ori_file, des_file)

result_file = 'Best_Optimized_Result_idx.txt'
with open(result_file, 'a') as w:
    for k, v in best_idx.items():
        w_line = str(k) + '\t' + str(v[2]) + '\t' + str(v[0]) + '\t' + str(v[1]) + '\n'
        w.write(w_line)




