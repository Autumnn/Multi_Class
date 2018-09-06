import os
import shutil
import numpy as np
from pandas import DataFrame


algorithm_list = ['RA', 'BA', 'GA', 'CMA']

for algorithm in algorithm_list:
    dataset_index = {}
    best_result = {}
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
                if idx == '0':
                    best_result[data_Set] = value
                    dataset_index[data_Set] = idx
                else:
                    if best_result[data_Set] < value:
                        best_result[data_Set] = value
                        dataset_index[data_Set] = idx

    para_path = 'KEEL_Cross_Folder_XGBoost_Para/Server/' + algorithm
    dataset_path = 'KEEL_Cross_Folder_npz'
    dataset_dir = os.listdir(dataset_path)
    for data_dir in dataset_dir:
        destination_path = para_path + '_use/' + data_dir
        os.makedirs(destination_path)

    for key, value in dataset_index.items():
        subname = key.split('_')[0]
        ori_file = para_path + '/' + algorithm + '_Validation_' + str(value) + '/' +str(subname) + '/' + key + '_' + algorithm + '.json'
        des_file = para_path + '_use/' + str(subname) + '/' + key + '_' + algorithm + '.json'
        shutil.copyfile(ori_file, des_file)

    result_file = 'Best_Optimized_Result_' + algorithm + '.txt'
    with open(result_file, 'a') as w:
        for k, v in best_result.items():
            w_line = str(k) + '\t' + str(v) + '\t' + str(dataset_index[k]) + '\n'
            w.write(w_line)





