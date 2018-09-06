import os
import numpy as np
from pandas import DataFrame


best_result = {'RA': {}, 'BA': {}, 'GA': {}, 'CMA': {}, 'PYS': {}}
dataset_list = []

path = 'Results/From_Server_o'
Dir = os.listdir(path)
for file in Dir:
    algorithm = file.split('_')[0]
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
                best_result[algorithm][data_Set] = value
            else:
                if best_result[algorithm][data_Set] < value:
                    best_result[algorithm][data_Set] = value

best_result_list = {'RA': [], 'BA': [], 'GA': [], 'CMA': [], 'PYS': []}
for k, v in best_result.items():
    for kk, vv in v.items():
        best_result_list[k].append(vv)
best_result_df = DataFrame(best_result_list, index=dataset_list)
print(best_result_df)


with open('Best_Optimized_Result.txt', 'a') as w:
    headline = 'G_mean' + '\t' + '\t'.join(str(x) + '\t' for x in best_result.keys()) + '\n'
    w.write(headline)
    for index, row in best_result_df.iterrows():
        l_m = list(row)
        seq = row.sort_values(ascending=False).values
        #seq = np.sort(np.asarray(l_m))
        w_line = index
        for i in range(len(l_m)):
            w_line += '\t' + str('%.4f' % l_m[i])
            w_line += '\t' + str(np.mean(np.where(seq == l_m[i])[0])+1)
        #            w_line += '(' + str(len(l_m) - np.mean(np.where(seq == l_m[i])[0])) + ')'
        #            w_line += '\t' + str('%.4f' % l_v[i])
        w_line += '\n'
        w.write(w_line)


