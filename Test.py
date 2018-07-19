import os
import numpy as np


dir = 'KEEL_Data/ecoli-5-fold/ecoli-5-1tra.dat'
Num_lines = len(open(dir, 'r').readlines())
num_columns = 0
data_info_lines = 0
class_label = []
with open(dir, "r") as get_info:
    for line in get_info:
        if line.find("@") == 0:
            data_info_lines += 1
            content = line.split(' ')
            if 'class' in content:
                num_labels = len(content)
                label_string = ''
                for j in range(num_labels-2):
                    label_string += content[j+2]
                label_string = label_string.strip('{')
                label_string = label_string.strip()
                label_string = label_string.strip('}')
                class_label = label_string.split(',')
                print(class_label)
        else:
            columns = line.split(",")
            num_columns = len(columns)
            break


global Num_Samples
Num_Samples = Num_lines - data_info_lines
print(Num_Samples)
global Num_Features
Num_Features = num_columns - 1

global Features
Features = np.ones((Num_Samples, Num_Features))
global Labels
Labels = np.ones((Num_Samples, 1))

with open(dir, "r") as data_file:
    print("Read Data", data_file.name)
    l = 0
    for line in data_file:
        l += 1
        if l > data_info_lines:
            # print(line)
            row = line.split(",")
            length_row = len(row)
            # print('Row length',length_row)
            # print(row[0])
            #print(l)
            for i in range(length_row):
                if i < length_row - 1:
                    if row[i] == '<null>':
                        row[i] = 0
                    Features[l - data_info_lines - 1][i] = row[i]
                    # print(Features[l-14][i])
                else:
                    label = class_label.index(row[i].strip())+1
                    Labels[l - data_info_lines - 1][0] = label



