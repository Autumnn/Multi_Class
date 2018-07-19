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





