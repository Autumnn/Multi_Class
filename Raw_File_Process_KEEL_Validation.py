from __future__ import print_function
import os
import numpy as np
from sklearn.model_selection import StratifiedKFold


def Initialize_Data(dir, symbol):
    Num_lines = len(open(dir, 'r').readlines())
    num_columns = 0
    data_info_lines = 0
    class_label = []
    with open(dir, "r") as get_info:
        print("name", get_info.name)
        for line in get_info:
            if line.find("@") == 0:
                data_info_lines += 1
                content = line.split(' ')
                if symbol in content:
                    num_labels = len(content)
                    label_string = ''
                    for j in range(num_labels - 2):
                        label_string += content[j + 2]
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
                        label = class_label.index(row[i].strip()) + 1
                        Labels[l - data_info_lines - 1][0] = label

    print("Read Completed")


def get_feature():
    return Features


def get_label():
    return Labels


path = "KEEL_Data_5_Folder"
files = os.listdir(path)
for file in files:
    print('File name: ', file)
    dir = path + '/' + file
    data_dir = os.listdir(dir)
    if file == 'glass-5-fold':
        symbol = 'typeGlass'
    else:
        symbol = 'class'
    for data_file in data_dir:
        name = data_file.split(".")[0]
        sub_name = name.split("-")[-1]
        data_name = sub_name[-3:]
        data_path = dir + '/' + data_file
        Initialize_Data(data_path, symbol)

        if data_name == 'tra':
            Feature_train = get_feature()
            Label_train = get_label()

            skf = StratifiedKFold(n_splits=4, shuffle=False)
            Feature_train_list = []
            Label_train_list = []
            Feature_valid_list = []
            Label_valid_list = []
            for train_idx, valid_idx in skf.split(Feature_train, Label_train):
                Feature_train_list.append(Feature_train[train_idx])
                Label_train_list.append(Label_train[train_idx])
                Feature_valid_list.append(Feature_train[valid_idx])
                Label_valid_list.append(Label_train[valid_idx])

        elif data_name == 'tst':
            Feature_test = get_feature()
            Label_test = get_label()
            npy_name = file.split("-")[0] + '_' + sub_name[0] + '_Cross_Folder.npz'
            np.savez(npy_name, F_tr_l = Feature_train_list, L_tr_l = Label_train_list,
                     F_va_l = Feature_valid_list, L_va_l = Label_valid_list,
                     F_te = Feature_test, L_te = Label_test)
