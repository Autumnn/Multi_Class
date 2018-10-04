from __future__ import print_function
import os
import numpy as np
from imblearn.over_sampling import SMOTE
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

save_path = "KEEL_SMOTE_npz"
smote_size = 100

path = "KEEL_Data/KEEL_Data_5_Folder_S"
files = os.listdir(path)
for file in files:
    print('File name: ', file)
    dir = path + '/' + file
    data_dir = os.listdir(dir)
    data_set = file.split('-')[0]
    data_folder = save_path + '/' + data_set
    os.makedirs(data_folder)

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
            seeds = np.random.randint(0, smote_size*100, smote_size)
            SMOTE_feature_train_list = []
            SMOTE_label_train_list = []
            SMOTE_feature_valid_list = []
            SMOTE_label_valid_list = []
            for k in range(smote_size):
                sm = SMOTE(random_state=seeds[k])
                Feature_train_o, Label_train_o = sm.fit_sample(Feature_train, Label_train.ravel())

                skf = StratifiedKFold(n_splits=4, shuffle=False)
                Feature_train_list = []
                Label_train_list = []
                Feature_valid_list = []
                Label_valid_list = []
                for train_idx, valid_idx in skf.split(Feature_train_o, Label_train_o):
                    Feature_train_list.append(Feature_train_o[train_idx])
                    Label_train_list.append(Label_train_o[train_idx])
                    Feature_valid_list.append(Feature_train_o[valid_idx])
                    Label_valid_list.append(Label_train_o[valid_idx])

                SMOTE_feature_train_list.append(Feature_train_list)
                SMOTE_label_train_list.append(Label_train_list)
                SMOTE_feature_valid_list.append(Feature_valid_list)
                SMOTE_label_valid_list.append(Label_valid_list)

        elif data_name == 'tst':
            Feature_test = get_feature()
            Label_test = get_label()
            npy_name = data_folder + '/' + data_set + '_' + sub_name[0] + '_Cross_Folder.npz'
            np.savez(npy_name, S_F_tr_l = SMOTE_feature_train_list, S_L_tr_l = SMOTE_label_train_list,
                     S_F_va_l = SMOTE_feature_valid_list, S_L_va_l = SMOTE_label_valid_list,
                     F_te = Feature_test, L_te = Label_test)
