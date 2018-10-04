from __future__ import print_function
import os
import numpy as np
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import StratifiedKFold


save_path = "Tang_SMOTE_npz"
smote_size = 100

dir = "Tang_npz"
print('File name: ', dir)
data_dir = os.listdir(dir)

for data_file in data_dir:
    name = data_file.split(".")[0]
    data_path = dir + '/' + data_file
    r = np.load(data_path)

    Feature_train = r['F_tr']
    Label_train = r['L_tr']
    Num_train = Feature_train.shape[0]

    Feature_test = r['F_te']
    Label_test = r['L_te']
    Label_test.ravel()

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

    npy_name = save_path + '/' + name + '_SMOTE.npz'
    np.savez(npy_name, S_F_tr_l = SMOTE_feature_train_list, S_L_tr_l = SMOTE_label_train_list,
             S_F_va_l = SMOTE_feature_valid_list, S_L_va_l = SMOTE_label_valid_list,
             F_te = Feature_test, L_te = Label_test)
