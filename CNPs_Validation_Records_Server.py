import os
import numpy as np
import xgboost
import json
import warnings
import cma
import tensorflow as tf
from metrics_list import MetricList
from datetime import datetime
from print_log import PrintLog
from CNPs_OP import CNPs_Optimization


warnings.filterwarnings('ignore')
PATH = "KEEL_Cross_Folder_Valid_npz"
#PATH = "KEEL_Cross_Folder_npz_S"
DIRS = os.listdir(PATH)         #   Get files in the folder

parameters = {'max_depth': (3, 10),         # int
              'learning_rate': (0.01, 0.3),
              'n_estimators': (50, 1000),       #int
              'gamma': (0, 1.),
              'min_child_weight': (1, 10),
              'max_delta_step': (0, 0.3),
              'subsample': (0.5, 1),
              'colsample_bytree': (0.5, 1)}
para_bounds = np.array(list(parameters.values()), dtype=np.float)
para_keys = list(parameters.keys())

def evaluate(para_value):
    keys = para_keys
    dim = len(keys)
#    print(type(individual))
#    print(individual)
    for i in range(dim):
        (lo, up) = para_bounds[i]
        if para_value[i] < lo:
            sentence = 'increase ' + str(keys[i]) + ' : ' + str(para_value[i]) + ' to lower bound !'
            #print(sentence)
            para_value[i] = lo
        elif para_value[i] > up:
            sentence = 'decrease ' + str(keys[i]) + ' : ' + str(para_value[i]) + ' to upper bound !'
            #print(sentence)
            para_value[i] = up

    BayesOp_Parameters = dict(zip(keys, para_value))
    BayesOp_Parameters['silent'] = True
    BayesOp_Parameters['nthread'] = -1
    BayesOp_Parameters['seed'] = 0
    #    BayesOp_Parameters['objective'] = "multi:softprob"
    BayesOp_Parameters['max_depth'] = int(BayesOp_Parameters['max_depth'])
    BayesOp_Parameters['n_estimators'] = int(BayesOp_Parameters['n_estimators'])

    r = np.load(name)
    Feature_train_list = r['F_tr_l']
    Label_train_list = r['L_tr_l']
    Feature_valid_list = r['F_va_l']
    Label_valid_list = r['L_va_l']
    Num_list = len(Feature_train_list)

    Num_Cross_Folders = Num_list
    ml_record = MetricList(Num_Cross_Folders)
    i = 0
    for j in range(Num_list):
        Feature_train = Feature_train_list[j]
        Label_train = Label_train_list[j]
        Feature_valid = Feature_valid_list[j]
        Label_valid = Label_valid_list[j]
        Label_valid.ravel()

        xgb = xgboost.XGBClassifier(**BayesOp_Parameters)
        xgb.fit(Feature_train, Label_train.ravel())
        Label_predict = xgb.predict(Feature_valid)
        ml_record.measure(i, Label_valid, Label_predict, 'weighted')
        i += 1

    result = ml_record.mean_G()

    return result

save_path = "KEEL_Cross_Folder_XGBoost_Para_CNPs"

for i_test in range(10):
    for Dir in DIRS:
        print("Data Set Name: ", Dir)
        dir_path = PATH + "/" + Dir
        files = os.listdir(dir_path)  # Get files in the folder
        list_path = save_path + '/CNPs_Timeline_Record_Validation_' + str(i_test) + '/' + Dir
        if not os.path.exists(list_path):
            os.makedirs(list_path)
        para_path = save_path + '/CNPs_Validation_' + str(i_test) + '/' + Dir
        if not os.path.exists(para_path):
            os.makedirs(para_path)

        for file in files:
            name = dir_path + '/' + file
            print("Data Set Folder: ", file)
            plog = PrintLog(para_keys)

            cnps = CNPs_Optimization(evaluate, parameters)
            cnps.maximize(num_iter=2, pop_size=3, uncertain_rate=0.2)
            time_list = cnps.timestamps_list
            target_list = cnps.target_list
            para_list = cnps.parameters_list
            max_val = max(target_list)
            target_index = target_list.index(max_val)
            max_params = para_list[target_index]

            sub_name = file.split(".")[0]

            with open('CNPs_Opt_G_Mean_Validation_Records_' + str(i_test) + '.txt', 'a') as w:
                line = sub_name + '\t' + str(max_val) + '\n'
                w.write(line)

            para_file = para_path + '/' + sub_name + '_CNPs.json'
            Output_Parameters = dict(zip(para_keys, max_params))
            with open(para_file, 'a') as outfile:
                json.dump(Output_Parameters, outfile, ensure_ascii=False)
                outfile.write('\n')

            list_file = list_path + '/' + sub_name + '_CNPs_List.json'
            Output_line = {}
            for i in range(len(target_list)):
                Output_line[time_list[i]] = {target_list[i]: para_list[i]}
            with open(list_file, 'a') as outfile:
                json.dump(Output_line, outfile, ensure_ascii=False)
                outfile.write('\n')



