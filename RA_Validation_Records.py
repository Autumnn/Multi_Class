import os
import numpy as np
import xgboost
import json
import warnings
from metrics_list import MetricList
from datetime import datetime
from print_log import PrintLog

warnings.filterwarnings('ignore')
PATH = "KEEL_Cross_Folder_Valid_npz"
DIRS = os.listdir(PATH)         #   Get files in the folder


def evaluation(max_depth,
              learning_rate,
              n_estimators,
              gamma,
              min_child_weight,
              max_delta_step,
              subsample,
              colsample_bytree,
              silent =True,
              nthread = -1,
              seed = 0,):

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

        xgb = xgboost.XGBClassifier(max_depth = int(max_depth),
                                    learning_rate = learning_rate,
                                    n_estimators = int(n_estimators),
                                    silent = silent,
                                    nthread = nthread,
                                    gamma = gamma,
                                    min_child_weight = min_child_weight,
                                    max_delta_step = max_delta_step,
                                    subsample = subsample,
                                    colsample_bytree = colsample_bytree,
                                    seed = seed,
                                    objective = "multi:softprob")
        xgb.fit(Feature_train, Label_train.ravel())
        Label_predict = xgb.predict(Feature_valid)
        ml_record.measure(i, Label_valid, Label_predict, 'weighted')
        i += 1

    return ml_record.mean_G()


parameters = {'max_depth': (3, 10),         # int        ###
              'learning_rate': (0.01, 0.3),         ###
              'n_estimators': (50, 1000),       #int
              'gamma': (0, 1.),                     ###
              'min_child_weight': (1, 10),          ###
              'max_delta_step': (0, 0.3),           ###
              'subsample': (0.5, 1),                ###
              'colsample_bytree': (0.5, 1)}         ###
para_bounds = np.array(list(parameters.values()), dtype=np.float)
para_keys = list(parameters.keys())


def random_search(f, para_b, num):
    begin_time = datetime.now()
    Timestamps_list = []
    Target_list = []
    keys = list(para_b.keys())
    bounds = np.array(list(para_b.values()), dtype=np.float)
    dim = len(keys)
    plog = PrintLog(keys)
    parameter_list = np.empty((num, dim))
    plog.print_header(initialization=True)
    for col, (lower, upper) in enumerate(bounds):
        parameter_list.T[col] = np.random.RandomState().uniform(lower, upper, size=num)
    plog.print_header(initialization=False)

    for i in range(num):
        params_dic = dict(zip(keys, parameter_list[i]))
        metric = f(**params_dic)
        Target_list.append(metric)
        elapse_time = (datetime.now() - begin_time).total_seconds()
        Timestamps_list.append(elapse_time)
        plog.print_step(parameter_list[i], metric)

    return Timestamps_list, Target_list, parameter_list.tolist()


for Dir in DIRS:
    print("Data Set Name: ", Dir)
    dir_path = PATH + "/" + Dir
    files = os.listdir(dir_path)  # Get files in the folder

    for file in files:
        name = dir_path + '/' + file
        print("Data Set Folder: ", file)

        time_list, target_list, para_list = random_search(evaluation, parameters, 100)
        max_val = max(target_list)
        target_index = target_list.index(max_val)
        max_params = para_list[target_index]

        sub_name = file.split(".")[0]

        Output_Para = []
        for k, v in dict(zip(target_list, para_list)).items():
            Output_Para.append({k: v})
        Output_line = dict(zip(time_list, Output_Para))
        with open(sub_name + '_RA_List.json', 'a') as outfile:
            json.dump(Output_line, outfile, ensure_ascii=False)
            outfile.write('\n')

        with open('RA_Opt_G_Mean_Validation_Records.txt', 'a') as w:
            line = sub_name + '\t' + str(max_val) + '\n'
            w.write(line)

        Output_Parameters = dict(zip(para_keys, max_params))
        with open(sub_name + '_RA.json', 'a') as outfile:
            json.dump(Output_Parameters, outfile, ensure_ascii=False)
            outfile.write('\n')




