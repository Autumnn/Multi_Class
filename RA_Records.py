import os
import numpy as np
import xgboost
import json
import warnings
from metrics_list import MetricList
from datetime import datetime
from print_log import PrintLog

warnings.filterwarnings('ignore')
PATH = "KEEL_Cross_Folder_npz"
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
    Num_Cross_Folders = 5
    ml_record = MetricList(Num_Cross_Folders)
    i = 0
    for file in files:
        name = dir_path + '/' + file
        r = np.load(name)

        Feature_train = r['F_tr']
        Label_train = r['L_tr']
        Num_train = Feature_train.shape[0]
        # print(Num_train)
        Feature_test = r['F_te']
        Label_test = r['L_te']
        Label_test.ravel()
        Num_test = Feature_test.shape[0]
        # print(Num_test)

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
        Label_predict = xgb.predict(Feature_test)

        ml_record.measure(i, Label_test, Label_predict, 'weighted')
        i += 1

    return ml_record.mean_G()


parameters_bounds = {'max_depth': (3, 10),         # int        ###
              'learning_rate': (0.01, 0.3),         ###
              'n_estimators': (50, 1000),       #int
              'gamma': (0, 1.),                     ###
              'min_child_weight': (1, 10),          ###
              'max_delta_step': (0, 0.3),           ###
              'subsample': (0.5, 1),                ###
              'colsample_bytree': (0.5, 1)}         ###

def random_search(f, para_b, num):
    begin_time = datetime.now()
    Timestamps_list = []
    Target_list = []
    keys = list(para_b.keys())
    bounds = np.array(list(para_b.values()), dtype=np.float)
    dim = len(keys)
    plog = PrintLog(keys)
    parameters = np.empty((num, dim))
    plog.print_header(initialization=True)
    for col, (lower, upper) in enumerate(bounds):
        parameters.T[col] = np.random.RandomState().uniform(lower, upper, size=num)
    plog.print_header(initialization=False)

    for i in range(num):
        params_dic = dict(zip(keys, parameters[i]))
        metric = f(**params_dic)
        Target_list.append(metric)
        elapse_time = (datetime.now() - begin_time).total_seconds()
        Timestamps_list.append(elapse_time)
        plog.print_step(parameters[i], metric)

    return Timestamps_list, Target_list, parameters.tolist()


for Dir in DIRS:
    print("Data Set Name: ", Dir)
    dir_path = PATH + "/" + Dir
    files = os.listdir(dir_path)  # Get files in the folder
    time_list, target_list, para_list = random_search(evaluation, parameters_bounds, 1000)

    Output_Para = []
    for k, v in dict(zip(target_list, para_list)).items():
        Output_Para.append({k: v})
    Output_line = dict(zip(time_list, Output_Para))
    with open(Dir + '_RA.json', 'a') as outfile:
        json.dump(Output_line, outfile, ensure_ascii=False)
        outfile.write('\n')




