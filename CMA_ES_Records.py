import os
import numpy as np
import xgboost
import json
import warnings
import cma
from metrics_list import MetricList
from datetime import datetime
from print_log import PrintLog

warnings.filterwarnings('ignore')
#PATH = "KEEL_Cross_Folder_npz"
PATH = "KEEL_Cross_Folder_npz_S"
DIRS = os.listdir(PATH)         #   Get files in the folder

parameters = {'max_depth': (5, 10),         # int
              'learning_rate': (0.01, 0.3),
              'n_estimators': (50, 1000),       #int
              'gamma': (0.01, 1.),
              'min_child_weight': (2, 10),
              'max_delta_step': (0, 0.1),
              'subsample': (0.7, 0.8),
              'colsample_bytree': (0.5, 0.99)}
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
    BayesOp_Parameters['seed'] = 1234
    #    BayesOp_Parameters['objective'] = "multi:softprob"
    BayesOp_Parameters['max_depth'] = int(BayesOp_Parameters['max_depth'])
    BayesOp_Parameters['n_estimators'] = int(BayesOp_Parameters['n_estimators'])
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

        xgb = xgboost.XGBClassifier(**BayesOp_Parameters)
        xgb.fit(Feature_train, Label_train.ravel())
        Label_predict = xgb.predict(Feature_test)
        ml_record.measure(i, Label_test, Label_predict, 'weighted')
        i += 1

    result = 1 - ml_record.mean_G()

    return result


def evolution_search(f, para_b):
    begin_time = datetime.now()
    Timestamps_list = []
    Target_list = []
    Parameters_list = []
    keys = list(para_b.keys())
    bounds = np.array(list(para_b.values()), dtype=np.float)
    dim = len(keys)
    plog = PrintLog(keys)
    para_value = np.empty((1, dim))
    plog.print_header(initialization=True)
    for col, (lower, upper) in enumerate(bounds):
        para_value.T[col] = np.random.RandomState().uniform(lower, upper)
    para_value = para_value.ravel().tolist()
    plog.print_header(initialization=False)

    es = cma.CMAEvolutionStrategy(para_value, 0.2, {'maxiter': 60, 'popsize': 50})
#    es = cma.CMAEvolutionStrategy(para_value, 0.5)
    while not es.stop():
        solutions = es.ask()
        es.tell(solutions, [f(x) for x in solutions])
#        es.tell(*es.ask_and_eval(f))
#        es.disp()
        res = es.result
#        metric = f(**params_dic)
        Parameters_list.append(res[0].tolist())
        Target_list.append(1-res[1])
        elapse_time = (datetime.now() - begin_time).total_seconds()
        Timestamps_list.append(elapse_time)
#        print("The best candidate: ", res[0])
#        print("The best result: ", res[1])
        plog.print_step(res[0], 1-res[1])

    return Timestamps_list, Target_list, Parameters_list


for Dir in DIRS:
    print("Data Set Name: ", Dir)
    dir_path = PATH + "/" + Dir
    files = os.listdir(dir_path)  # Get files in the folder

    time_list, target_list, para_list = evolution_search(evaluate, parameters)
    Output_line = {}
    for i in range(len(target_list)):
        Output_line[time_list[i]] = {target_list[i]: para_list[i]}
    with open(Dir + '_CMA.json', 'a') as outfile:
        json.dump(Output_line, outfile, ensure_ascii=False)
        outfile.write('\n')



