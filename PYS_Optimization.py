import os
import numpy as np
import xgboost
import json
import warnings
import pyswarms as ps
from pyswarms.single import GlobalBestPSO
from metrics_list import MetricList


warnings.filterwarnings('ignore')
PATH = "KEEL_Cross_Folder_npz"
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
    BayesOp_Parameters = dict(zip(keys, para_value))
    BayesOp_Parameters['silent'] = True
    BayesOp_Parameters['nthread'] = -1
    BayesOp_Parameters['seed'] = 0
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

    return 1-ml_record.mean_G()



for Dir in DIRS:
    print("Data Set Name: ", Dir)
    dir_path = PATH + "/" + Dir
    files = os.listdir(dir_path)  # Get files in the folder

    pys_op = GlobalBestPSO(n_particles=20, dimensions=len(para_keys), options=parameters)

    xgboostBO.maximize(init_points=50, n_iter=200)
    # csv_file = Dir + '.csv'
    # xgboostBO.points_to_csv(csv_file)
    #    xgboostBO.maximize()

    print('-' * 53)
    print('Final Results')
    print('XGBoost: %f Parameters: %s' % (xgboostBO.res['max']['max_val'], xgboostBO.res['max']['max_params']))

    with open('BayesOpt_G_Mean.txt', 'a') as w:
        line = Dir + '\t' + str(xgboostBO.res['max']['max_val']) + '\n'
        w.write(line)

    with open(Dir + '.json', 'a') as outfile:
        json.dump(xgboostBO.res['max']['max_params'], outfile, ensure_ascii=False)
        outfile.write('\n')

    time_list = xgboostBO.res['all']['timestamp']
    target_list = xgboostBO.res['all']['values']
    para_list = xgboostBO.res['all']['params']
    Output_Para = []
    for k, v in dict(zip(target_list, para_list)).items():
        Output_Para.append({k: v})
    Output_line = dict(zip(time_list, Output_Para))
    with open(Dir + '_BA.json', 'a') as outfile:
        json.dump(Output_line, outfile, ensure_ascii=False)
        outfile.write('\n')

