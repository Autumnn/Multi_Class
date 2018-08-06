import os
import numpy as np
import xgboost
import json
import warnings
from metrics_list import MetricList
from bayes_opt import BayesianOptimization


warnings.filterwarnings('ignore')
PATH = "KEEL_Cross_Folder_npz"
DIRS = os.listdir(PATH)         #   Get files in the folder


def xgboostcv(max_depth,
              learning_rate,
              n_estimators,
              gamma,
              min_child_weight,
              max_delta_step,
              subsample,
              colsample_bytree,
              silent =True,
              nthread = -1,
              seed = 1234,
              ):
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



for Dir in DIRS:
    print("Data Set Name: ", Dir)
    dir_path = PATH + "/" + Dir
    files = os.listdir(dir_path)  # Get files in the folder

    xgboostBO = BayesianOptimization(xgboostcv,
                                     {'max_depth': (5, 10),
                                      'learning_rate': (0.01, 0.3),
                                      'n_estimators': (50, 1000),
                                      'gamma': (1., 0.01),
                                      'min_child_weight': (2, 10),
                                      'max_delta_step': (0, 0.1),
                                      'subsample': (0.7, 0.8),
                                      'colsample_bytree': (0.5, 0.99)
                                      })

    xgboostBO.maximize()

    print('-' * 53)
    print('Final Results')
    print('SVC: %f Parameters: %s' % (xgboostBO.res['max']['max_val'], xgboostBO.res['max']['max_params']))

    with open(Dir + '.json', 'a') as outfile:
        json.dump(xgboostBO.res['max']['max_params'], outfile, ensure_ascii=False)
        outfile.write('\n')



