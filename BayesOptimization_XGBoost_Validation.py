import os
import numpy as np
import xgboost
import json
import warnings
from metrics_list import MetricList
from bayes_opt import BayesianOptimization


warnings.filterwarnings('ignore')
PATH = "KEEL_Cross_Folder_Valid_npz"
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
#        Num_train = Feature_train_list[j].shape[0]

        Feature_valid = Feature_valid_list[j]
        Label_valid = Label_valid_list[j]
        Label_valid.ravel()
#        Num_valid = Feature_valid_list[j].shape[0]

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



for Dir in DIRS:
    print("Data Set Name: ", Dir)
    dir_path = PATH + "/" + Dir
    files = os.listdir(dir_path)  # Get files in the folder
    for file in files:
        name = dir_path + '/' + file

        xgboostBO = BayesianOptimization(xgboostcv,
                                         {'max_depth': (5, 10),
                                          'learning_rate': (0.01, 0.3),
                                          'n_estimators': (50, 1000),
                                          'gamma': (0.01, 1.),
                                          'min_child_weight': (2, 10),
                                          'max_delta_step': (0, 0.1),
                                          'subsample': (0.6, 0.9),
                                          'colsample_bytree': (0.5, 0.99)
                                          })

        xgboostBO.maximize(init_points=50, n_iter=200)
        # csv_file = Dir + '.csv'
        # xgboostBO.points_to_csv(csv_file)
        #    xgboostBO.maximize()

        print('-' * 53)
        print('Final Results')
        print('XGBoost: %f Parameters: %s' % (xgboostBO.res['max']['max_val'], xgboostBO.res['max']['max_params']))

        sub_name = file.split(".")[0]

        with open('BayesOpt_G_Mean_Validation.txt', 'a') as w:
            line = sub_name + '\t' + str(xgboostBO.res['max']['max_val']) + '\n'
            w.write(line)

        with open(sub_name + '_BaOp.json', 'a') as outfile:
            json.dump(xgboostBO.res['max']['max_params'], outfile, ensure_ascii=False)
            outfile.write('\n')



