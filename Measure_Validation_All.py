import os
import numpy as np
import xgboost
import json
import warnings
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from deslib.dcs.mcb import MCB
from deslib.des.meta_des import METADES
from deslib.des.des_MI import DESMI
from imblearn.over_sampling import SMOTE
from metrics_list import MetricList

warnings.filterwarnings('ignore')
path = "KEEL_Cross_Folder_npz_S"
dirs = os.listdir(path) #Get files in the folder
#GA_Path = 'KEEL_Cross_Folder_XGBoost_Para_From_GA_Validation'
#RA_Path = 'KEEL_Cross_Folder_XGBoost_Para_From_RA_Validation'
Para_Path = 'KEEL_Cross_Folder_XGBoost_Para/Server/'

for Dir in dirs:
    print("Data Set Name: ", Dir)
    dir_path = path + "/" + Dir
    files = os.listdir(dir_path)  # Get files in the folder

    methods = ["AdaBoost-DT", "META-DES", "MCB", "DES-MI", "XGBoost",
               "XGBoost-RA", "XGBoost-BA", "XGBoost-GA", "XGBoost-CMA", "XGBoost-PYS", "XGBoost-All"]
    for m in methods:
        print('Method: ', m)
        Num_Cross_Folders = 5
        ml_record = MetricList(Num_Cross_Folders)
        i = 0
        for file in files:
            name = dir_path + '/' + file
            r = np.load(name)

            Feature_train = r['F_tr']
            Label_train = r['L_tr']
            Num_train = Feature_train.shape[0]
            #print(Num_train)
            Feature_test = r['F_te']
            Label_test = r['L_te']
            Label_test.ravel()
            Num_test = Feature_test.shape[0]
            #print(Num_test)
            print(i, " folder; ", "Number of train: ", Num_train, "Number of test: ", Num_test)

            if m == 'AdaBoost-DT':
                bdt = AdaBoostClassifier(DecisionTreeClassifier(max_depth=2, min_samples_split=20, min_samples_leaf=5),
                                         algorithm='SAMME', n_estimators=200, learning_rate=0.8)
                bdt.fit(Feature_train, Label_train.ravel())
                Label_predict = bdt.predict(Feature_test)
            elif m == 'META-DES':
                pool_classifiers = RandomForestClassifier(n_estimators=10)
                pool_classifiers.fit(Feature_train, Label_train.ravel())
                metades = METADES(pool_classifiers)
                metades.fit(Feature_train, Label_train.ravel())
                Label_predict = metades.predict(Feature_test)
            elif m == 'MCB':
                pool_classifiers = RandomForestClassifier(n_estimators=10)
                pool_classifiers.fit(Feature_train, Label_train.ravel())
                mcb = MCB(pool_classifiers)
                mcb.fit(Feature_train, Label_train.ravel())
                Label_predict = mcb.predict(Feature_test)
            elif m == 'DES-MI':
                pool_classifiers = RandomForestClassifier(n_estimators=10)
                pool_classifiers.fit(Feature_train, Label_train.ravel())
                dmi = DESMI(pool_classifiers)
                dmi.fit(Feature_train, Label_train.ravel())
                Label_predict = dmi.predict(Feature_test)
            elif m == 'XGBoost':
                xgb = xgboost.XGBClassifier()
                # xgb = xgboost.XGBClassifier(max_depth=5, learning_rate=0.01, n_estimators=50, gamma=0.01,
                #                             min_child_weight=5, max_delta_step=0, subsample=0.7,
                #                             colsample_bytree=0.5, silent=True, nthread=-1,seed=1234)
                xgb.fit(Feature_train, Label_train.ravel())
                Label_predict = xgb.predict(Feature_test)
            else:
                sub_name = file.split('.')[0]
                algorithm = m.split('-')[-1]
                if m != 'XGBoost-All':
                    para_file = sub_name + '_' + str(algorithm) + '.json'
                else:
                    para_file = sub_name + '.json'
                para_dir = Para_Path + algorithm + '_use/' + Dir + "/" + para_file
                with open(para_dir, 'r') as op_data:
                    Op_Parameters = json.load(op_data)
                Op_Parameters['silent'] = True
                Op_Parameters['nthread'] = -1
                Op_Parameters['seed'] = 0
                #    BayesOp_Parameters['objective'] = "multi:softprob"
                Op_Parameters['max_depth'] = int(Op_Parameters['max_depth'])
                Op_Parameters['n_estimators'] = int(Op_Parameters['n_estimators'])

                xgb = xgboost.XGBClassifier(**Op_Parameters)
                xgb.fit(Feature_train, Label_train.ravel())
                Label_predict = xgb.predict(Feature_test)


            ml_record.measure(i, Label_test, Label_predict, 'weighted')
            i += 1

        file_wirte = "Result_Validation_Op_Compare_S.txt"
        ml_record.output(file_wirte, m, Dir)



