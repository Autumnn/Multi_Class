import os
import numpy as np
import xgboost
import json
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.multiclass import OneVsRestClassifier
from deslib.dcs.mcb import MCB
from deslib.des.meta_des import METADES
from deslib.des.des_MI import DESMI
from imblearn.over_sampling import SMOTE
from metrics_list import MetricList

path = "KEEL_Cross_Folder_npz"
dirs = os.listdir(path) #Get files in the folder

for Dir in dirs:
    print("Data Set Name: ", Dir)
    dir_path = path + "/" + Dir
    files = os.listdir(dir_path)  # Get files in the folder

    BayesOp_Path = 'KEEL_Cross_Folder_XGBoost_Para_Used'
    BayesOp_File = Dir + '_GA.json'
    BayesOp_Dir = BayesOp_Path + "/" + BayesOp_File
    with open(BayesOp_Dir, 'r') as BayesOP_data:
        BayesOp_Parameters = json.load(BayesOP_data)
    BayesOp_Parameters['silent'] = True
    BayesOp_Parameters['nthread'] = -1
    BayesOp_Parameters['seed'] = 1234
#    BayesOp_Parameters['objective'] = "multi:softprob"
    BayesOp_Parameters['max_depth'] = int(BayesOp_Parameters['max_depth'])
    BayesOp_Parameters['n_estimators'] = int(BayesOp_Parameters['n_estimators'])

    Esemble_Path = 'KEEL_Cross_Folder_XGBoost_Para_From_GA_Esemble'
    Esemble_File = Dir + '_Esemble.json'
    Esemble_Dir = Esemble_Path + "/" + Esemble_File
    with open(Esemble_Dir, 'r') as Esemble_data:
        Esemble_Parameters = json.load(Esemble_data)

    sorted_metric_values = sorted(Esemble_Parameters.keys(), reverse=True)
    pool_classifiers = []
    for i in range(3):
        metric_value = sorted_metric_values[i]
        parameters = Esemble_Parameters[metric_value]

        parameters['silent'] = True
        parameters['nthread'] = -1
        parameters['seed'] = 1234
        #    BayesOp_Parameters['objective'] = "multi:softprob"
        parameters['max_depth'] = int(parameters['max_depth'])
        parameters['n_estimators'] = int(parameters['n_estimators'])
        pool_classifiers.append(xgboost.XGBClassifier(**parameters))
    if len(pool_classifiers) < 3:
        for j in range(3-len(pool_classifiers)):
            pool_classifiers.append(pool_classifiers[j])


    methods = ["XGBoost", "META-DES-XGBoost", "MCB-XGBoost", "DES-MI-XGBoost"]
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

            if m == 'XGBoost':
                xgb = xgboost.XGBClassifier(**BayesOp_Parameters)
                xgb.fit(Feature_train, Label_train.ravel())
                Label_predict = xgb.predict(Feature_test)
            else:
                for classifier in pool_classifiers:
                    classifier.fit(Feature_train, Label_train.ravel())

                if m == 'META-DES-XGBoost':
                    metades = METADES(pool_classifiers)
                    metades.fit(Feature_train, Label_train.ravel())
                    Label_predict = metades.predict(Feature_test)
                elif m == 'MCB-XGBoost':
                    mcb = MCB(pool_classifiers)
                    mcb.fit(Feature_train, Label_train.ravel())
                    Label_predict = mcb.predict(Feature_test)
                elif m == 'DES-MI-XGBoost':
                    dmi = DESMI(pool_classifiers)
                    dmi.fit(Feature_train, Label_train.ravel())
                    Label_predict = dmi.predict(Feature_test)

            ml_record.measure(i, Label_test, Label_predict, 'weighted')
            i += 1

        file_wirte = "Result_One_vs_All_GAOp_XGBoost_G_mean_Esemble_test.txt"
        ml_record.output(file_wirte, m, Dir)



