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

    BayesOp_Path = 'KEEL_Cross_Folder_XGBoost_Para_From_BayesOP_120_iter'
    BayesOp_File = Dir + '.json'
    BayesOp_Dir = BayesOp_Path + "/" + BayesOp_File
    with open(BayesOp_Dir, 'r') as BayesOP_data:
        BayesOp_Parameters = json.load(BayesOP_data)
    BayesOp_Parameters['silent'] = True
    BayesOp_Parameters['nthread'] = -1
    BayesOp_Parameters['seed'] = 1234
#    BayesOp_Parameters['objective'] = "multi:softprob"
    BayesOp_Parameters['max_depth'] = int(BayesOp_Parameters['max_depth'])
    BayesOp_Parameters['n_estimators'] = int(BayesOp_Parameters['n_estimators'])

    methods = ["XGBoost", "SMOTE-XGBoost", "AdaBoost-DT", "SMOTE-AdaBoost-DT", "META-DES", "MCB", "DES-MI", "One_vs_Rest-SMOTE-XGBoost", "One_vs_Rest-XGBoost"]
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
            elif m == 'SMOTE-XGBoost':
                sm = SMOTE()
                Feature_train_o, Label_train_o = sm.fit_sample(Feature_train, Label_train.ravel())
                sgb = xgboost.XGBClassifier(**BayesOp_Parameters)
                sgb.fit(Feature_train_o, Label_train_o)
                Label_predict = sgb.predict(Feature_test)
            elif m == 'AdaBoost-DT':
                bdt = AdaBoostClassifier(DecisionTreeClassifier(max_depth=2, min_samples_split=20, min_samples_leaf=5),
                                         algorithm='SAMME', n_estimators=200, learning_rate=0.8)
                bdt.fit(Feature_train, Label_train.ravel())
                Label_predict = bdt.predict(Feature_test)
            elif m == 'SMOTE-AdaBoost-DT':
                sm = SMOTE()
                Feature_train_o, Label_train_o = sm.fit_sample(Feature_train, Label_train.ravel())
                bdt = AdaBoostClassifier(DecisionTreeClassifier(max_depth=2, min_samples_split=20, min_samples_leaf=5),
                                         algorithm='SAMME', n_estimators=200, learning_rate=0.8)
                bdt.fit(Feature_train_o, Label_train_o)
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
            elif m == 'One_vs_Rest-SMOTE-XGBoost':
                sm = SMOTE()
                Feature_train_o, Label_train_o = sm.fit_sample(Feature_train, Label_train.ravel())
                clf = OneVsRestClassifier(xgboost.XGBClassifier(**BayesOp_Parameters))
                clf.fit(Feature_train_o, Label_train_o)
                Label_predict = clf.predict(Feature_test)
            elif m == 'One_vs_Rest-XGBoost':
                clf = OneVsRestClassifier(xgboost.XGBClassifier(**BayesOp_Parameters))
                clf.fit(Feature_train, Label_train.ravel())
                Label_predict = clf.predict(Feature_test)

            ml_record.measure(i, Label_test, Label_predict, 'weighted')
            i += 1

        file_wirte = "Result_One_vs_All_BayOp_XGBoost_G_mean_120_iter.txt"
        ml_record.output(file_wirte, m, Dir)



