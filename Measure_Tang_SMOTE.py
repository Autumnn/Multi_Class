import os
import numpy as np
import xgboost
import json
from deslib.dcs.mcb import MCB
from deslib.des.meta_des import METADES
from deslib.des.des_MI import DESMI
from metrics_list import MetricList
from sklearn.metrics import confusion_matrix
import warnings

warnings.filterwarnings('ignore')

Op_Path = 'Tang_SMOTE_Result_and_Para/GA_Validation'


def classifiers_generate(data_file):
    pool_classifiers = []
    para_file = Op_Path + '/' + data_file.split('.')[0] + '_SMOTE_GA.json'
    with open(para_file, 'r') as OP_data:
        Op_Parameters_dic = json.load(OP_data)

    for key, val in Op_Parameters_dic.items():
        Op_Parameters = val
        Op_Parameters['silent'] = True
        Op_Parameters['nthread'] = -1
        Op_Parameters['seed'] = 0
        #    BayesOp_Parameters['objective'] = "multi:softprob"
        Op_Parameters['max_depth'] = int(Op_Parameters['max_depth'])
        Op_Parameters['n_estimators'] = int(Op_Parameters['n_estimators'])
        pool_classifiers.append(xgboost.XGBClassifier(**Op_Parameters))

    return pool_classifiers

path = "Tang_npz"
dir_path = "Tang_SMOTE_npz"
files = os.listdir(path)  # Get files in the folder

methods = ["XGBoost-GA", "META-DES-XGBoost", "MCB-XGBoost", "DES-MI-XGBoost"]
for m in methods:
    print('Method: ', m)
    Num_Cross_Folders = 5
    ml_record = MetricList(Num_Cross_Folders)
    i = 0
    for file in files:
        name = path + '/' + file
        smote_name = dir_path + '/' + file.split('.')[0] + '_SMOTE.npz'
        r = np.load(name)
        r_smote = np.load(smote_name)

        SMOTE_feature_train_list = r_smote['S_F_tr_l']
        SMOTE_label_train_list = r_smote['S_L_tr_l']
        SMOTE_feature_valid_list = r_smote['S_F_va_l']
        SMOTE_label_valid_list = r_smote['S_L_va_l']

        Feature_train = r['F_tr']
        Label_train = r['L_tr']
        Num_train = Feature_train.shape[0]
        #print(Num_train)
        Feature_test = r['F_te']
        Label_test = r['L_te']
        Label_test.ravel()
        Num_test = Feature_test.shape[0]
        #print(Num_test)
        print(i, " folder; ", "Number of test: ", Num_test)

        if m == 'XGBoost-GA':
            para_dir = 'Tang_Results_and_Para/Tang.json'
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

        else:
            num_classifiers_in_pool = 100
            pool_classifiers = classifiers_generate(file)
            for k in range(num_classifiers_in_pool):
                #print("Data Set Folder: ", file, ", SMOTE folder id: ", str(k))
                Feature_train_smote = np.concatenate((SMOTE_feature_train_list[k][0], SMOTE_feature_valid_list[k][0]))
                Label_train_smote = np.concatenate((SMOTE_label_train_list[k][0], SMOTE_label_valid_list[k][0]))
                pool_classifiers[k].fit(Feature_train_smote, Label_train_smote.ravel())

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

        print(confusion_matrix(Label_test, Label_predict))

        ml_record.measure(i, Label_test, Label_predict, 'weighted')
        i += 1

    file_wirte = "Result_Esemble_Tang.txt"
    ml_record.output(file_wirte, m, 'Tang')



