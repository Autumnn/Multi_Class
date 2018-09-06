import os
import numpy as np
import xgboost
import json
from deslib.dcs.mcb import MCB
from deslib.des.meta_des import METADES
from deslib.des.des_MI import DESMI
from metrics_list import MetricList
import warnings

warnings.filterwarnings('ignore')
path = "KEEL_Cross_Folder_npz_S"
dirs = os.listdir(path) #Get files in the folder

algorithm_list = ['RA', 'BA', 'GA', 'CMA', 'PYS']
Op_Path = 'KEEL_Cross_Folder_XGBoost_Para/Server/'


def classifiers_generate(data_file):
    pool_classifiers = []
    data_set = data_file.split('_')[0]
    for alg in algorithm_list:
        para_path = Op_Path + alg
        para_dirs = os.listdir(para_path)
        for subpara_dir in para_dirs:
            para_file = para_path + '/' + subpara_dir + '/' + data_set + '/' + data_file.split('.')[0] + '_' + alg + '.json'
            print(para_file)
            with open(para_file, 'r') as OP_data:
                Op_Parameters = json.load(OP_data)

            Op_Parameters['silent'] = True
            Op_Parameters['nthread'] = -1
            Op_Parameters['seed'] = 0
            #    BayesOp_Parameters['objective'] = "multi:softprob"
            Op_Parameters['max_depth'] = int(Op_Parameters['max_depth'])
            Op_Parameters['n_estimators'] = int(Op_Parameters['n_estimators'])
            pool_classifiers.append(xgboost.XGBClassifier(**Op_Parameters))

    return pool_classifiers

for Dir in dirs:
    print("Data Set Name: ", Dir)
    dir_path = path + "/" + Dir
    files = os.listdir(dir_path)  # Get files in the folder

    methods = ["XGBoost", "XGBoost-GA", "META-DES-XGBoost", "MCB-XGBoost", "DES-MI-XGBoost"]
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
                xgb = xgboost.XGBClassifier()
                xgb.fit(Feature_train, Label_train.ravel())
                Label_predict = xgb.predict(Feature_test)
            elif m == 'XGBoost-GA':
                sub_name = file.split('.')[0]
                # print(sub_name)
                RA_File = sub_name + '_GA.json'
                RA_Dir = Op_Path + "/GA_use/" + Dir + "/" + RA_File
                with open(RA_Dir, 'r') as RA_OP_data:
                    RaOp_Parameters = json.load(RA_OP_data)
                RaOp_Parameters['silent'] = True
                RaOp_Parameters['nthread'] = -1
                RaOp_Parameters['seed'] = 0
                #    BayesOp_Parameters['objective'] = "multi:softprob"
                RaOp_Parameters['max_depth'] = int(RaOp_Parameters['max_depth'])
                RaOp_Parameters['n_estimators'] = int(RaOp_Parameters['n_estimators'])

                xgb = xgboost.XGBClassifier(**RaOp_Parameters)
                xgb.fit(Feature_train, Label_train.ravel())
                Label_predict = xgb.predict(Feature_test)
            else:
                pool_classifiers = classifiers_generate(file)
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

        file_wirte = "Result_Esemble_test_S.txt"
        ml_record.output(file_wirte, m, Dir)



