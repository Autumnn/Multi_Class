import os
import json


path = "KEEL_Cross_Folder_npz"
dirs = os.listdir(path) #Get files in the folder

for Dir in dirs:
    Esemble_Path = 'KEEL_Cross_Folder_XGBoost_Para_From_GA_Esemble'
    Esemble_File = Dir + '_Esemble.json'
    Esemble_Dir = Esemble_Path + "/" + Esemble_File
    with open(Esemble_Dir, 'r') as Esemble_data:
        Esemble_Parameters = json.load(Esemble_data)

    print(str(Dir))
    print(len(Esemble_Parameters))
    print(Esemble_Parameters)

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
        for j in range(3 - len(pool_classifiers)):
            pool_classifiers.append(pool_classifiers[j])


    pool_classifiers = []
    for metric_value, parameters in Esemble_Parameters.items():
        parameters['silent'] = True
        parameters['nthread'] = -1
        parameters['seed'] = 1234
        #    BayesOp_Parameters['objective'] = "multi:softprob"
        parameters['max_depth'] = int(parameters['max_depth'])
        parameters['n_estimators'] = int(parameters['n_estimators'])
'''