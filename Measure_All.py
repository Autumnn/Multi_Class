import os
import numpy as np
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from deslib.dcs.mcb import MCB
from deslib.des.meta_des import METADES
from imblearn.over_sampling import SMOTE
from metrics_list import MetricList

path = "KEEL_Cross_Folder_npz"
dirs = os.listdir(path) #Get files in the folder

for Dir in dirs:
    print("Data Set Name: ", Dir)
    dir_path = path + "/" + Dir
    files = os.listdir(dir_path)  # Get files in the folder

    methods = ["AdaBoost-DT", "SMOTE-AdaBoost-DT", "DESlib"]
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
            elif m == 'SMOTE-AdaBoost-DT':
                sm = SMOTE()
                Feature_train_o, Label_train_o = sm.fit_sample(Feature_train, Label_train.ravel())
                bdt = AdaBoostClassifier(DecisionTreeClassifier(max_depth=2, min_samples_split=20, min_samples_leaf=5),
                                         algorithm='SAMME', n_estimators=200, learning_rate=0.8)
                bdt.fit(Feature_train_o, Label_train_o)
                Label_predict = bdt.predict(Feature_test)
            elif m == 'DESlib':
                pool_classifiers = RandomForestClassifier(n_estimators=10)
                pool_classifiers.fit(Feature_train, Label_train.ravel())
                # Initialize the DS model
                #metades = METADES(pool_classifiers)
                mcb = MCB(pool_classifiers)
                # Preprocess the Dynamic Selection dataset (DSEL)
                #metades.fit(X_dsel, y_dsel)
                mcb.fit(Feature_train, Label_train.ravel())
                # Predict new examples:
                #metades.predict(X_test)
                Label_predict = mcb.predict(Feature_test)


            ml_record.measure(i, Label_test, Label_predict, 'weighted')
            i += 1

        file_wirte = "Result_All.txt"
        ml_record.output(file_wirte, m, Dir)



