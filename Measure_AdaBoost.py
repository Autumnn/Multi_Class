import os
import numpy as np
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
from imblearn.over_sampling import SMOTE
from metrics_list import MetricList

path = "KEEL_Cross_Folder_npz"
dirs = os.listdir(path) #Get files in the folder

for Dir in dirs:
    print("Data Set Name: ", Dir)
    dir_path = path + "/" + Dir
    files = os.listdir(dir_path)  # Get files in the folder

    methods = ["AdaBoost-DT", "SMOTE"]
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
                Feature_train_o = Feature_train
                Label_train_o = Label_train.ravel()
            elif m == 'SMOTE':
                sm = SMOTE()
                Feature_train_o, Label_train_o = sm.fit_sample(Feature_train, Label_train.ravel())

            bdt = AdaBoostClassifier(DecisionTreeClassifier(max_depth=2, min_samples_split=20, min_samples_leaf=5),
                                 algorithm='SAMME', n_estimators=200, learning_rate=0.8)
            bdt.fit(Feature_train_o, Label_train_o)

            Label_predict = bdt.predict(Feature_test)
#            print(metrics.accuracy_score(Label_test, Label_predict))
            ml_record.measure(i, Label_test, Label_predict, 'weighted')
#            print(metrics.f1_score(Label_test, Label_predict, average='weighted'))
            i += 1

        file_wirte = "Result_AdaBoost.txt"
        ml_record.output(file_wirte, m, Dir)



