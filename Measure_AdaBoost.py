import os
import numpy as np
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics

path = "KEEL_Cross_Folder_npz"
dirs = os.listdir(path) #Get files in the folder

for Dir in dirs:
    print("Data Set Name: ", Dir)
    dir_path = path + "/" + Dir
    files = os.listdir(dir_path)  # Get files in the folder

    Num_Cross_Folders = 5
#    ml_record = metric_list(np.array([1]), np.array([1]), Num_Cross_Folders)
    Accuracy_score = []
    F_measure_weighted_score = []
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

        bdt = AdaBoostClassifier(DecisionTreeClassifier(max_depth=2, min_samples_split=20, min_samples_leaf=5),
                                 algorithm='SAMME', n_estimators=200, learning_rate=0.8)
        bdt.fit(Feature_train, Label_train.ravel())
        Label_predict = bdt.predict(Feature_test)

        Accuracy_score.append(metrics.accuracy_score(Label_test, Label_predict))
        F_measure_weighted_score.append(metrics.f1_score(Label_test, Label_predict, average='weighted'))

    print('Accuracy mean: ', np.mean(Accuracy_score))
    print('F_measure weighted: ', np.mean(F_measure_weighted_score))

