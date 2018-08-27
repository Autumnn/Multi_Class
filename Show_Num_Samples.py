import os
import numpy as np

path = "KEEL_Cross_Folder_npz"
dirs = os.listdir(path) #Get files in the folder

for Dir in dirs:
    print("Data Set Name: ", Dir)
    dir_path = path + "/" + Dir
    files = os.listdir(dir_path)  # Get files in the folder

    i = 0
    for file in files:
        name = dir_path + '/' + file
        r = np.load(name)

        Feature_train = r['F_tr']
        Label_train = r['L_tr']
        Num_train = Feature_train.shape[0]
        t = Label_train.ravel()
        t = t.astype(int)
        print(np.bincount(t), Feature_train.shape[1])
        Feature_test = r['F_te']
        Label_test = r['L_te']
        Label_test.ravel()
        Num_test = Feature_test.shape[0]
        print(np.bincount(Label_test.ravel().astype(int)))
        print(i, " folder; ", "Number of train: ", Num_train, "Number of test: ", Num_test)

        i += 1
