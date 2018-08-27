import os
import numpy as np

path = "KEEL_Cross_Folder_Valid_npz"
dirs = os.listdir(path) #Get files in the folder

for Dir in dirs:
    print("Data Set Name: ", Dir)
    dir_path = path + "/" + Dir
    files = os.listdir(dir_path)  # Get files in the folder

    i = 0
    for file in files:
        name = dir_path + '/' + file
        r = np.load(name)

        Feature_train_list = r['F_tr_l']
        Label_train_list = r['L_tr_l']
        Feature_valid_list = r['F_va_l']
        Label_valid_list = r['L_va_l']
        Num_list = len(Feature_train_list)
        for j in range(Num_list):
            Num_train = Feature_train_list[j].shape[0]
            t = Label_train_list[j].ravel()
            t = t.astype(int)
            print('The %d th folder training data' % j)
            print(np.bincount(t)[1:], Feature_train_list[j].shape[1])
            Num_valid = Feature_valid_list[j].shape[0]
            s = Label_valid_list[j].ravel()
            s = s.astype(int)
            print('The %d th folder valid data' % j)
            print(np.bincount(s)[1:], Feature_valid_list[j].shape[1])

        Feature_test = r['F_te']
        Label_test = r['L_te']
        Label_test.ravel()
        Num_test = Feature_test.shape[0]
        print(np.bincount(Label_test.ravel().astype(int))[1:])
        print(i, " folder; ", "Number of train: ", Num_train, "Number of test: ", Num_test)

        i += 1