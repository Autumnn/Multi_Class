import pandas as pd
import numpy as np
from imblearn.over_sampling import SMOTE

name = 'KEEL_Cross_Folder_npz/yeast/yeast_1_Cross_Folder.npz'
r = np.load(name)

Feature_train = r['F_tr']
Label_train = r['L_tr']
Num_train = Feature_train.shape[0]
# print(Num_train)
Feature_test = r['F_te']
Label_test = r['L_te']
Label_test.ravel()
Num_test = Feature_test.shape[0]
# print(Num_test)

df = pd.DataFrame(Label_train)
print(df[0].value_counts())

'''
show = np.concatenate((Feature_train, Label_train), axis=1)
sm = SMOTE()
Feature_train_o, Label_train_o = sm.fit_sample(Feature_train, Label_train.ravel())
Label_train_o = Label_train_o[:,np.newaxis]
'''