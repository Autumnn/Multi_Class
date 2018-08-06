import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier

y = ['A','A','B','C','A','B','C','A','B']
print(y)
enc = LabelEncoder()
y_ind = enc.fit_transform(y)
print(y_ind)
print(enc.classes_)

#target = np.array([0,0,1,2,0,1,2,0,1])
target = y_ind
predictions = np.random.randint(0,3,(9,5))
print(predictions)
print(predictions[:,1:])
BKS_dsel = predictions
processed_dsel = BKS_dsel == target[:, np.newaxis]
print(processed_dsel)
idx_neighbors = np.random.randint(0,8,(9,3))
print(idx_neighbors)

f_1 = processed_dsel[idx_neighbors,:].swapaxes(1,2).reshape(-1,3)
print(f_1)

pct_agree = np.sum(processed_dsel,axis=1)/5
print(pct_agree)
Hc = 0.9
print(np.where(Hc > pct_agree)[0])
print(np.where(pct_agree > (1-Hc))[0])
indices_selected = np.hstack((np.where(Hc > pct_agree)[0],np.where(pct_agree > (1-Hc))[0]))
print(indices_selected)
indices_selected = np.unique(indices_selected)
print(indices_selected)

dsel_scores = np.random.rand(9,5,3)
f_2 = dsel_scores[idx_neighbors, :, target[idx_neighbors]].swapaxes(1, 2)
f_2 = f_2.reshape(-1, 3)
print(f_2)

print(processed_dsel[idx_neighbors, :])
f_3 = np.mean(processed_dsel[idx_neighbors, :], axis=1)
print(f_3)
f_3 = f_3.reshape(-1, 1)
print(f_3)

print(np.hstack((f_1, f_2, f_3)))

'''
dsel_output_profiles = dsel_scores.reshape(9, 5*3)
print(dsel_output_profiles)
op_knn = KNeighborsClassifier(n_neighbors=3, n_jobs=1, algorithm='auto')
op_knn.fit(dsel_output_profiles, y_ind)
'''

meta_feature_target = processed_dsel[indices_selected, :].reshape(-1,)
print(meta_feature_target)
