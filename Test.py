import numpy as np
from sklearn.preprocessing import normalize

target = np.array([1,1,2,0,1,2,0,1,2])
idx_neighbors = np.random.randint(0,9,(4,7))
frequency = np.bincount(target)
neighbour_label = target[idx_neighbors]
print(target)
print(idx_neighbors)
#print(neighbour_label)
#print(frequency)
#print(frequency[neighbour_label])
num = frequency[neighbour_label]
weight = 1./(1 + np.exp(0.9 * num))
#print(weight)
weight = normalize(weight, norm='l1')
print(weight)
print(np.sum(weight,axis=1))
processed_dsel = np.random.randint(0,2,(9,5))
#print(processed_dsel)
correct_num = processed_dsel[idx_neighbors, :]
print(correct_num)
correct = np.zeros((4, 7, 5))
for i in range(5):
    correct[:,:,i] = correct_num[:,:,i] * weight
print(correct)
accuracy = np.mean(correct, axis=1)
print(accuracy)
competent_indices = np.argsort(accuracy, axis=1)[:, ::-1][:, 0:2]
print(competent_indices)