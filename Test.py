import numpy as np

pop_size = 10
bounds = np.array([(0, 1), (1, 2), (2, 3)], dtype=np.float)
dim = len(bounds)
print(dim)
para = np.empty((pop_size, dim))

for col, (lower, upper) in enumerate(bounds):
    para[:, col] = np.random.RandomState().uniform(lower, upper, pop_size)

print(para[1])