import os
import numpy as np
import cma
from datetime import datetime


pop = 5*np.random.uniform(-1,1,(2, 10))
print(pop[:, 1])
zbest = pop[:,1]
zbest.reshape(2,10)
print(zbest-pop)
