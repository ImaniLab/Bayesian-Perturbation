import numpy
import numpy as np
from itertools import product
import pickle

nums_missing = 5
network_size = 10

combs = product(range(3), repeat=nums_missing)
Combinations = numpy.array(list(combs))-1

ConnectivitySpace = numpy.zeros((3**nums_missing, network_size**2))

OptimalC = np.array(np.mat('1 -1 -1 -1 -1  0  0  0  0  0;'
                           '0  1  1  1  1  0  0  0  0  0;'
                           '0  1  1  1  1  0  0  0  0  0;'
                           '0  1  1  1  1  0  0  0  0  0;'
                           '0  0  0  0  0 -1  0  0  0  0;'
                           '0  0  0  0  0  1  0  1  0  0;'
                           '0  0  0  0  0  0  1  1  0  0;'
                           '0  0  0  0  0  0  0  0  0  1;'
                           '0  0  0  0  0  0  0  1  1  0;'
                           '0  0  0  0  0  0  0  1  0  1').T)  # true #Transpose!!!!

OptimalCVector = numpy.reshape(OptimalC,(1,network_size**2))

for i in range(3**nums_missing):

        temp = OptimalCVector

        temp[0, (3 - 1) * network_size + 2 - 1] = Combinations[i, 0] #+1
        temp[0, (8 - 1) * network_size + 6 - 1] = Combinations[i, 1] #+1
        temp[0, (8 - 1) * network_size + 10 - 1] = Combinations[i, 2] #+1
        temp[0, (10 - 1) * network_size + 8 - 1] = Combinations[i, 3] #+1
        temp[0, (1 - 1) * network_size + 4 - 1] = Combinations[i, 4] #0

        ConnectivitySpace[i,:] = temp

with open('MCBOVals10Gene5MissingAllOnes.pkl', 'wb') as f:
    pickle.dump(ConnectivitySpace, f)
