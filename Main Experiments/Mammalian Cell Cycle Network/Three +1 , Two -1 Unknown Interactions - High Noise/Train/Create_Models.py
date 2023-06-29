import numpy
import numpy as np
from itertools import product
import pickle

nums_missing = 5
network_size = 10

combs = product(range(3), repeat=nums_missing)
Combinations = numpy.array(list(combs))-1

ConnectivitySpace = numpy.zeros((3**nums_missing, network_size**2))

OptimalC = np.array(np.mat('+1 -1 -1 0 0 0 0 0 0 0;'
                           '0 0 0 -1 -1 -1 0 0 0 0;'
                           '0 +1 +1 +1 +1 0 0 +1 0 0;'
                           '0 0 0 0 +1 +1 0 0 0 0;'
                           '0 -1 -1 0 -1 0 0 0 0 0;'
                           '0 -1 -1 -1 -1 +1 0 -1 +1 0;'
                           '0 0 0 0 0 -1 0 +1 +1 -1;'
                           '0 0 0 0 0 -1 -1 0 -1 -1;'
                           '0 0 0 0 0 -1 0 0 +1 0;'
                           '0 -1 -1 -1 0 0 +1 -1 +1 0').T)  # true model #Transpose!

OptimalCVector = numpy.reshape(OptimalC, (1, network_size**2))

# create all possible models

for i in range(3**nums_missing):

    temp = OptimalCVector

    temp[0, (8 - 1) * network_size + 3 - 1] = Combinations[i, 0] #+1
    temp[0, (5 - 1) * network_size + 4 - 1] = Combinations[i, 1] #+1
    temp[0, (3 - 1) * network_size + 5 - 1] = Combinations[i, 2] #-1
    temp[0, (9 - 1) * network_size + 6 - 1] = Combinations[i, 3] #+1
    temp[0, (6 - 1) * network_size + 7 - 1] = Combinations[i, 4] #-1

    # temp[0, (8 - 1) * network_size + 3 - 1] = Combinations[i, 0] #+1
    # temp[0, (5 - 1) * network_size + 4 - 1] = Combinations[i, 1] #+1
    # temp[0, (9 - 1) * network_size + 6 - 1] = Combinations[i, 2] #+1
    # temp[0, (8 - 1) * network_size + 7 - 1] = Combinations[i, 3] #+1
    # temp[0, (7 - 1) * network_size + 10 - 1] = Combinations[i, 4] #+1

    ConnectivitySpace[i, :] = temp

# save all possible models

with open('GRNBOVals10Gene5Missing.pkl', 'wb') as f:
    pickle.dump(ConnectivitySpace, f)

# with open('GRNBOVals10Gene5MissingAllMissingOnes.pkl', 'wb') as f:
#     pickle.dump(ConnectivitySpace, f)

