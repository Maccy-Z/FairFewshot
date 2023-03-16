import numpy as np

arr1 = np.array([1, 2, 3, 4])
arr2 = np.array(['a', 'b', 'c'])

pairs = np.transpose([np.repeat(arr1, len(arr2)), np.tile(arr2, len(arr1))])

#pairs = pairs.reshape(len(arr1), len(arr2), 2)
print(pairs)
