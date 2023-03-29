import numpy as np

# Create two lists
list1 = [1, 2, 3, 4, 5]
list2 = [3, 4, 5, 6, 7]

# Get the non-overlapping items
non_overlap = np.setdiff1d(list1, list2)

print(non_overlap)
