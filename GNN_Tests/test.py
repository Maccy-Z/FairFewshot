import numpy as np

# example list
my_list = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

indicies = np.random.choice(10, size=10, replace=False)

a, b, c = np.split(indicies, (3, 4))
print(a, b, c)

