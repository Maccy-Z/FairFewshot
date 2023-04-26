# create a sample DataFrame with multi-level columns
import pandas as pd
import numpy as np

arrays = [['header1', 'header1', 'header2', 'header2'],
          ['col1', 'col2', 'col3', 'col4']]
tuples = list(zip(*arrays))
index = pd.MultiIndex.from_tuples(tuples)
df = pd.DataFrame(np.random.randn(5, 4), columns=index)


print(df)
# select only the column 'col2' under the secondary header named 'header1'
xs = df.loc[:, ('header1', 'col2')]

print(xs)


