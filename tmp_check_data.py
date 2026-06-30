import pandas as pd
import numpy as np

raw = pd.read_csv('data/voxel/DataStruct.csv', delimiter=' ', header=0)
print('Total shape:', raw.shape)
print('Column names (first 3):', list(raw.columns[:3]))
print('Column names (last 5):', list(raw.columns[-5:]))
print()
print('Unique values in last 5 cols:')
for c in raw.columns[-5:]:
    print(f'  {c}: {raw[c].nunique()} unique, sample: {raw[c].iloc[:5].tolist()}')
print()
print('Unique values in first col (timestamps):', raw.iloc[:, 0].nunique())
print('Total rows:', len(raw))

labels = pd.read_csv('data/voxel/Voxel_Labels.csv')
print('\nVoxel_Labels.csv:')
print('Shape:', labels.shape)
print('Columns:', list(labels.columns))
print('Unique IDs:', labels['ID'].nunique())
print('Unique X:', labels['X'].nunique())
print('Unique Y:', labels['Y'].nunique())
print('Unique Z:', labels['Z'].nunique())
print('Sample:')
print(labels.head(10))
