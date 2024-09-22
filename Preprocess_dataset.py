import pandas as pd
from sklearn.preprocessing import StandardScaler

data = pd.read_csv('Dataset.csv')

print("\nSummary statistics of the dataset:")
print(data.describe())

print("\nMissing values in the dataset:")
print(data.isnull().sum())

data.fillna(data.mean(), inplace=True)

columns_to_normalize = ['alt', 'MN', 'TRA', 'Wf', 'Fn', 'Nf', 'Nc', 'epr', 'phi', 'NfR', 'NcR', 'BPR', 'farB', 'htBleed']
scaler = StandardScaler()
data[columns_to_normalize] = scaler.fit_transform(data[columns_to_normalize])

print("\nNormalized dataset:")
print(data.head())

data['Nf_Nc_ratio'] = data['Nf'] / data['Nc']

data['Wf_moving_avg'] = data['Wf'].rolling(window=3).mean()
data['alt_diff'] = data['alt'].diff()

print("\nDataset with new features:")
print(data.head())

data.to_csv('preprocessed_Dataset.csv', index=False)

print("\nPreprocessed data has been saved to 'preprocessed_Dataset.csv'")
