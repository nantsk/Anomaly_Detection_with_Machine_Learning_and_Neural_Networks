#Local Outlier Factor // https://github.com/Rhythmica02/Anomaly_Detection_with_Machine_Learning_and_Neural_Networks
import pandas as pd
import numpy as np
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

# Loading the dataset
df = pd.read_csv('dataset.csv')

df['time'] = pd.to_datetime(df['time'], errors='coerce')

df = df.sort_values('time')

X = df.drop(['time'], axis=1)

numeric_columns = X.select_dtypes(include=[np.number]).columns
X_numeric = X[numeric_columns]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_numeric)

lof = LocalOutlierFactor(n_neighbors=20, contamination=0.1)
anomaly_labels = lof.fit_predict(X_scaled)

df['anomaly'] = anomaly_labels

# For Plotting
def plot_anomalies(data, feature):
    plt.figure(figsize=(12, 6))
    normal = data[data['anomaly'] == 1]
    anomalous = data[data['anomaly'] == -1]
    
    plt.scatter(normal['time'], normal[feature], c='blue', label='Normal', alpha=0.5)
    plt.scatter(anomalous['time'], anomalous[feature], c='red', label='Anomaly', alpha=0.5)
    
    plt.title(f'Anomalies in {feature} over Time')
    plt.xlabel('Time')
    plt.ylabel(feature)
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
for feature in numeric_columns:
    plot_anomalies(df, feature)

# For correlation Matrix
plt.figure(figsize=(12, 10))
sns.heatmap(X_numeric.corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.tight_layout()
plt.show()

# For summary
anomaly_summary = pd.DataFrame({
    'Feature': numeric_columns,
    'Anomaly_Count': [(df['anomaly'] == -1).sum() for _ in numeric_columns],
    'Anomaly_Percentage': [(df['anomaly'] == -1).mean() * 100 for _ in numeric_columns]
})
print(anomaly_summary)

plt.figure(figsize=(15, 10))
for feature in numeric_columns:
    plt.plot(df['time'], df[feature], label=feature, alpha=0.7)
plt.title('Time Series of All Numeric Parameters')
plt.xlabel('Time')
plt.ylabel('Value')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()

anomaly_counts = df[df['anomaly'] == -1].groupby(df['time'].dt.date)['anomaly'].count()
plt.figure(figsize=(12, 6))
anomaly_counts.plot(kind='bar')
plt.title('Distribution of Anomalies Over Time')
plt.xlabel('Date')
plt.ylabel('Number of Anomalies')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

anomaly_scores = -lof.negative_outlier_factor_
df['anomaly_score'] = anomaly_scores
print("\nTop 10 Anomalies:")
print(df.sort_values('anomaly_score', ascending=False)[['time'] + list(numeric_columns) + ['anomaly_score']].head(10))
