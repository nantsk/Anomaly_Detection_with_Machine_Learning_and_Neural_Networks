import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import IsolationForest
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('dataset.csv')
df['time'] = pd.to_datetime(df['time'], errors='coerce')
df = df.sort_values('time')
X = df.drop(['time'], axis=1)
numeric_columns = X.select_dtypes(include=[np.number]).columns
X_numeric = X[numeric_columns]

# Isolation Forest model
iso_forest = IsolationForest(contamination=0.1, random_state=42)
anomaly_labels = iso_forest.fit_predict(X_numeric)
df['anomaly'] = anomaly_labels

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

# Correlation Matrix
plt.figure(figsize=(12, 10))
sns.heatmap(X_numeric.corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.tight_layout()
plt.show()

# Summary of anomalies
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

# Anomaly distribution over time
anomaly_counts = df[df['anomaly'] == -1].groupby(df['time'].dt.date)['anomaly'].count()
plt.figure(figsize=(12, 6))
anomaly_counts.plot(kind='bar')
plt.title('Distribution of Anomalies Over Time')
plt.xlabel('Date')
plt.ylabel('Number of Anomalies')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
