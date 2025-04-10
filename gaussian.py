# Gaussian Mixture Model
import pandas as pd
import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('dataset.csv')
df['time'] = pd.to_datetime(df['time'], errors='coerce')
df = df.sort_values('time')
X = df.drop(['time'], axis=1)
numeric_columns = X.select_dtypes(include=[np.number]).columns
X_numeric = X[numeric_columns]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_numeric)

# Gaussian mixture model
n_components = 5
gmm = GaussianMixture(n_components=n_components, random_state=42)
gmm.fit(X_scaled)
probability_scores = gmm.score_samples(X_scaled)

threshold = np.percentile(probability_scores, 10)
anomaly_labels = (probability_scores <= threshold).astype(int)
anomaly_labels = [-1 if x == 1 else 1 for x in anomaly_labels] 
df['anomaly'] = anomaly_labels
df['anomaly_score'] = -probability_scores

def classify_anomalies(df, window_size=5, collective_threshold=3):
    df['anomaly_type'] = 'Normal'
        
    df.loc[df['anomaly'] == -1, 'anomaly_type'] = 'Point'
    
    for feature in numeric_columns:
        rolling_mean = df[feature].rolling(window=window_size).mean()
        rolling_std = df[feature].rolling(window=window_size).std()
        z_scores = (df[feature] - rolling_mean) / rolling_std
        contextual_mask = (abs(z_scores) > 3) & (df['anomaly'] == -1)
        df.loc[contextual_mask, 'anomaly_type'] = 'Contextual'
    
    anomaly_groups = df['anomaly'].rolling(window=window_size).sum()
    collective_mask = (anomaly_groups <= -collective_threshold) & (df['anomaly'] == -1)
    df.loc[collective_mask, 'anomaly_type'] = 'Collective'
    return df

df = classify_anomalies(df)

def plot_anomalies(data, feature):
    plt.figure(figsize=(15, 8))
    
    normal = data[data['anomaly'] == 1]
    plt.scatter(normal['time'], normal[feature], c='blue', label='Normal', alpha=0.5)
    
    for anomaly_type, color in zip(['Point', 'Contextual', 'Collective'], ['red', 'green', 'purple']):
        anomalous = data[data['anomaly_type'] == anomaly_type]
        plt.scatter(anomalous['time'], anomalous[feature], c=color, label=anomaly_type, alpha=0.7)
    
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
plt.title('Correlation Matrix')
plt.tight_layout()
plt.show()

anomaly_summary = df['anomaly_type'].value_counts().reset_index()
anomaly_summary.columns = ['Anomaly Type', 'Count']
anomaly_summary['Percentage'] = anomaly_summary['Count'] / len(df) * 100
print("\nAnomaly Summary:")
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

anomaly_counts = df[df['anomaly'] == -1].groupby([df['time'].dt.date, 'anomaly_type'])['anomaly'].count().unstack()
plt.figure(figsize=(15, 8))
anomaly_counts.plot(kind='bar', stacked=True)
plt.title('Distribution of Anomaly Types Over Time')
plt.xlabel('Date')
plt.ylabel('Number of Anomalies')
plt.legend(title='Anomaly Type')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

for anomaly_type in ['Point', 'Contextual', 'Collective']:
    print(f"\nTop 10 {anomaly_type} Anomalies:")
    print(df[df['anomaly_type'] == anomaly_type].sort_values('anomaly_score', ascending=False)[['time', 'anomaly_type'] + list(numeric_columns) + ['anomaly_score']].head(10))

plt.figure(figsize=(10, 6))
plt.hist(probability_scores, bins=50, density=True, alpha=0.7)
plt.axvline(threshold, color='r', linestyle='dashed', linewidth=2)
plt.title('Probability Density of GMM Scores')
plt.xlabel('Log Probability Score')
plt.ylabel('Density')
plt.show()
