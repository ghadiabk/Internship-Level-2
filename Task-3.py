import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

# 1. Load and Preprocess Data
df = pd.read_csv('Level-2/Sentiment_Data.csv')

def simplify_sentiment(s):
    s = str(s).strip().lower()
    positives = ['positive', 'happiness', 'joy', 'love', 'excited', 'inspired', 'contentment', 'gratitude']
    negatives = ['negative', 'anger', 'fear', 'sadness', 'disgust', 'disappointed', 'bitter', 'frustrated']
    if any(p in s for p in positives): return 1
    if any(n in s for n in negatives): return -1
    return 0

df['Sentiment_Score'] = df['Sentiment'].apply(simplify_sentiment)

features = ['Retweets', 'Likes', 'Sentiment_Score']
X = df[features].dropna()
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

wcss = []
sil_scores = []
k_range = range(2, 11)

for k in k_range:
    kmeans = KMeans(n_clusters=k, init='k-means++', random_state=42, n_init=10)
    kmeans.fit(X_scaled)
    wcss.append(kmeans.inertia_)
    sil_scores.append(silhouette_score(X_scaled, kmeans.labels_))

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(range(2, 11), wcss, marker='o', color='b')
plt.title('Elbow Method (WCSS)')
plt.xlabel('Number of Clusters')

plt.subplot(1, 2, 2)
plt.plot(range(2, 11), sil_scores, marker='o', color='r')
plt.title('Silhouette Scores')
plt.xlabel('Number of Clusters')
plt.savefig('Level-2/plots/t3/optimization_metrics.png')

optimal_k = 3
kmeans = KMeans(n_clusters=optimal_k, init='k-means++', random_state=42, n_init=10)
df['Cluster'] = kmeans.fit_predict(X_scaled)

pca = PCA(n_components=2)
pca_data = pca.fit_transform(X_scaled)
df['PCA1'], df['PCA2'] = pca_data[:, 0], pca_data[:, 1]

plt.figure(figsize=(10, 6))
sns.scatterplot(x='PCA1', y='PCA2', hue='Cluster', data=df, palette='viridis', s=100)
plt.title('K-Means Clusters (PCA Projection)')
plt.savefig('Level-2/plots/t3/cluster_visualization.png')

print("Cluster Summary (Mean Metrics):")
print(df.groupby('Cluster')[features].mean())