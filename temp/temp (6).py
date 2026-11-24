
# Experiment 6: K-Means Clustering (Iris dataset)
# and visualization

# Aim:  
# To group the Iris flower samples into clusters using the K-Means algorithm, choose the number of clusters 
# with the Elbow method and visualize the clusters.

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# Load Dataset
iris = load_iris()

X = pd.DataFrame(
    iris.data,
    columns=iris.feature_names
)

# Feature Scaling
X_scaled = StandardScaler().fit_transform(X)

# Elbow Method to choose k
inertia = []

for k in range(1, 11):

    km = KMeans(
        n_clusters=k,
        n_init=10,
        random_state=42
    )

    km.fit(X_scaled)

    inertia.append(km.inertia_)

print('Inertia for k = 1 to 10:')
print([round(v, 2) for v in inertia])

# Final Model with k = 3
kmeans = KMeans(
    n_clusters=3,
    n_init=10,
    random_state=42
)

labels = kmeans.fit_predict(X_scaled)

# Silhouette Score
print(
    '\nSilhouette score (k=3): %.4f'
    % silhouette_score(X_scaled, labels)
)

# Points in each Cluster
print('\nPoints in each cluster:')
print(
    pd.Series(labels).value_counts().sort_index()
)

# Cluster vs Actual Species
print('\nCluster vs actual species:')

print(
    pd.crosstab(
        labels,
        iris.target_names[iris.target],
        rownames=['Cluster'],
        colnames=['Species']
    )
)

# Plot Graphs
fig, axes = plt.subplots(1, 2, figsize=(11, 4))

# Elbow Graph
axes[0].plot(
    range(1, 11), inertia, marker='o'
)

axes[0].set_xlabel('Number of clusters (k)')
axes[0].set_ylabel('Inertia')
axes[0].set_title('Elbow Method')

# K-Means Clusters
axes[1].scatter(
    X.iloc[:, 2],
    X.iloc[:, 3],
    c=labels,
    cmap='viridis',
    s=30
)

# Centroids
centers = StandardScaler().fit(X).inverse_transform(
    kmeans.cluster_centers_
)

axes[1].scatter(
    centers[:, 2],
    centers[:, 3],
    c='red',
    marker='X',
    s=200,
    label='Centroids'
)

axes[1].set_xlabel('Petal length (cm)')
axes[1].set_ylabel('Petal width (cm)')
axes[1].set_title('K-Means Clusters (k = 3)')
axes[1].legend()

plt.tight_layout()
plt.savefig('figs/exp6_kmeans.png', dpi=110)
plt.show()
