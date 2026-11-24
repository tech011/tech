
# Dimensionality Reduction using PCA (Wine dataset, 13 features)

# 1. Aim
# To reduce the 13-dimensional Wine dataset to fewer dimensions using Principal Component Analysis (PCA) and study the variance explained by each component.


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_wine
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

wine = load_wine()
X, y = wine.data, wine.target
print('Original shape:', X.shape)

X_scaled = StandardScaler().fit_transform(X)

# PCA with all components to study explained variance
pca_full = PCA().fit(X_scaled)
ratio = pca_full.explained_variance_ratio_
cum = np.cumsum(ratio)

print('\nExplained variance ratio of each component:')
print(np.round(ratio, 4))
print('\nCumulative explained variance:')
print(np.round(cum, 4))
print('\nComponents needed for 95% variance:',
      np.argmax(cum >= 0.95) + 1)

# Reduce to 2 components
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)
print('\nShape after PCA:', X_pca.shape)
print('Variance kept by 2 components: %.4f' %
      pca.explained_variance_ratio_.sum())

# Compare a classifier on original vs reduced data
for name, data in [
    ('Original (13 features)', X_scaled),
    ('PCA (2 components)', X_pca)
]:
    Xtr, Xte, ytr, yte = train_test_split(
        data, y, test_size=0.3,
        random_state=42, stratify=y
    )
    acc = LogisticRegression(max_iter=1000).fit(
        Xtr, ytr
    ).score(Xte, yte)
    print('Accuracy with %-24s: %.4f' % (name, acc))

fig, axes = plt.subplots(1, 2, figsize=(11, 4))

axes[0].bar(range(1, 14), ratio, label='Individual')
axes[0].step(range(1, 14), cum, where='mid',
             color='red', label='Cumulative')
axes[0].set_xlabel('Principal component')
axes[0].set_ylabel('Explained variance ratio')
axes[0].set_title('Scree Plot')
axes[0].legend()

for cls in np.unique(y):
    axes[1].scatter(
        X_pca[y == cls, 0],
        X_pca[y == cls, 1],
        label=wine.target_names[cls], s=25
    )

axes[1].set_xlabel('PC1')
axes[1].set_ylabel('PC2')
axes[1].set_title('Wine Data in 2 Principal Components')
axes[1].legend()

plt.tight_layout()
plt.savefig('figs/exp7_pca.png', dpi=110)
plt.show()
