# Logistic Regression for a classification problem 
# Dataset: Breast Cancer Wisconsin (built into scikit-learn) - malignant (0) / benign (1) 

# Aim:  
# To build a logistic regression model that classifies breast tumours as malignant or benign and to evaluate 
# it using accuracy, confusion matrix, classification report and ROC curve. 


# Dataset: Breast Cancer Wisconsin

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    accuracy_score, confusion_matrix, classification_report,
    ConfusionMatrixDisplay, roc_auc_score, RocCurveDisplay
)

# Load Dataset
data = load_breast_cancer()

X = pd.DataFrame(data.data, columns=data.feature_names)
y = data.target

print('Shape:', X.shape)
print('Malignant samples:', (y == 0).sum(),
      ' Benign samples:', (y == 1).sum())

# Split Dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Feature Scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train Model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Prediction
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

# Evaluation
print('\nAccuracy : %.4f' % accuracy_score(y_test, y_pred))
print('ROC-AUC : %.4f' % roc_auc_score(y_test, y_prob))

print('\nConfusion Matrix:')
print(confusion_matrix(y_test, y_pred))

print('\nClassification Report:')
print(classification_report(
    y_test, y_pred, target_names=data.target_names
))

# Plot Graphs
fig, axes = plt.subplots(1, 2, figsize=(10, 4))

ConfusionMatrixDisplay.from_predictions(
    y_test, y_pred,
    display_labels=data.target_names,
    cmap='Blues', ax=axes[0]
)

axes[0].set_title('Confusion Matrix')

RocCurveDisplay.from_predictions(
    y_test, y_prob, ax=axes[1]
)

axes[1].set_title('ROC Curve')

plt.tight_layout()
plt.savefig('figs/exp4_logreg.png', dpi=110)
plt.show()
