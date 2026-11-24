
# : Decision Tree vs Random Forest

# Aim:  
# To implement Decision Tree and Random Forest classifiers for predicting Titanic passenger survival and 
# to compare their performance. 

# Titanic Survival Prediction

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score, precision_score,
    recall_score, f1_score
)

# Load Dataset
df = sns.load_dataset('titanic')

df = df[['survived', 'pclass', 'sex', 'age',
         'sibsp', 'parch', 'fare', 'embarked']]

# Handle Missing Values
df['age'] = df['age'].fillna(df['age'].median())

df['embarked'] = df['embarked'].fillna(
    df['embarked'].mode()[0]
)

# One Hot Encoding
df = pd.get_dummies(
    df, columns=['sex', 'embarked'], drop_first=True
)

# Input and Output
X = df.drop(columns='survived')
y = df['survived']

# Split Dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2,
    random_state=42, stratify=y
)

# Create Models
models = {
    'Decision Tree': DecisionTreeClassifier(
        max_depth=5, random_state=42
    ),

    'Random Forest': RandomForestClassifier(
        n_estimators=200,
        max_depth=7,
        random_state=42
    )
}

# Train and Evaluate
rows = []

for name, model in models.items():

    model.fit(X_train, y_train)

    pred = model.predict(X_test)

    rows.append([
        name,
        accuracy_score(y_test, pred),
        precision_score(y_test, pred),
        recall_score(y_test, pred),
        f1_score(y_test, pred)
    ])

# Comparison Table
result = pd.DataFrame(
    rows,
    columns=['Model', 'Accuracy', 'Precision', 'Recall', 'F1']
).round(4)

print('Comparison of models on the test set:')
print(result.to_string(index=False))

# Feature Importance
importance = pd.Series(
    models['Random Forest'].feature_importances_,
    index=X.columns
)

print('\nRandom Forest feature importance:')
print(importance.sort_values(ascending=False).round(4))

# Plot Graphs
fig, axes = plt.subplots(1, 2, figsize=(11, 4))

result.set_index('Model')[
    ['Accuracy', 'Precision', 'Recall', 'F1']
].plot(kind='bar', ax=axes[0], rot=0)

axes[0].set_ylim(0, 1)
axes[0].set_title('Decision Tree vs Random Forest')

importance.sort_values().plot(
    kind='barh', ax=axes[1], color='seagreen'
)

axes[1].set_title('Random Forest Feature Importance')

plt.tight_layout()
plt.savefig('figs/exp5_dt_rf.png', dpi=110)
plt.show()
