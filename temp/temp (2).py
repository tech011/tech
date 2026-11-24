# Exploratory Data Analysis (EDA) on the Titanic dataset 

#Aim:  
#To perform Exploratory Data Analysis (EDA) on the Titanic dataset using summary statistics and 
#visualization techniques such as histogram, count plot, box plot, scatter plot and correlation heatmap. 

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os

os.makedirs('figs', exist_ok=True)



df=sns.load_dataset('titanic')

print('shape:',df.shape)
print(df[['survived', 'pclass', 'sex',
          'age', 'fare', 'embarked', 'class']].head())

print()
print(df.describe().round(2))
print()
print(df.isnull().sum())



print() #statistical calculation

print('statistical calculation')

print(df['age'].mean())
print(df['age'].median())
print(df['age'].mode()[0])

print(df['age'].std())
print(df['age'].var())
print(df['age'].skew())
print(df['age'].kurt())




print() #passenger count

print(df['class'].value_counts())
print(df.groupby('class', observed=True)['survived'].mean().round(3))


fig, axes = plt.subplots(2, 2, figsize=(11, 8))

sns.histplot(df['age'].dropna(), kde=True, bins=20,
             color='steelblue', ax=axes[0, 0])

sns.countplot(x='class', hue='class', data=df,
              palette='Set2', legend=False, ax=axes[0, 1])

sns.boxplot(y=df['fare'], color='lightcoral',
            ax=axes[1, 0])

sns.scatterplot(x='age', y='fare', hue='survived',
                data=df, s=30, ax=axes[1, 1])

plt.tight_layout()
plt.savefig('figs/exp2_eda.png', dpi=110)
plt.show()


plt.figure(figsize=(6, 5))

sns.heatmap(
    df.select_dtypes(include=[np.number]).corr(),
    annot=True,
    cmap='coolwarm',
    fmt='.2f'
)

plt.title('Correlation Heatmap')
plt.tight_layout()
plt.savefig('figs/exp2_corr.png', dpi=110)
plt.show()