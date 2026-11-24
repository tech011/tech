#1: Data Cleaning and Preprocessing using Pandas and NumPy 

#Aim:  
#To clean and preprocess a raw employee dataset using Pandas and NumPy by handling missing values, 
#removing duplicates, standardizing text, converting data types, treating outliers, encoding categorical 
#columns and normalizing numerical columns. 

import pandas as pd
import numpy as np

pd.set_option('display.width', 100)
pd.set_option('display.max_columns', 10)

# Load dataset
df = pd.read_excel('employee_raw_data.xlsx')
print('Shape:', df.shape)
print(df.head())

# Missing values and duplicates
print(df.isnull().sum())
print(df.duplicated().sum())

df = df.drop_duplicates()

# Fill missing values
df['Salary'] = pd.to_numeric(df['Salary'], errors='coerce')
df['Age'] = df['Age'].fillna(df['Age'].median())
df['Salary'] = df['Salary'].fillna(df['Salary'].median())
df['Gender'] = df['Gender'].fillna(df['Gender'].mode()[0])
df['Department'] = df['Department'].fillna(df['Department'].mode()[0])

# Standardize text
df['Name'] = df['Name'].str.strip()
df['Gender'] = df['Gender'].str.strip().str.upper().replace(
    {'MALE': 'Male', 'M': 'Male', 'FEMALE': 'Female', 'F': 'Female'})

df['Department'] = df['Department'].str.strip().str.upper()

# Convert data types
df['JoiningDate'] = pd.to_datetime(df['JoiningDate'], errors='coerce', format='mixed')
df['Age'] = df['Age'].astype(int)

# Outlier treatment
Q1 = df['Age'].quantile(0.25)
Q3 = df['Age'].quantile(0.75)
IQR = Q3 - Q1

lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

df['Age'] = np.clip(df['Age'], lower_bound, upper_bound)

# Encoding
df_encoded = pd.get_dummies(
    df, columns=['Gender', 'Department'], drop_first=True)

# Normalization
df_encoded['Salary_Normalized'] = (
    (df_encoded['Salary'] - df_encoded['Salary'].min()) /
    (df_encoded['Salary'].max() - df_encoded['Salary'].min())
)

print(df_encoded.head())
