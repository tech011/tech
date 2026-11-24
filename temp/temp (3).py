#  Linear Regression (Years of Experience -> Salary) 

# Aim:  
# To implement a simple linear regression model that predicts salary from years of experience and to 
# evaluate it using MAE, MSE and R-squared. 



import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load Dataset
df = pd.read_csv('Salary_dataset.csv')
df = df.drop(columns=['Unnamed: 0'])

print('Shape:', df.shape)
print(df.head())
print('\nMissing values:\n', df.isnull().sum())

# Select X and Y
X = df[['YearsExperience']]
Y = df['Salary']

# Split Dataset
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.2, random_state=42
)

# Create and Train Model
model = LinearRegression()
model.fit(X_train, Y_train)

# Slope and Intercept
print('\nSlope (coefficient): %.2f' % model.coef_[0])
print('Intercept          : %.2f' % model.intercept_)

# Prediction
Y_pred = model.predict(X_test)

# Evaluation
print('\nMean Absolute Error (MAE) : %.2f' % mean_absolute_error(Y_test, Y_pred))
print('Mean Squared Error (MSE) : %.2f' % mean_squared_error(Y_test, Y_pred))
print('R-Squared Score (R2)      : %.4f' % r2_score(Y_test, Y_pred))

# Predict Salary for 5 Years Experience
print('\nPredicted salary for 5 years experience: %.2f' %
      model.predict(pd.DataFrame({'YearsExperience': [5]}))[0])

# Plot Graph
plt.figure(figsize=(6, 4))
plt.scatter(X, Y, label='Actual Data')
plt.plot(X, model.predict(X), color='red', label='Regression Line')

plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.title('Linear Regression: Salary vs Experience')
plt.legend()
plt.tight_layout()
plt.savefig('figs/exp3_linreg.png', dpi=110)
plt.show()
