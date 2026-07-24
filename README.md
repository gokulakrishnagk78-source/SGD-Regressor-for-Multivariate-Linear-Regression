# SGD-Regressor-for-Multivariate-Linear-Regression

## AIM:
To write a program to predict the price of the house and number of occupants in the house with SGD regressor.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
Load the California housing dataset and select input and output variables.
Split the dataset into training and testing data, then standardize the values.
Train the Multi-Output SGD Regression model using the training data.
Predict the output for test data and display the predicted values.

## Program:
```
/*
Program to implement the multivariate linear regression model for predicting the price of the house and number of occupants in the house with SGD regressor.
Developed by: GOKULAKRISHNA S
RegisterNumber:  212225230077
*/
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import SGDRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler

data = fetch_california_housing()

X = data.data[:, :3]
y = np.column_stack((data.target, data.data[:, 6]))

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler_x = StandardScaler()
scaler_y = StandardScaler()

X_train = scaler_x.fit_transform(X_train)
X_test = scaler_x.transform(X_test)

y_train = scaler_y.fit_transform(y_train)
y_test = scaler_y.transform(y_test)

sgd = SGDRegressor(max_iter=1000, tol=1e-6, random_state=42)

multi_output_sgd = MultiOutputRegressor(sgd)

multi_output_sgd.fit(X_train, y_train)

y_pred = multi_output_sgd.predict(X_test)

y_pred = scaler_y.inverse_transform(y_pred)
y_test = scaler_y.inverse_transform(y_test)

print("Predictions:")
print(y_pred[:5])

mse = mean_squared_error(y_test, y_pred)
print("\nMean Squared Error:", mse)

plt.scatter(y_test[:, 0], y_pred[:, 0])
plt.plot(
    [y_test[:, 0].min(), y_test[:, 0].max()],
    [y_test[:, 0].min(), y_test[:, 0].max()],
    'r--'
)
plt.xlabel("Actual House Value")
plt.ylabel("Predicted House Value")
plt.title("Actual vs Predicted House Value")
plt.show()
```


## Output:
<img width="794" height="650" alt="image" src="https://github.com/user-attachments/assets/5e2c447d-826d-47c4-bdbe-996bc3511c5d" />


## Result:
Thus the program to implement the multivariate linear regression model for predicting the price of the house and number of occupants in the house with SGD regressor is written and verified using python programming.
