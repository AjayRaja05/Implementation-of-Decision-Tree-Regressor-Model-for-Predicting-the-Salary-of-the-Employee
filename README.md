# Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee

## AIM:
To write a program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the libraries and read the data frame using pandas


2.Calculate the null values present in the dataset and apply label encoder.


3.Determine test and training data set and apply decison tree regression in dataset.


4.calculate Mean square error,data prediction and r2.

## Program:
```
Program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.
Developed by: AJAYRAJA RATHINAM T
RegisterNumber: 212224240006
```
```

import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn import metrics

data = pd.read_csv("Salary.csv")
print(data.head(), "\n")
print(data.info(), "\n")
print(data.isnull().sum(), "\n")

le = LabelEncoder()
data["Position"] = le.fit_transform(data["Position"])
print(data.head(), "\n")

x = data[["Position", "Level"]]
y = data["Salary"]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=2)

dt = DecisionTreeRegressor(random_state=2)
dt.fit(x_train, y_train)

y_pred = dt.predict(x_test)

mse = metrics.mean_squared_error(y_test, y_pred)
r2 = metrics.r2_score(y_test, y_pred)

print(mse)
print(r2)

print(dt.predict([[5, 6]])[0])



```

## Output:
<img width="387" height="384" alt="image" src="https://github.com/user-attachments/assets/b8490509-59f2-408a-923c-b4c7442cd158" />



<img width="217" height="103" alt="image" src="https://github.com/user-attachments/assets/3f2eee39-469c-449f-927c-a51dae2afcdc" />



<img width="1324" height="99" alt="image" src="https://github.com/user-attachments/assets/f9cc1b40-9383-4d6b-8fad-ed7d31c7b2ea" />





## Result:
Thus the program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee is written and verified using python programming.
