# Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee

## AIM:
To write a program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the standard libraries.

2. Upload the dataset and check for any null values using .isnull() function.

3. Import LabelEncoder and encode the dataset.

4. Import DecisionTreeRegressor from sklearn and apply the model on the dataset.

5. Predict the values of arrays.

6. Import metrics from sklearn and calculate the MSE and R2 of the model on the dataset.

7. Predict the values of array.

8. Apply to new unknown values.

## Program:
```python
/*
Program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.
Developed by: Sriram K
Register Number: 212222080052
*/

import pandas as pd
data=pd.read_csv("Salary.csv")
data.head()

data.info()

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data["Position"]=le.fit_transform(data["Position"])

x=data[["Position","Level"]]
x.head()

y=data[["Salary"]]
y.head()

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=2)
from sklearn.tree import DecisionTreeRegressor
dt=DecisionTreeRegressor()
dt.fit(x_train,y_train)
y_pred=dt.predict(x_test)
y_pred

from sklearn import metrics
mse=metrics.mean_squared_error(y_test,y_pred)
mse

r2=metrics.r2_score(y_test,y_pred)
r2

dt.predict([[5,6]])
```

## Output:
### Data
![DATA](https://github.com/user-attachments/assets/6fa792bf-4828-45ed-a288-bab03090245a)

### X_variables
![X_variables](https://github.com/user-attachments/assets/1ba98fbd-47a5-45ea-bb9a-45c98de61abc)

### y_variables
![y_variables](https://github.com/user-attachments/assets/ab09b1f7-5011-46d8-b10e-49ed4cc7f1d0)

### y_pred
![y_pred](https://github.com/user-attachments/assets/087ffa6a-5221-49ef-8cc0-f49a93bf4ff7)

### MSE
![MSE](https://github.com/user-attachments/assets/56fb7b9b-8c8d-4e05-a493-7f65bcef89a5)

### r^2
![r^2](https://github.com/user-attachments/assets/a73fdcb6-faa6-400f-a3fd-a360ace03544)

### predictions
![predictions](https://github.com/user-attachments/assets/8dc521eb-8094-4f26-bcf3-cbd966f662e0)

## Result:
Thus the program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee is written and verified using python programming.
