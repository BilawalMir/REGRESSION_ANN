# Boston Housing Price Prediction

## Overview
This project implements data preprocessing, exploratory data analysis (EDA), and machine learning models to predict housing prices using the Boston Housing dataset. It includes data normalization, visualization, correlation analysis, and model training using both a Neural Network (Keras) and Linear Regression.

## Dataset
The dataset used in this project is the Boston Housing dataset, obtained from the UCI Machine Learning Repository:

**URL:** [Boston Housing Dataset](https://archive.ics.uci.edu/ml/machine-learning-databases/housing/housing.data)

The dataset consists of 506 observations with 14 attributes:

- CRIM: Per capita crime rate by town
- ZN: Proportion of residential land zoned for large lots
- INDUS: Proportion of non-retail business acres per town
- CHAS: Charles River dummy variable
- NOX: Nitrogen oxide concentration
- RM: Average number of rooms per dwelling
- AGE: Proportion of owner-occupied units built before 1940
- DIS: Weighted distances to employment centers
- RAD: Accessibility to radial highways
- TAX: Property tax rate per $10,000
- PTRATIO: Pupil-teacher ratio
- B: Proportion of Black residents
- LSTAT: Percentage of lower status population
- MEDV: Median value of owner-occupied homes ($1000s)

## Requirements
Ensure the following libraries are installed:
```bash
pip install pandas numpy scikit-learn matplotlib seaborn tensorflow
```

## Steps

### 1. Load Data
The dataset is loaded and converted into a Pandas DataFrame.
```python
import pandas as pd
import numpy as np

url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/housing/housing.data'
column_names = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']
data = pd.read_csv(url, delim_whitespace=True, names=column_names)
```

### 2. Data Preprocessing
Summary statistics and missing values are checked. The dataset is then normalized using MinMaxScaler.
```python
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(data)
df_scaled = pd.DataFrame(data_scaled, columns=column_names)
```

### 3. Data Visualization
Box plots and correlation matrices are generated to analyze feature distributions and relationships.
```python
import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(10,5))
df_scaled.boxplot()
plt.show()
```

### 4. Train-Test Split
The dataset is split into training and testing sets.
```python
from sklearn.model_selection import train_test_split

X = df_scaled.drop('MEDV', axis=1)
Y = df_scaled['MEDV']
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=5)
```

### 5. Neural Network Model (Keras)
A simple feedforward neural network is trained to predict housing prices.
```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

model = Sequential()
model.add(Dense(20, input_dim=13, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(1, activation='linear'))

model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])
model.fit(X_train, Y_train, epochs=100, verbose=1)
```

### 6. Linear Regression Model
A simple linear regression model is trained.
```python
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

linear_model = LinearRegression()
linear_model.fit(X_train, Y_train)
Y_pred = linear_model.predict(X_test)

mse = mean_squared_error(Y_test, Y_pred)
print("Mean Squared Error:", mse)
```

### 7. Regularization: Lasso and Ridge Regression
Lasso and Ridge Regression are applied to regularize the model.
```python
from sklearn.linear_model import Lasso, Ridge

lasso = Lasso(alpha=0.1)
lasso.fit(X_train, Y_train)
ridge = Ridge(alpha=0.1)
ridge.fit(X_train, Y_train)
```

## Results
- The trained models predict housing prices based on input features.
- Neural Network and Linear Regression models are compared.
- Regularization techniques (Lasso and Ridge) help in feature selection and model tuning.


## Conclusion
This project demonstrates the application of machine learning for housing price prediction. Feature engineering, data scaling, visualization, and model comparisons are performed to improve model performance.

## Author
Your Name

## License
This project is open-source and free to use.

