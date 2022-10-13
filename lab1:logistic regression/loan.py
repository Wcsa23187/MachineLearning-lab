import pandas as pd

df = pd.read_csv('loan.csv')

df.drop("Loan_ID", axis=1, inplace=True)
# Task1 deal with NULL rows, you can either choose to drop them or replace them with mean or other value
# Task1.A :I tink the Discrete Variables should be replaced with The element that occurs the most in a column
# Task1.B :and Numerical Variables  I will replace them with mean

# Task1.A
df['Gender'].fillna(df['Gender'].mode()[0], inplace=True)
df['Married'].fillna(df['Married'].mode()[0], inplace=True)
df['Education'].fillna(df['Education'].mode()[0], inplace=True)
df['Dependents'].fillna(df['Dependents'].mode()[0], inplace=True)
df['Self_Employed'].fillna(df['Self_Employed'].mode()[0], inplace=True)
df['Credit_History'].fillna(df['Credit_History'].mode()[0], inplace=True)
df['Loan_Amount_Term'].fillna(df['Loan_Amount_Term'].mode()[0], inplace=True)
# Task1.B
df['LoanAmount'].fillna(df['LoanAmount'].mean(), inplace=True)
# presentation


# Task2 deal with categorical features
# we focus on the columns which the Dtype is object

df.Gender = df.Gender.map({'Male': 1, 'Female': 0})
df.Married = df.Married.map({'Yes': 1, 'No': 0})
df.Education = df.Education.map({'Graduate': 1, 'Not Graduate': 0})
df.Self_Employed = df.Self_Employed.map({'Yes': 1, 'No': 0})
df.Property_Area = df.Property_Area.map({'Urban': 1, 'Rural': 2, 'Semiurban': 3})
df.Loan_Status = df.Loan_Status.map({'Y': 1, 'N': 0})
df.Dependents = df.Dependents.map({'0': 0, '1': 1, '2': 2, '3+': 3})

import numpy as np

# Task3 split the dataset into X_train, X_test, y_train, y_test
# Optional: you can also use  normalization
# Task3.A : we should min-max normalize the numerical variables and columns has more than two options
df = (df - df.min()) / (df.max() - df.min())

# Task3.B : split dataset into  X_train, X_test, y_train, y_test
# At first i will classify the dataset by the lable
df_yes = df[df['Loan_Status'] == 1]
df_no = df[df['Loan_Status'] == 0]
df_yes.reset_index(drop=True, inplace=True)  # reset the index
df_no.reset_index(drop=True, inplace=True)
# df.info() will get the length of the dataframe df_yes = 422 df_no = 192
length_yes = 422
length_no = 192
train_yes = df_yes[:int(4 * 422 / 5)]
test_yes = df_yes[int(4 * 422 / 5):]
train_no = df_no[:int(4 * 192 / 5)]
test_no = df_no[int(4 * 192 / 5):]
# cat them
train = pd.concat([train_yes, train_no], axis=0)  # 490
test = pd.concat([test_yes, test_no], axis=0)  # 124
# split the dataset into X_train, X_test, y_train, y_test

y_train = train['Loan_Status']
y_test = test['Loan_Status']
x_train = train.drop('Loan_Status', axis=1)
x_test = test.drop('Loan_Status', axis=1)

from Logistic import LogisticRegression
import matplotlib.pyplot as plt

# Task4 train your model and plot the loss curve of training
model = LogisticRegression(gamma=0.001,penalty='l2',epochs=500,batch_size=16)
# loss_list = model.fit_GD(X=x_train, y=y_train,lr=0.001)
loss_list = model.fit_BGD(X=x_train, y=y_train,lr=0.005)
model.predict(X=x_test, y=y_test)
model.loss_graph(list=loss_list)
