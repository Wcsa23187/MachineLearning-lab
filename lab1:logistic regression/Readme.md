## Logistic regression    ML lab1

This repository contains Logistic regression algorithm,with the Regularization L1 , L2  and None.

## Requirements

- seaborn==0.11.2 
- python==3.7.0 
- pandas==1.3.5
- numpy==1.21.5 
- matplotlib==3.5.3

## How to run Logistic Model.

```python
python loan.py
```

## Gradient Descent Choice

we provide the minibatch gradient descent and the direct gradient descent.
you can choose them by change the fit function, just like:

```python
model.fit_GD(X=x_train, y=y_train,lr=0.001)
model.fit_BGD(X=x_train, y=y_train,lr=0.01)
