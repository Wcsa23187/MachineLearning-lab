from math import log

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

class LogisticRegression:

    def __init__(self, penalty="l1", epochs=100, batch_size=16, gamma=0.001, fit_intercept=True):
        # we have 11 features and the bias b
        self.w = np.random.normal(size=(12, 1))
        self.batch_size = batch_size
        self.epochs = epochs
        self.gamma = gamma
        self.penalty = penalty

    def sigmoid(self, x):
        """The logistic sigmoid function"""
        return 1 / (1 + np.exp(-x))

    def data_shuffle(self, train):
        '''
        shuffle the data and give the split the data to batch
        '''
        batch_size = self.batch_size  # goodfellow say Preferably no more than 16
        np.random.shuffle(train)
        lenth_train = train.shape[0]
        data = [train[i:i + batch_size]
                for i in range(0, lenth_train, batch_size)]
        return data

    def fit_BGD(self, X, y, lr=0.007):
        """
        Fit the regression coefficients via gradient descent or other methods
        MSE :loss = (self.sigmoid((self.w.T*x)[0,0])-y)**2
        MSE partial : dw=(self.sigmoid(self.w.T*x)-y)[0,0]*x
        """
        list = []
        gamma = self.gamma
        penalty = self.penalty
        train = pd.concat([X, y], axis=1)
        train = np.array(train)
        epochs = self.epochs
        for i in range(epochs):
            loss_per_epoch = 0
            b_loss = 0
            idx = 0
            for batch in self.data_shuffle(train):
                dp = 0
                d_w = np.zeros(shape=(12, 1))
                d_p = np.zeros(shape=(12, 1))
                for k in batch:
                    idx += 1
                    x0 = np.r_[k[0:-1], 1]
                    x = np.mat(x0).T
                    y = k[-1]
                    if penalty == 'None':
                        loss = -y * np.log(self.sigmoid((self.w.T * x)[0, 0])) - (1 - y) * (
                            np.log(1 - self.sigmoid((self.w.T * x)[0, 0])))
                        dw = (self.sigmoid(self.w.T * x) - y)[0, 0] * x
                        d_w += dw
                        b_loss += loss
                        self.w -= lr * d_w / self.batch_size
                    if penalty == 'l1':
                        loss = -y * np.log(self.sigmoid((self.w.T * x)[0, 0])) - (1 - y) * (
                            np.log(1 - self.sigmoid((self.w.T * x)[0, 0])))
                        dw = (self.sigmoid(self.w.T * x) - y)[0, 0] * x+ gamma * (
                                   np.abs(np.mat(self.w).T) * np.mat(np.ones(self.w.size)).T)[0, 0]
                        dp = gamma * np.sign(self.w)
                        d_w += dw
                        d_p += dp
                        b_loss += loss
                    if penalty == 'l2':
                        loss = -y * np.log(self.sigmoid((self.w.T * x)[0, 0])) - (1 - y) * (
                            np.log(1 - self.sigmoid((self.w.T * x)[0, 0])))
                        dw = (self.sigmoid(self.w.T * x) - y)[0, 0] * x+ gamma * (np.mat(self.w).T * self.w)[0, 0]
                        dp = gamma * self.w
                        d_w += dw
                        d_p += dp
                        b_loss += loss

                self.w -= (lr * d_w + dp)/ self.batch_size
            loss_per_epoch += b_loss
            list.append(loss_per_epoch / idx)
            print("Epoch [{:03d}]  loss is {:.3f}".format(i, loss_per_epoch / idx))
        return list

    def fit_GD(self, X, y, lr=0.001):
        """
        Fit the regression coefficients via gradient descent or other methods
        """
        list = []
        penalty = self.penalty
        gamma = self.gamma
        train = pd.concat([X, y], axis=1)
        train = np.array(train)
        epochs = self.epochs
        loss_per_epoch = 0
        idx = 0
        for i in range(epochs):
            for k in train:
                idx += 1
                x0 = np.r_[k[0:-1], 1]
                x = np.mat(x0).T
                y = k[-1]
                if penalty == 'None':
                    loss = -y * np.log(self.sigmoid((self.w.T * x)[0, 0])) - (1 - y) * (
                        np.log(1 - self.sigmoid((self.w.T * x)[0, 0])))
                    dw = (self.sigmoid(self.w.T * x) - y)[0, 0] * x
                    self.w -= lr * dw
                if penalty == 'l1':
                    loss = -y * np.log(self.sigmoid((self.w.T * x)[0, 0])) - (1 - y) * (
                        np.log(1 - self.sigmoid((self.w.T * x)[0, 0]))) + gamma * (
                                   np.abs(np.mat(self.w).T) * np.mat(np.ones(self.w.size)).T)[0, 0]
                    dw = (self.sigmoid(self.w.T * x) - y)[0, 0] * x
                    self.w -= lr * dw + gamma * np.sign(self.w)

                if penalty == 'l2':
                    loss = -y * np.log(self.sigmoid((self.w.T * x)[0, 0])) - (1 - y) * (
                        np.log(1 - self.sigmoid((self.w.T * x)[0, 0]))) + gamma * (np.mat(self.w).T * self.w)[0, 0]
                    dw = (self.sigmoid(self.w.T * x) - y)[0, 0] * x
                    self.w -= lr * dw + gamma * self.w
                loss_per_epoch += loss
            list.append(loss_per_epoch / idx)
            print("Epoch [{:03d}]  loss is {:.3f}".format(i, loss_per_epoch / idx))
        return list

    def predict(self, X, y):
        """
        Use the trained model to generate prediction probabilities on a new
        collection of data points.
        """
        final=[]
        y_test = np.array(y)
        test = np.array(X)
        list = []
        for k in test:
            x0 = np.r_[k[0:], 1]
            x = np.mat(x0).T
            y = self.sigmoid((self.w.T * x)[0, 0])
            final.append(y)
            if y >= 0.5:
                list.append(1)
            else:
                list.append(0)
        true = 0
        for i in range(len(list)):
            if list[i] == y_test[i]:
                true += 1
        print(true / len(list))
        print(final)

    def loss_graph(self,list):
        epochs=self.epochs
        x = [i+1 for i in range(epochs)]
        sns.set(style="whitegrid")  # 这是seaborn默认的风格
        # 使用标记而不是破折号来识别组
        plt.xlabel('epochs')  # 添加x轴的名称
        plt.ylabel('loss')
        sns.lineplot(x=x, y=list,
                          markers=True, dashes=False)
        plt.show()



