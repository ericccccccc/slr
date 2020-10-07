# -*- coding: utf-8 -*-
"""
Created on Wed Oct  7 13:16:58 2020

@author: Eric
"""

import numpy as np 
from scipy.stats import norm
import keras
from keras.models import Sequential
from keras.layers import Dense
from sklearn.linear_model import LinearRegression
from sklearn import preprocessing
import matplotlib.pyplot as plt

def crtCDF(x):
    if(type(x) == np.ndarray):
        loc = x.mean()
        scale = x.std()
        N = x.size
        pos = norm.cdf(x, loc, scale)*N
        return pos
    else:
        print("Wrong Type! x must be np.ndarray ~")    
        return

def main():
    
    x = np.random.random_integers(1000,size=100)
    x = np.sort(x)
    y = crtCDF(x)
    norm_x = preprocessing.scale(x)    # 標準化: 零均值化
    
    model = Sequential()
    model.add(Dense(1, input_dim=1, activation="linear"))
    sgd=keras.optimizers.SGD(lr=0.01)
    model.compile(loss="mse", optimizer=sgd, metrics=["mse"])
    model.fit(norm_x, y, epochs=100, batch_size=32, verbose=0)
    pred_y = model.predict(norm_x)
    print("pred_y:",pred_y)
    
    plt.title("Gradient Descent")
    plt.plot(x, y, '.',label="Origin")
    plt.plot(x, pred_y,'.',label="Model")
    plt.legend()
    plt.xlabel("Key")
    plt.ylabel("Pred_Pos = CDF(Key)")
    plt.show()
    
    x = np.reshape(x,(-1,1))
    model=LinearRegression()
    model.fit(x,y)
    plt.title("Least Square")
    plt.plot(x, y, '.',label="Origin")
    plt.plot(x, model.predict(x),'.',label="Model")
    plt.legend()
    plt.xlabel("Key")
    plt.ylabel("Pred_Pos = CDF(Key)")
    plt.show()
    
    
    
    
    
if __name__ == "__main__":
    main()