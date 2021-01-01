import numpy as np
import math
import pandas as pd
import random
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn import datasets
from numpy import linalg as LA

# import some data to play with
def normalize(data):
    data = np.transpose(data)
    for i in range(len(data)):
        aMax = np.max(data[i])
        aMin = np.min(data[i])
        for j in range(len(data[i])):
            data[i][j] = (data[i][j] - aMin) / (aMax - aMin)

    return np.transpose(data)

data = pd.read_csv("/Users/brendan/Documents/AlgorithmPracticePlayground/Data/spotify.csv")
data.drop(['genres','artists','key','mode','count'], axis=1, inplace=True)
data = data.sample(frac=1)
X = data.to_numpy()

import tensorflow as tf
from tensorflow import keras
from keras import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import Adam

print(X.shape)

model = Sequential(
    [
    Dense(11),
    Dense(256, activation="relu"),
    Dense(128, activation="relu"),
    Dense(1, activation="tanh"),
    Dense(128, activation="relu"),
    Dense(256, activation="relu"),
    Dense(11, activation="sigmoid")
    ]
)


#"""
X = normalize(X)
np.random.shuffle(X)

train = X[5000:]
val = X[:5000]

print(X.shape)
print(val.shape)
print(train.shape)
model.compile(optimizer='adam', loss='binary_crossentropy', metrics="accuracy")

history = model.fit(
    train,
    train,
    epochs=50,
    batch_size=10,
    shuffle=True,
    # We pass some validation for
    # monitoring validation loss and metrics
    # at the end of each epoch
    validation_data=(val, val),
)
model.summary()
#"""
from sklearn.metrics import classification_report, accuracy_score,precision_score,recall_score,f1_score,matthews_corrcoef

pred = model.predict(val)

print(pred[0])
print(val[0])

#print(classification_report(val, pred))
#"""