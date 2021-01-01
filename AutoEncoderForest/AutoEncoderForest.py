

import tensorflow as tf
import numpy as np
from tensorflow import keras
from sklearn.metrics import accuracy_score
from keras import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import Adam

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}

class AutoEncoderForest():
    def __init__(self, data, sample_ratio = 0.1):
        self.data = data
        self.sample_ratio = sample_ratio
        self.n_features = data.shape[1]
        self.models = None
        self.avgs = None
        self.sample_size = int(sample_ratio*len(data))
        self.n_forests = int(len(self.data) / float(self.sample_size))

    def split_data(self):
        dataList = []
        print(len(self.data))
        print(self.sample_size)
        for i in range(1,self.n_forests):
            print(i*self.sample_size)
            tempData = self.data[(i-1)*self.sample_size:i*self.sample_size]
            dataList.append(tempData)
        tempData = self.data[(self.n_forests-1)*self.sample_size:]
        dataList.append(tempData)
        return dataList

    def getAvgDiff(self):
        x_data = self.data
        if self.models==None:
            print("Error: use model.fit() before getting avg")
            return None
        self.avgs=[]
        for i in range(len(self.models)):
            avg = 0
            model = self.models[i]
            pred = model.predict(x_data)
            diff = np.abs(x_data - pred)
            localMax = 0
            for j in range(len(diff)):
                argDiff = np.sum(diff[j]) / len(diff[j])
                avg+=argDiff
                if(argDiff>localMax):
                    localMax = argDiff
            avg/=float(len(diff))
            print(localMax, avg)
            self.avgs.append(avg)

    def getModels(self):
        return self.models

    def setModels(self, models):
        self.models = models
        self.getAvgDiff()

    def predict(self, x_data):
        if self.models==None:
            print("Error: use model.fit() before making predictions")
            return None
        
        predArrs = np.zeros((len(x_data)))
        for i in range(len(self.models)):
            model = self.models[i]
            pred = model.predict(x_data)
            diff = np.abs(x_data - pred)
            thresh = self.avgs[i] + .02
            for j in range(len(diff)):
                argDiff = np.sum(diff[j]) / len(diff[j])
                if(argDiff>=thresh):
                    predArrs[j]+=1

        for i in range(len(predArrs)):
            if predArrs[i] >= 5:
               predArrs[i]=1
            else:
                predArrs[i]=0
        return predArrs
        
    def fit(self):

        self.models = []
        featureSize = self.n_features
        innerLayerSize = int(featureSize/2)
        #inner2Layer = int(featureSize/3)
        middleLayerSize = int(featureSize/4)

        data = self.split_data()

        for i in range(len(data)):

            print("Training forest:", i+1,"/",len(data))

            batch = data[i]

            model = Sequential(
                [
                Dense(featureSize),
                Dropout(0.1),
                Dense(innerLayerSize, activation="relu"),
                Dropout(0.1),
                Dense(middleLayerSize, activation="relu"),
                Dropout(0.1),
                Dense(innerLayerSize, activation="relu"),
                Dropout(0.1),
                Dense(featureSize, activation="sigmoid")
                ]
            )

            model.compile(optimizer='adam', loss='binary_crossentropy', metrics="accuracy")

            history = model.fit(
                batch,
                batch,
                epochs=50,
                batch_size=5,
                shuffle=True,
                # We pass some validation for
                # monitoring validation loss and metrics
                # at the end of each epoch
            )

            self.models.append(model)

        
        self.getAvgDiff()