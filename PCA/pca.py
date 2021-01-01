import numpy as np
import math
import random
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn import datasets
from numpy import linalg as LA

# import some data to play with
iris = datasets.load_iris()

def normalize(data):
    data = np.transpose(data)
    for i in range(len(data)):
        aMax = np.max(data[i])
        aMin = np.min(data[i])
        for j in range(len(data[i])):
            data[i][j] = (data[i][j] - aMin) / (aMax - aMin)

    return np.transpose(data)

class PCA:
    def __init__(self, data):
        self.data = np.transpose(normalize(data))
        self.eigVals, self.loadings = self.getEignen()
    
    def getLoadings(self):
        return self.loadings

    def getMeanAdjusted(self):
        meanAdjusted = np.zeros((self.data.shape))
        for i in range(len(self.data)):
            mean = np.mean(self.data[i])
            for j in range(len(self.data[i])):
                meanAdjusted[i][j] = self.data[i][j] - mean
        return meanAdjusted

    def getEignen(self):
        meanAdjusted = self.getMeanAdjusted()
        covarianceMatrix = np.cov(meanAdjusted)
        w, v = LA.eig(covarianceMatrix)
        return (w, v)

    def setLoadings(self, loadings):
        self.loadings = loadings

    def analyze(self, n_dims):
        eigVals = self.eigVals
        loadings = self.getLoadings()
        meanAdjusted = self.getMeanAdjusted()

        arrInds = eigVals.argsort()
        scores = np.transpose(meanAdjusted)
        scores = np.dot(scores, loadings)
        scores = np.transpose(scores)
        scores = scores[arrInds[::-1]]
        return scores[0:n_dims]

    def getCoverage(self, n_dims):
        eigVals = self.eigVals
        coverage = np.sum(eigVals[0:n_dims]) / np.sum(eigVals)
        return coverage



X = iris.data
pca = PCA(X)
print(pca.getCoverage(2))
meanAdjusted = pca.analyze(3)
w = meanAdjusted[0]
x = meanAdjusted[1]
y = meanAdjusted[2]
labs = iris.target
ax = plt.axes()
#""" uncomment for 3d
plt.scatter(w, x, c=labs)
"""
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(w, x, y, c=labs)
#"""
plt.show()
