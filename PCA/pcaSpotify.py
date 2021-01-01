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
        eigVals = np.sort(eigVals)
        eigVals = eigVals[::-1]
        coverage = np.sum(eigVals[0:n_dims]) / np.sum(eigVals)
        return coverage

data = pd.read_csv("/Users/brendan/Documents/AlgorithmPracticePlayground/Data/spotify.csv")
allGenres = {}
for i in range(len(data)):
    genres = data.loc[i, "genres"]
    genres = genres.replace("[", "")
    genres = genres.replace("]", "")
    genres = genres.split(",")
    for genre in genres:
        if genre in allGenres:
            allGenres[genre]+=1
        else:
            allGenres[genre]=0

sorted_dict = {}
sorted_keys = sorted(allGenres, key=allGenres.get, reverse=True)
inorder = []
for w in sorted_keys:
    sorted_dict[w] = allGenres[w]
    inorder.append(w)
"""
print(inorder[1:80])

includedGenres = inorder[2]
includedGenres += inorder[3]
includedGenres = [inorder[40], inorder[37]]

print(includedGenres)

newData = pd.DataFrame(data, columns = data.columns) 

for i in range(len(data)):
    genres = data.loc[i, "genres"]
    genres = genres.replace("[", "")
    genres = genres.replace("]", "")
    genres = genres.split(",")
    isin = False
    newVal = -1
    for genre in genres:
        if genre in includedGenres:
            isin = True
            newVal = includedGenres.index(genre)
            print(genre)
            break
    if not isin:
        data.loc[i,"genres"] = "N"
    else:
        data.loc[i,"genres"] = newVal
"""

data = data[data["genres"] != "N"]

print(data)

labs = data["genres"]

data.drop(['genres','artists','key','mode','count'], axis=1, inplace=True)
data = data.sample(frac=1)
X = data.to_numpy()

pca = PCA(X)
for i in range(12):
    print("" + str(i) + ":", pca.getCoverage(i))
meanAdjusted = pca.analyze(4)
eigVals, eigVecs = pca.getEignen()
arrInds = eigVals.argsort()
eigVecs = eigVecs[arrInds[::-1]]

w = meanAdjusted[0]
x = meanAdjusted[1]
y = meanAdjusted[2]
ax = plt.axes()
""" #uncomment for 3d
plt.scatter(w, x)
"""
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(w, x, y)
#"""
plt.show()
