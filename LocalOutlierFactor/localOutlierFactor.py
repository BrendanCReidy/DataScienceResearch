import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
data = pd.read_csv("/Users/brendan/Documents/AlgorithmPracticePlayground/Data/creditcard.csv")
print(data.columns)
data.head()

from sklearn.preprocessing import StandardScaler, RobustScaler

dataset = data

std_scaler = StandardScaler()
rob_scaler = RobustScaler()

dataset['amount_scale'] = rob_scaler.fit_transform(dataset['Amount'].values.reshape(-1,1))
dataset['time_scale'] = rob_scaler.fit_transform(dataset['Time'].values.reshape(-1,1))

dataset.drop(['Time','Amount'], axis=1, inplace=True)

amount_scale = dataset['amount_scale']
time_scale = dataset['time_scale']

dataset.drop(['amount_scale', 'time_scale'], axis=1, inplace=True)
dataset.insert(0, 'amount_scale', amount_scale)
dataset.insert(1, 'time_scale', time_scale)

data = dataset

Fraud = data[data['Class'] == 1]
Valid = data[data['Class'] == 0]

outlier_fraction = len(Fraud)/float(len(Valid))
print(outlier_fraction)

print('Fraud Cases: {}'.format(len(data[data['Class'] == 1])))
print('Valid Transactions: {}'.format(len(data[data['Class'] == 0])))
X=data.drop(['Class'], axis=1)
Y=data["Class"]
#getting just the values for the sake of processing (its a numpy array with no columns)
X_data=X.values
Y_data=Y.values

from sklearn.model_selection import train_test_split
# Split the data into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X_data, Y_data, test_size = 0.33, random_state = 21)

from sklearn.neighbors import LocalOutlierFactor
from sklearn.datasets import make_blobs
from numpy import quantile, where, random

lof = LocalOutlierFactor(n_neighbors=20, contamination=0.03)
y_pred = lof.fit_predict(X_train)
lofs_index = where(y_pred==-1)
print(Y_train.shape)
print(y_pred.shape)

acc = 0
for i in range(len(y_pred)):
    if(y_pred[i]==-1):
        y_pred[i]=1
    else:
        y_pred[i]=0
from sklearn.metrics import classification_report, accuracy_score,precision_score,recall_score,f1_score,matthews_corrcoef
from sklearn.metrics import confusion_matrix

print(accuracy_score(Y_train, y_pred))
print(classification_report(Y_train, y_pred))