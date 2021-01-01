import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
data = pd.read_csv("/Users/brendan/Documents/AlgorithmPracticePlayground/Data/creditcard.csv")
print(data.columns)
data.head()

Fraud = data[data['Class'] == 1]
Valid = data[data['Class'] == 0]

outlier_fraction = len(Fraud)/float(len(Valid))
print(outlier_fraction)

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

N = 1

dataset.head()

dataset = dataset.sample(frac=1)

fraud = dataset.loc[dataset['Class'] == 1]
normal = dataset.loc[dataset['Class'] == 0][:492*N]

normal_distributed_data = pd.concat([fraud, normal])

sample_data = normal_distributed_data.sample(frac=1, random_state=42)

sample_data.head()
print(sample_data)
X = sample_data.drop('Class', axis=1)
y = sample_data['Class']
X_data=X.values
Y_data=y.values

print(X_data)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_data, Y_data, test_size=0.001, random_state=21)


from sklearn.neighbors import LocalOutlierFactor
from sklearn.datasets import make_blobs
from numpy import quantile, where, random

lof = LocalOutlierFactor(n_neighbors=492, contamination=1/float(N+1))
y_pred = lof.fit_predict(X_train)
lofs_index = where(y_pred==-1)
print(y_train.shape)
print(y_pred.shape)

acc = 0
for i in range(len(y_pred)):
    if(y_pred[i]==-1):
        y_pred[i]=1
    else:
        y_pred[i]=0
from sklearn.metrics import classification_report, accuracy_score,precision_score,recall_score,f1_score,matthews_corrcoef
from sklearn.metrics import confusion_matrix


LABELS = ['Normal', 'Fraud']
conf_matrix = confusion_matrix(y_train, y_pred)
plt.figure(figsize=(6, 6))
sns.heatmap(conf_matrix, xticklabels=LABELS, yticklabels=LABELS, annot=True, fmt="d");
plt.title("Confusion matrix")
plt.ylabel('True class')
plt.xlabel('Predicted class')
plt.show()

# Run classification metrics
plt.figure(figsize=(6, 5))

print(accuracy_score(y_train, y_pred))
print(classification_report(y_train, y_pred))