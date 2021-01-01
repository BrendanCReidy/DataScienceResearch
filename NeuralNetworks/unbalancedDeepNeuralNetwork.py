# import the necessary packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
data = pd.read_csv("/Users/brendan/Documents/AlgorithmPracticePlayground/Data/creditcard.csv")
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

dataset.head()

dataset = dataset.sample(frac=1)

data = dataset

Fraud = data[data['Class'] == 1]
Valid = data[data['Class'] == 0]

outlier_fraction = len(Fraud)/float(len(Valid))
print(outlier_fraction)

print('Fraud Cases: {}'.format(len(data[data['Class'] == 1])))
print('Valid Transactions: {}'.format(len(data[data['Class'] == 0])))
X=data.drop(['Class'], axis=1)
Y=data["Class"]

print(X)
print(Y)
#getting just the values for the sake of processing (its a numpy array with no columns)
X_data=X.values
Y_data=Y.values

from sklearn.model_selection import train_test_split
# Split the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(X_data, Y_data, test_size = 0.33, random_state = 21)

import tensorflow as tf
from tensorflow import keras
from keras import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import Adam

print(x_train.shape)
print(y_train.shape)

model = Sequential(
    [
    Dense(30),
    Dropout(0.2),  
    Dense(200, activation="tanh"),
    Dropout(0.2),
    Dense(1, activation="sigmoid")
    ]
)

model.compile(optimizer='adam', loss='binary_crossentropy', metrics = ['accuracy'])

history = model.fit(
    x_train,
    y_train,
    batch_size=20,
    epochs=20,
    # We pass some validation for
    # monitoring validation loss and metrics
    # at the end of each epoch
    validation_data=(x_test, y_test),
)
from sklearn.metrics import classification_report, accuracy_score,precision_score,recall_score,f1_score,matthews_corrcoef
from sklearn.metrics import confusion_matrix

pred = model.predict_classes(x_test)

classA = 0
predA = 0
for i in range(len(pred)):
    if y_test[i] >= 0.5:
        classA+=1
    if pred[i] >= 0.5:
        predA += 1

print("Class A Accuracy:", (predA / float(classA)))

print(classification_report(y_test, pred))

LABELS = ['Normal', 'Fraud']
conf_matrix = confusion_matrix(y_test, pred)
plt.figure(figsize=(6, 6))
sns.heatmap(conf_matrix, xticklabels=LABELS, yticklabels=LABELS, annot=True, fmt="d")
plt.title("Confusion matrix")
plt.ylabel('True class')
plt.xlabel('Predicted class')
plt.show()

# Run classification metrics
plt.figure(figsize=(6, 5))


#"""