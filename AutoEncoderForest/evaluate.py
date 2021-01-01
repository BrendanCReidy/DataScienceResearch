import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
import AutoEncoderForest as aef
from sklearn.preprocessing import StandardScaler, RobustScaler

def loadModels(baseName, num):
    models = []
    for i in range(num):
        name = baseName + str(i) + ".h5"
        model = tf.keras.models.load_model(name)
        models.append(model)
    return models

def getData():
    data = pd.read_csv("/Users/brendan/Documents/AlgorithmPracticePlayground/Data/creditcard.csv")
    print(data.columns)
    data.head()

    Fraud = data[data['Class'] == 1]
    Valid = data[data['Class'] == 0]

    outlier_fraction = len(Fraud)/float(len(Valid))
    print(outlier_fraction)

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

    fraud = dataset.loc[dataset['Class'] == 1]
    normal = dataset.loc[dataset['Class'] == 0]#[:492]

    normal_distributed_data = pd.concat([fraud, normal])

    sample_data = normal_distributed_data.sample(frac=1, random_state=42)

    sample_data.head()

    X = sample_data.drop('Class', axis=1)
    y = sample_data['Class']

    X_data=X.values
    Y_data=y.values

    from sklearn import preprocessing

    min_max_scaler = preprocessing.MinMaxScaler()
    X_data = min_max_scaler.fit_transform(X_data)

    return X_data, Y_data

from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

X_data, Y_data = getData()
X_train, X_test, y_train, y_test = train_test_split(X_data, Y_data, test_size=0.33, random_state=21)

forest = aef.AutoEncoderForest(X_data)

#models = loadModels('25EpochModel/tempEncoderModel', 10)
#models2 = loadModels('10EpochModel/encoderModel', 20)
#models = models2 + models
models = loadModels('10EpochModel/encoderModel', 20)
forest.setModels(models)

print("Getting prediction...")
pred = forest.predict(X_data)
print(classification_report(Y_data, pred))

from sklearn.metrics import confusion_matrix

LABELS = ['Normal', 'Fraud']
conf_matrix = confusion_matrix(Y_data, pred)
plt.figure(figsize=(6, 6))
sns.heatmap(conf_matrix, xticklabels=LABELS, yticklabels=LABELS, annot=True, fmt="d")
plt.title("Confusion matrix")
plt.ylabel('True class')
plt.xlabel('Predicted class')
plt.show()

# Run classification metrics
plt.figure(figsize=(6, 5))