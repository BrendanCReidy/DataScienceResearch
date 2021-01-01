# import the necessary packages
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

dataset.head()

dataset = dataset.sample(frac=1)

fraud = dataset.loc[dataset['Class'] == 1]
normal = dataset.loc[dataset['Class'] == 0][:(492*10)]

normal_distributed_data = pd.concat([fraud, normal])

sample_data = normal_distributed_data.sample(frac=1, random_state=42)

sample_data.head()

X = sample_data.drop('Class', axis=1)
y = sample_data['Class']
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=21)

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
rf = RandomForestClassifier()
rf_params = {'n_estimators': [100,200,500],
            'max_features': [3,5,7],
            'min_samples_split':[5,10,20]}
rf_cv_model = GridSearchCV(rf, rf_params, cv=7, n_jobs=-1, verbose=1).fit(X_train, y_train)
best_params = rf_cv_model.best_params_
print(best_params)
rf = RandomForestClassifier(max_features=best_params['max_features'], min_samples_split=best_params['min_samples_split'], n_estimators=best_params['n_estimators']).fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)
from imblearn.metrics import classification_report_imbalanced, sensitivity_specificity_support
print('sensitivity and specificity:', sensitivity_specificity_support(y_test, y_pred_rf, average='micro', labels=pd.unique(dataset.Class)))
print(classification_report_imbalanced(y_test, y_pred_rf))
from sklearn.metrics import classification_report, accuracy_score,precision_score,recall_score,f1_score,matthews_corrcoef
from sklearn.metrics import confusion_matrix

n_errors = (y_pred_rf != y_test).sum()
print('{}: {}'.format("Random Forest", n_errors))
print(accuracy_score(y_test, y_pred_rf))
print(classification_report(y_test, y_pred_rf))


LABELS = ['Normal', 'Fraud']
conf_matrix = confusion_matrix(y_test, y_pred_rf)
plt.figure(figsize=(6, 6))
sns.heatmap(conf_matrix, xticklabels=LABELS, yticklabels=LABELS, annot=True, fmt="d")
plt.title("Confusion matrix")
plt.ylabel('True class')
plt.xlabel('Predicted class')
plt.show()

# Run classification metrics
plt.figure(figsize=(6, 5))