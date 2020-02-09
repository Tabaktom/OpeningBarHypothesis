import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier
import numpy as np

df = pd.read_csv('simple_features_SPY.txt')


labels = df.Label
df= df.drop(columns = ['Label'])
X_train, X_test, y_train, y_test = train_test_split(df, labels, shuffle = True, test_size= 0.3)
dates = X_train.Date +  X_test.Date
X_train = X_train.drop(columns = ['Date'])
X_test = X_test.drop(columns = ['Date'])

logistic = LogisticRegression(multi_class='multinomial', solver = 'lbfgs')


xgb = GradientBoostingClassifier(learning_rate=0.01, n_estimators=50)
trained_logistic = xgb.fit(X_train, y_train)
train_pred = trained_logistic.predict(X_train)
test_pred = trained_logistic.predict(X_test)

train_classification = classification_report(y_train, train_pred)
test_classification = classification_report(y_test, test_pred)

print('Training classification Martix:')
print(train_classification)
print('Test classification Matrix:')
print(test_classification)



