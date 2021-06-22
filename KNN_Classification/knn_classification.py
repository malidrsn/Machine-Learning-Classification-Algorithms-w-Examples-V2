# sayısal olmayan değerlerin tahminlerine classification denir. Kategorik veriler üzerinde

from sklearn.impute import SimpleImputer
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# veri Önişleme
# Veri Yükleme
veriler = pd.read_csv("veriler.csv")

x = veriler.iloc[:, 1:4].values  # bağımsız değişkenler
y = veriler.iloc[:, 4:].values  # bağımlı değişken

# önemli test ve eğitim verileri olarlak bölmeye yarar eğitilecek 2/3 test 1/3 olacak şekilde
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=0)
# print(x_train)
# print(x_test)
# print(y_train)
# print(y_test)

# verilerin standartscaler küütphanesi ile ölçeklenmesi standart sapma kullandık
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
X_train = sc.fit_transform(x_train)
X_test = sc.transform(x_test)

# logistic regression
from sklearn.linear_model import LogisticRegression

log_reg = LogisticRegression(random_state=0)
log_reg.fit(X_train, y_train)

y_pred = log_reg.predict(X_test)
print(y_pred)
print(y_test)

# confusion matrix Karışıklık matrisi
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred)
print(cm)

print("-------------------KNN------------------")
# KNN
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=1, metric="minkowski")
knn.fit(X_train, y_train)
y_pred_knn = knn.predict(X_test)
con_matrix_knn = confusion_matrix(y_test, y_pred_knn)
print("confusion matrix_knn", con_matrix_knn)
print("score", knn.score(X_test, y_test))
