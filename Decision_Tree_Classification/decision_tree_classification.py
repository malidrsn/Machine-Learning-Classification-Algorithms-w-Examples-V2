# çoklu bağımsız değişkenlerde hangi future ile başlamalıyız ağaca sorusu için entropi kavramı ortaya çıkmıştır.
# enformasyon kazanımı olarak geçer
# ID3 algorithm* - c4.5 algorithm- cart algorithm
# yaş veya diğer değerler için enformasyon işlemi yapılır ve buna göre hangi future ile başlamamız gerektiği belirlenir.
# Gain(age) = info(total)-info(age) bu işlem yapılarak kazançlar belirlenir total=bağımlı değişken
# gain değeri yüksek olan future ağaç için seçilir
# information gain = ID3

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

# SVM Destek Vektör Makineleri support vector classifier
from sklearn.svm import SVC

svm = SVC(kernel="rbf")  # kernel = linear, rbf*, poly, sigmoid
svm.fit(X_train, y_train)

y_pred_svm = svm.predict(X_test)

con_matrix_svm = confusion_matrix(y_test, y_pred_svm)
print("confusion matrix_svm\n", con_matrix_svm)
print("score SVM", svm.score(X_test, y_test))

# naive bayes
from sklearn.naive_bayes import GaussianNB

gnb = GaussianNB()
gnb.fit(X_train, y_train)

y_pred_gnb = gnb.predict(X_test)
con_matrix_gnb = confusion_matrix(y_test, y_pred_gnb)
print("confusion matrix_gnb\n", con_matrix_gnb)
print("score GNB", svm.score(X_test, y_test))

# decision Tree
print("decision tree classifier")
from sklearn.tree import DecisionTreeClassifier

dec_tree = DecisionTreeClassifier(criterion="entropy")  # hangi tree algorithm kullanılacağı default = gini
dec_tree.fit(X_train, y_train)
y_pred_dtc = dec_tree.predict(X_test)
print(y_pred_dtc)
print(y_test)
con_matrix_dtc = confusion_matrix(y_test, y_pred_dtc)
print("confusion matrix_dtc\n", con_matrix_dtc)
print("score DTC", svm.score(X_test, y_test))
