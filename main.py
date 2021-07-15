import os

import sklearn.metrics
import joblib
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import sklearn as skl
from sklearn import svm
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import datasets
from sklearn.datasets import fetch_openml

import pandas as pd

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.datasets import cifar10

#Building an SVM
# x = tf.constant([[1, 2, 3], [2, 3, 4]], dtype=tf.float32)
# x = tf.zeros((4, 4))
# x = tf.random.normal((1, 4), mean=0, stddev=2)
# x = tf.constant([1, 2, 3])
# y = tf.constant([[4], [2], [3]])
# print(x, y)
#
# x= tf.range(9)
# print(x)
# x=tf.reshape(x,(3,3))
# print(x)
# x=tf.transpose(x)
# print(x)
#
# iris = datasets.load_iris()
# # print(iris.feature_names)
# # print(iris.values())
# # print(iris.target_names)
# # print(iris.DESCR)
#
# mice =fetch_openml(name='miceprotein',version=4)
# print(mice.values())
#
# total_data = pd.read_csv("Seed_Data.csv")
# # print(total_data.describe())
# X = total_data.iloc[:, 0:7]
# y = total_data.iloc[:, 7]
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=13)
# sc = StandardScaler()
# X_train = sc.fit_transform(X_train)
# X_test = sc.transform(X_test) #we do not need to use fit_transform as the same data from before such as mean normalization
# #would be used to scale the test set when we use transform
#
# clf = svm.SVC()
# clf.fit(X_train,y_train)
# pred_clf =clf.predict(X_test)
# print(sklearn.metrics.accuracy_score(y_test,pred_clf))
#
# #export model using the joblib library
# model = joblib.dump(clf,"model.pkl")





