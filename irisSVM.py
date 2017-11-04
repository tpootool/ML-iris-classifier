import numpy
import pandas
import sklearn
from sklearn import datasets
from sklearn import svm

iris = datasets.load_iris()
classification_model = svm.SVC()

train_data = 130 # Set the number of samples used for training

classification_model.fit(iris.data[:train_data],iris.target[:train_data])
result = classification_model.predict(iris.data[train_data:])

match = 0.0
total = 0.0

for i in range(150-train_data):
    if (result[i] == iris.target[train_data+i]):
        match += 1
    total += 1

print ((match/total)*100)
# print(result == iris.target[train_data:])