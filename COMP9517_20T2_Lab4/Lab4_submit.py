# -*- coding: utf-8 -*-
"""
Created on Mon Jul 13 10:44:36 2020

@author: sanch
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier


digits = load_digits()
#plt.imshow(np.reshape(digits.data[0], (8, 8)), cmap='gray')
#plt.title('Label: %i\n' % digits.target[0], fontsize=25)

x_train, x_test, y_train, y_test = train_test_split(digits.data, digits.target, test_size = 0.25,random_state=6)

def KNN():
    x_train, x_test, y_train, y_test = train_test_split(digits.data, digits.target, test_size = 0.25,random_state=6)
    #Initialize the classifier model.
    ##   n_neighbors=5 because for n<5 accuracy remains the same (~0.989) and for n>5 accuracy starts decreasing
    ## weights are used to distribute wieghts equally among neighborhood points. as data is unkown
    ## if  manhatten distance is used(p=1) there's a l=slight increase in the accuracy of the model
    estimator=KNeighborsClassifier(n_neighbors=5,weights='uniform',p=1, metric='minkowski')
    #Fit the model to the training data.
    estimator.fit(x_train,y_train)
    y_predict=estimator.predict(x_test)
    #Use the trained/fitted model to evaluate the test data.
    accuracy_score=metrics.accuracy_score(y_test, y_predict)
    #For each of the three classifiers, evaluate the digit classification performance by
    #calculating the accuracy, the recall, and the confusion matrix.
    print("KNN Accuracy: %0.3f"%accuracy_score,end='\t')
    recall_score=metrics.recall_score(y_test, y_predict, average='macro')
    print("Recall: %0.3f"%recall_score)
    confusion_matrix=metrics.confusion_matrix(y_test, y_predict)
    return confusion_matrix

def DT():
    x_train, x_test, y_train, y_test = train_test_split(digits.data, digits.target, test_size = 0.25,random_state=6)
    #The entropy is a metric frequently used to measure the uncertainty in a distribution
    #Since we want our subnodes to be homogenous, we desire to split on the feature with the minimum value for entropy
    estimator=DecisionTreeClassifier(criterion="entropy")
    estimator.fit(x_train,y_train)
    y_predict=estimator.predict(x_test)
    accuracy_score=metrics.accuracy_score(y_test, y_predict)
    print("DT Accuracy: %0.3f"%accuracy_score,end='\t')
    recall_score=metrics.recall_score(y_test, y_predict, average='macro')
    print("Recall: %0.3f"%recall_score)

def SGD():
    x_train, x_test, y_train, y_test = train_test_split(digits.data, digits.target, test_size = 0.25,random_state=6)
    estimator=SGDClassifier()
    estimator.fit(x_train,y_train)
    y_predict=estimator.predict(x_test)
    accuracy_score=metrics.accuracy_score(y_test, y_predict)
    print("SGD Accuracy: %0.3f"%accuracy_score,end='\t')
    recall_score=metrics.recall_score(y_test, y_predict, average='macro')
    print("Recall: %0.3f"%recall_score)

print()
test_size=x_test.shape[0]/(x_train.shape[0]+x_test.shape[0])
print('Test size = %0.2f'%test_size)
confusion_matrix=KNN()
SGD()
DT()
print()
print('KNN Confusion Matrix:')
print(confusion_matrix)