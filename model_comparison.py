#!/usr/bin/env python
# coding: utf-8
# author: miasya

"""
Classification of breast cancer data from digitized image of fine needle aspirate of mass from UCI Machine Learning.
Link to Kaggle Dataset: https://www.kaggle.com/uciml/breast-cancer-wisconsin-data/version/2

Columns: 
ID (1), diagnosis (2), and features (3-32)
Classes (diagnosis): 357 benign (B), 212 malignant (M)

I compare sklearn supervised classification algorithms to see which gives the best accuracy.
Parameters are selected manually based on the one that gives me the best performance.

Models to be considered:
    - Decision Tree | DecisionTreeClassifier()
            ^ pros: no standardization req, intuitive and explainable. 
    - Random Forest | RandomForestClassifier()
            ^ pros: no standardization req, good with class balance. cons: black box
    - LDA (Linear Discriminant Analysis) | LinearDiscriminantAnalysis()
    - Logistic Regression | LogisticRegression()
            ^ cons: only works with binary dependant variables, needs large sample size,
                tendency to overfit, variables must be indep
    
    - * K-Nearest Neighbours | KNeighborsClassifier()    
    - * SVM (Support Vector Machines) | SVC()

    - Naive Bayes | GaussianNB() 
            ^ also do Bernouilli since only 2 classes?
    
  
* careful! Might benefit greatly with standardization or normalization

    Unsupervised (for fun, or I should do another dataset probably. Try something new :D )
    - K-Means Clustering (Llyod Algorithm Version)
    - SVM again maybe?

List of sklearn classifiers that I'm interested in exploring maybe later (focus on bagging and boosting more):
    ExtraTreesClassifier()
    BaggingClassifier()
    GradientBoostingClassifier()

TO DO AT END--- Final findings:
visualize results, explain whats best (pros and cons each), and why that might be, due to our specific data and application
    - Meta comparison at end(? to make a even more robust classifier? Or will this severely overfit)
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import tree
from sklearn import cluster
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression

"""
Obtain Data
"""
df = pd.read_csv('data.csv')

y = df['diagnosis'].replace('B', 0).replace('M', 1)
X = df.drop(columns='diagnosis').fillna(0)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=888)

"""
Decision Tree
"""
clf = DecisionTreeClassifier(max_depth=3)            # 3 was manually selected (lower=underfitting, higher=overfitting)
clf.fit(X_train, y_train)

print('accuracy is {}'.format(accuracy_score(y_test, clf.predict(X_test))))

# .txt file produced can be entered on WebGraphViz website to get diagram, or cmd lines can also be used.
with open("tree.txt", "w") as f:
    f = tree.export_graphviz(clf, out_file=f)

"""
Random Forest
"""
clf = RandomForestClassifier(n_estimators=100)
clf.fit(X_train, y_train)
print('accuracy is {}'.format(accuracy_score(y_test, clf.predict(X_test))))

"""
LDA
"""
clf = LinearDiscriminantAnalysis()
clf.fit(X_train, y_train)
print('accuracy is {}'.format(accuracy_score(y_test, clf.predict(X_test))))

"""
Logistic Regression
"""
clf = LogisticRegression()
clf.fit(X_train, y_train)
print('accuracy is {}'.format(accuracy_score(y_test, clf.predict(X_test))))

# Note to self: Later have a list of classifiers and iterate through them, saving a list of resulting accuracies.
# Do this many times to really compare them. Save a dataframe with each row being the series for one sample of data

"""
K-Nearest Neighbours
"""
clf = KNeighborsClassifier(n_neighbors=2)
# ^^ Try with different types of distance! Manhattan (i.e. taxi-cab) and Euclidean. add std
clf.fit(X_train, y_train)
print('accuracy is {}'.format(accuracy_score(y_test, clf.predict(X_test))))

"""
SVM
"""
# to do next

"""
K Means (Unsupervised - in here for fun)
"""
clf = KMeans(n_clusters=2)
clf.fit(X_train, y_train) # y_train not used, because this is unsupervised.
clf.predict(X_test)