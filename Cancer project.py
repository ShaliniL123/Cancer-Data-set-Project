# -*- coding: utf-8 -*-
"""
Created on Fri Oct 18 15:56:07 2019

@author: shalini lingampally
"""

# Lasso model
#'concave points_mean', 'radius_se', 'radius_worst', 'texture_worst','smoothness_worst', 'compactness_worst', 'concave points_worst','symmetry_worst'
# 8 variables

import pandas as pd
import numpy as np

wisc=pd.read_csv("C:/Users/shalini lingampally/Desktop/R program/Material/All Practice xlx sheets/(10) Regularization/wisc_bc_data-KNN.csv")
wisc.columns

#separate X and Y variables
X=wisc.drop(wisc.columns[[0,1,2,3,4,5,6,7,8,10,11,13,14,15,16,17,18,19,20,21,24,25,28,31]],axis=1)   # selecting all independent variables
X.columns                               # to see all columns names

wisc.as_matrix()         # as it is a discrete
Da = wisc.diagnosis.map(dict(M=1,B=0))    # it will assign M=1 and B=0

Y=Da     # selecting only target variable

#write all packages
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection._split import train_test_split
from sklearn import metrics
from sklearn.metrics import precision_score,recall_score

# import warnings filter , ignore all future warnings
from warnings import simplefilter
simplefilter(action='ignore', category=FutureWarning)


#splitting into training and testing datasets
random=range(1,1000)       # resampling

training_accuracy = []     # create an array
testing_accuracy = []      # create an array

for rand_state in random:
    X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.30,random_state=rand_state)  # testing data is 30 % and resampling takes 1000 times
    logreg=LogisticRegression()
    logreg.fit(X_train,Y_train)                              # to fit a model
    
    Y_train_pred=logreg.predict(X_train)
    training_accuracy.append(logreg.score(X_train,Y_train))  # to know how much % is accurate
    
    Y_test_pred=logreg.predict(X_test)
    testing_accuracy.append(logreg.score(X_test,Y_test))   # to know how much % is accurate and to know where it is near to traindataset
    
#confuson matrix
cm=metrics.confusion_matrix(Y_test,Y_test_pred)                        # how many are 0s and 1s
print(cm)
print(np.mean(Y_test_pred==Y_test).round(3))                            #it is same as accuray
print("Accuracy:",metrics.accuracy_score(Y_test,Y_test_pred).round(3))  # 91.2% to know what is accuracy of a model
print("Percision:",precision_score(Y_test,Y_test_pred).round(3))        # 98.3%
print("Recall:",recall_score(Y_test,Y_test_pred).round(3))              # 80.8%

# To know accuracy
train = pd.DataFrame(training_accuracy )
np.average(train).round(3)                    # how much % is accurate

test = pd.DataFrame(testing_accuracy )
np.average(test).round(3)                     # how much % is accurate

# plotting histogram
test_values=pd.DataFrame(testing_accuracy)    # create a data frame of test accuracy
test_values.plot.hist()                       # give a graph is symmetric or not negative skew
test_values.describe()                        # gives mean,std......

#normal distrubution is to know the range
minvalue = 0.938957 - (3*0.018705)     # to know minvalue
maxvalue = 0.938957 + (3*0.018705)    # to know maxvalue
print(minvalue,maxvalue)               # we can say the accuracy lies from 88.2% to 99.5%

##############################################################################################

# Model Ridge
#16 variables
#['concave points_mean', 'symmetry_mean', 'fractal_dimension_mean','radius_se', 'smoothness_se', 'compactness_se', 'concavity_se','concave points_se', 'symmetry_se', 'fractal_dimension_se',
       #'radius_worst', 'texture_worst', 'smoothness_worst',
       #'compactness_worst', 'concave points_worst', 'symmetry_worst'],
      
import pandas as pd
import numpy as np

wisc=pd.read_csv("C:/Users/shalini lingampally/Desktop/R program/Material/All Practice xlx sheets/(10) Regularization/wisc_bc_data-KNN.csv")
wisc.columns

#separate X and Y variables
X1=wisc.drop(wisc.columns[[0,1,2,3,4,5,6,8,12,13,14,15,23,24,25,28]],axis=1)   # selecting all independent variables
X1.columns                               # to see all columns names

wisc.as_matrix()         # as it is a discrete
Da = wisc.diagnosis.map(dict(M=1,B=0))    # it will assign M=1 and B=0

Y1=Da     # selecting only target variable

#write all packages
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection._split import train_test_split
from sklearn import metrics
from sklearn.metrics import precision_score,recall_score

# import warnings filter , ignore all future warnings
from warnings import simplefilter
simplefilter(action='ignore', category=FutureWarning)


#splitting into training and testing datasets
random=range(1,1000)       # resampling

training_accuracy = []     # create an array
testing_accuracy = []      # create an array

for rand_state in random:
    X1_train,X1_test,Y1_train,Y1_test=train_test_split(X1,Y1,test_size=0.30,random_state=rand_state)  # testing data is 30 % and resampling takes 1000 times
    logreg=LogisticRegression()
    logreg.fit(X1_train,Y1_train)                              # to fit a model
    
    Y1_train_pred=logreg.predict(X1_train)
    training_accuracy.append(logreg.score(X1_train,Y1_train))  # to know how much % is accurate
    
    Y2_test_pred=logreg.predict(X1_test)
    testing_accuracy.append(logreg.score(X1_test,Y1_test))   # to know how much % is accurate and to know where it is near to traindataset
    
#confuson matrix
cm=metrics.confusion_matrix(Y1_test,Y2_test_pred)                        # how many are 0s and 1s
print(cm)
print(np.mean(Y2_test_pred==Y1_test).round(3))                            #it is same as accuray
print("Accuracy:",metrics.accuracy_score(Y1_test,Y2_test_pred).round(3))  # 91.8% to know what is accuracy of a model
print("Percision:",precision_score(Y1_test,Y2_test_pred).round(3))        # 98.4
print("Recall:",recall_score(Y1_test,Y2_test_pred).round(3))              # 82.2

# To know accuracy
train1 = pd.DataFrame(training_accuracy )
np.average(train1).round(3)                    # how much % is accurate

test1 = pd.DataFrame(testing_accuracy )
np.average(test1).round(3)                     # how much % is accurate

# plotting histogram
test_values=pd.DataFrame(testing_accuracy)    # create a data frame of test accuracy
test_values.plot.hist()                       # give a graph is symmetric or not negative skew
test_values.describe()                        # gives mean,std......

#normal distrubution is to know the range
minvalue = 0.939437 - (3*0.018980)     # to know minvalue
maxvalue = 0.939437 + (3*0.018980)    # to know maxvalue
print(minvalue,maxvalue)               # we can say the accuracy lies from 88.2% to 99.6%
#######################################################################################################

# Model Elastic
# 8 variables
# ['radius_mean', 'concave points_mean', 'radius_se', 'radius_worst',
       #'texture_worst', 'smoothness_worst', 'compactness_worst',
       #'concave points_worst', 'symmetry_worst']
import pandas as pd
import numpy as np

wisc=pd.read_csv("C:/Users/shalini lingampally/Desktop/R program/Material/All Practice xlx sheets/(10) Regularization/wisc_bc_data-KNN.csv")
wisc.columns

#separate X and Y variables
X2=wisc.drop(wisc.columns[[0,1,3,4,5,6,7,8,10,11,13,14,15,16,17,18,19,20,21,24,25,28,31]],axis=1)   # selecting all independent variables
X2.columns                               # to see all columns names

wisc.as_matrix()         # as it is a discrete
Da = wisc.diagnosis.map(dict(M=1,B=0))    # it will assign M=1 and B=0

Y2=Da     # selecting only target variable

#write all packages
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection._split import train_test_split
from sklearn import metrics
from sklearn.metrics import precision_score,recall_score

# import warnings filter , ignore all future warnings
from warnings import simplefilter
simplefilter(action='ignore', category=FutureWarning)


#splitting into training and testing datasets
random=range(1,1000)       # resampling

training_accuracy = []     # create an array
testing_accuracy = []      # create an array

for rand_state in random:
    X2_train,X2_test,Y2_train,Y2_test=train_test_split(X2,Y2,test_size=0.30,random_state=rand_state)  # testing data is 30 % and resampling takes 1000 times
    logreg=LogisticRegression()
    logreg.fit(X2_train,Y2_train)                              # to fit a model
    
    Y3_train_pred=logreg.predict(X2_train)
    training_accuracy.append(logreg.score(X2_train,Y2_train))  # to know how much % is accurate
    
    Y4_test_pred=logreg.predict(X2_test)
    testing_accuracy.append(logreg.score(X2_test,Y2_test))   # to know how much % is accurate and to know where it is near to traindataset
    
#confuson matrix
cm=metrics.confusion_matrix(Y2_test,Y4_test_pred)                        # how many are 0s and 1s
print(cm)
print(np.mean(Y2_test_pred==Y2_test).round(3))                            #it is same as accuray
print("Accuracy:",metrics.accuracy_score(Y2_test,Y4_test_pred).round(3))  # 90.6% to know what is accuracy of a model
print("Percision:",precision_score(Y2_test,Y4_test_pred).round(3))        # 96.7
print("Recall:",recall_score(Y2_test,Y4_test_pred).round(3))              # 80.8

# To know accuracy
train3 = pd.DataFrame(training_accuracy )
np.average(train3).round(3)                    # how much % is accurate

test4 = pd.DataFrame(testing_accuracy )
np.average(test4).round(3)                     # how much % is accurate

# plotting histogram
test_values=pd.DataFrame(testing_accuracy)    # create a data frame of test accuracy
test_values.plot.hist()                       # give a graph is symmetric or little negative skew
test_values.describe()                        # gives mean,std......

#normal distrubution is to know the range
minvalue = 0.938957 - (3*0.018705)     # to know minvalue
maxvalue = 0.938957 + (3*0.018705)    # to know maxvalue
print(minvalue,maxvalue)              # 88.2% to 99.5% it may lie

############################################################################################################

# To know how M & B varies in graph
import matplotlib.pyplot as plt
import seaborn as sns

ax = sns.countplot(Y, label="Count")
B, M = Y.value_counts()
print('Number of benign cancer: ', B)
print('Number of malignant cancer: ', M)

#ROC Curve
y_pred_proba = logreg.predict_proba(X_test)[::,1]
fpr, tpr, _ = metrics.roc_curve(Y_test,  y_pred_proba)
auc = metrics.roc_auc_score(Y_test, y_pred_proba)
plt.plot(fpr,tpr,label="data 1, auc="+str(auc))
plt.legend(loc=4)
plt.show()

# Show confusion matrix in a separate window
import matplotlib.pyplot as plt
plt.matshow(cm)
plt.title('Confusion matrix')
plt.colorbar()
plt.xlabel('Actual label')
plt.ylabel('Predicted label')
plt.show()

############################################################################################################