#A python program to recognise digit
#Uses Scikit-learn module to recognise digits
#Uses Support Vector Mechanism(SVM) alorithm
#Dataset is already given under Scikit-learm module (digits)
#Dataset contains an already structured and labeled set of samples that contains
#pixel information for numbers up to 9 that we can use for training and testing


#importing needed modules

import matplotlib.pyplot as plt     #to show graphically the digit predicted
from sklearn import datasets        #importing dataset
from sklearn import svm             #to apply SVM on dataset

#Loading Dataset
digits = datasets.load_digits()

#Specifying classifier
clf=svm.SVC(gamma=0.001, C=100)
X,y=digits.data[:-10], digits.target[:-10]

#Training model
clf.fit(X,y)

#Showing which digit has been recognised as per the given test dataset
plt.imshow(digits.images[-5], cmap=plt.cm.gray_r, interpolation='nearest')
plt.show()
