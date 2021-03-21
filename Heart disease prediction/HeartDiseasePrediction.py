#Heart Disease Prediction with machiene learning
#Importing Libraries
from numpy import genfromtxt
import numpy as np
from numpy import *
import matplotlib
import matplotlib.pyplot as plt


from sklearn.svm import LinearSVC
from sklearn.decomposition import PCA
import pylab as pl
from itertools import cycle
from sklearn import cross_validation
from sklearn.svm import SVC
#Loading data
dataset = genfromtxt('dataset.csv',delimiter=',')

#Printing the datasetd
X = dataset[:,0:12] #Feature set
Y = dataset[:,13]    #label Set

# Item with 0 value is already indexed as 0 , so rest are indexed as 1
for index, item in enumerate(Y):   # Last row gives 4 diff types of output , so convert them to 0  or 1 
	if not (item == 0.0):       #either Yes or No
		Y[index] = 1
print(Y)
target_names = ['0','1']

#Method to plot the graph for reduced Dimensions
def plot_2D(data,target,target_names):
	colors = cycle('rgbcmykw')
	target_ids = range(len(target_names))
	plt.figure()
	for i,c, label in zip(target_ids, colors, target_names):
		plt.scatter(data[target == i, 0], data[target == i, 1], c=c, label=label)
	plt.legend()
    
# Classifying the data using a linear SVM and perdicting the probabilities

modelSVM = LinearSVC(C=0.1)
pca = PCA(n_components=2, whiten=True).fit(X)   # n denotes number of components to keep after Dimensionality Reduction
X_new = pca.transform(X)

# Applying cross validation on the training and test set to validate our linear SVM model
X_train,X_test,Y_train,Y_test = cross_validation.train_test_split(X_new, Y, test_size = 0.2, train_size=0.8, random_state=0)
modelSVM = modelSVM.fit(X_train,Y_train)
print("Linear SVC values with Split")
print(modelSVM.score(X_test, Y_test))

#Calling the above defined function plot_2D
plot_2D(X_new, Y, target_names)

