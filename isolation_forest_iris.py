# Isolation forest
# Importing libraries
import numpy as np
import pandas as pd
from numpy import genfromtxt
from sklearn.ensemble import IsolationForest

# Create numpy ndarray from pandas dataframe
X = genfromtxt('Iris.csv',delimiter=',',skip_header=1,usecols=(1,2,3,4))

# Create the iforest object
iforest = IsolationForest(n_estimators=100, max_samples='auto',
                          contamination=0.04, max_features=1.0,
                          bootstrap=False, n_jobs=-1, random_state=1)

# Apply the iforest object on the numpy ndarray X to create pred
# pred is a numpy ndarray that returns 1 for inliers, -1 for outliers
pred = iforest.fit_predict(X)

# Extract outliers
outlier_index = np.where(pred==-1)
outlier_values = X[outlier_index]
print("Outliers in training data")
print(outlier_index)

# Real time use
Xrt = genfromtxt('Iris-1.csv',delimiter=',',skip_header=1,usecols=(1,2,3,4))
yrt=iforest.fit_predict(Xrt)
print(yrt)