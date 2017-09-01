import pandas as pd
import pylab as plt
import numpy as np
from sklearn import linear_model
from sklearn.utils import shuffle
from sklearn.preprocessing import StandardScaler

# Read in data
data = pd.read_csv('crime.csv', index_col=0)
print (data.head())

# extract feature matrix and target vector
# Feature Matrix
feature_cols = ['Education','Police','Income','Inequality']
# Target Vector
target = ['Crime']
# Loading into array
X = np.array(data[feature_cols])
Y = np.array(data[target])
# Give em a shuffle
X, Y = shuffle(X, Y, random_state=1)

# 1. Plot Education vs Crime
plt.scatter(X[:,0],Y)
plt.xlabel('Education')
plt.ylabel('Crime');
# Display Plot
plt.show()

# 2. Plot Police vs Crime
plt.scatter(X[:,1],Y)
plt.xlabel('Police')
plt.ylabel('Crime');
# Display Plot
plt.show()

# 3. Plot Income vs Crime
plt.scatter(X[:,2],Y)
plt.xlabel('Income')
plt.ylabel('Crime');
# Display Plot
plt.show()

# 4. Plot Inequality vs Crime
plt.scatter(X[:,3],Y)
plt.xlabel('Inequality')
plt.ylabel('Crime');
# Display Plot
plt.show()

# 5. Is the education variable positively or negatively correlated with crime?
# 5. Answer: Negatively
    
# 6. Is the police variable positively or negatively correlated with crime?
# 6. Answer: Positively

# 7: Split the data in 2 halves: training set and test set
# set training size to half
train_set_size = int(X.shape[0]/2)
# X
X_train = X[:train_set_size, :] # select first half for train
X_test = X[train_set_size:, :] # select last half for test
print(X_train.shape)
print(X_test.shape)
# Y
Y_train = Y[:train_set_size] # select first half for train
Y_test = Y[train_set_size:] # select last half for test
print(Y_train.shape)
print(Y_test.shape)

# 8. Fit a multivariate linear regression model on the training data using all the features available
# Import linear regression model
model = linear_model.LinearRegression()

#fitting to model
model.fit(X_train, Y_train)

# 9: What are the intercept ( θ0θ0 ) and coefficients ( θ1θ1 ,  θ2θ2 ,  θ3θ3  and  θ4θ4 ) of the model?
# 9: Answer: The intercept coefficient is [-1455.49210253], The other coefficients are 0:-72.25192429, 1: 43.58467713, 2:0.28595797, 3: 61.61793052  
#print coefficients
print(model.coef_) #theta1,2,3 and 4
print(model.intercept_) #theta 0

# 10: What is the  R2R2  score (i.e. the coefficient of determination that measures the proportion of the outcomes variation explained by the model) for the training data? and for the test data?
# 10: The Coefficient of Determination is 0.664843765409
# Print R2 Score
print(model.score(X_test, Y_test))

# 11: Given the following imaginary cities with the provided values for the predictors education, police, income and inequality, which city should have the highest level of crime according to your model?:
# 11: Answer: City 2

# 12: Re-instantiate your linear regression model with the parameter fit_intercept set to False and rerun your analysis on the entire feature matrix  XX . When we set the fit_intercept to False we are basically fitting a model with no intercept parameter  θ0θ0 . Output the coefficients you get for  θ1...θ4θ1...θ4 .

# Set fit intercept to false
model = linear_model.LinearRegression(fit_intercept=False)
# Refit model
model.fit(X_train, Y_train);

#Reprint Coefficients
print(model.coef_)

# 13: Calculate the coefficients for  θ1...θ4θ1...θ4  using the analytical/close form solution of linear regression. Make sure those estimates coincide with what you get in Exercise 12 to be certain you got it right. Use the matrix algebra functionality provided by the numpy library to find the optimal vector  θθ . Provide the line of code you created to calculate the solution.

# 14: Read the data from the file into the appropriate  XX  and  yy  data structures and shuffle it.
# read in file
data = pd.read_csv('churn.csv', index_col=0)

#Feature Matrix for Churn
feature_cols = ['Account Length','Int\'l Plan','VMail Plan','VMail Message','Day Mins','Day Calls','Day Charge','Eve Mins','Eve Calls','Eve Charge','Night Mins','Night Calls','Night Charge','Intl Mins','Intl Calls','Intl Charge','CustServ Calls']
# Target Vector
target = ['label']
# Loading into array
X = np.array(data[feature_cols])
Y = np.array(data[target])
# Give em a shuffle too
X, Y = shuffle(X, Y, random_state=1)

# 15: Split the data into a training set and test set (test set size should be 33%)
train_set_size = int(X.shape[0]/3)
# X
X_train = X[:(train_set_size*2), :] # select first two-thirds for train
X_test = X[train_set_size:, :] # select last third for test
print(X_train.shape)
print(X_test.shape)
# Y
Y_train = Y[:(train_set_size*2)] # select first two-thirds for train
Y_test = Y[train_set_size:] # select last third for test
print(Y_train.shape)
print(Y_test.shape)

# 16: Scale the data using the StandardScaler class from scikit-learn
scaler = StandardScaler().fit(X)
print(scaler.mean_)
print(scaler.scale_)
X_test_scaled = scaler.transform(X_test)
print(X_test_scaled) 

# 17: Train a logistic regression model and estimate its performance on the test data
from sklearn.linear_model import LogisticRegression
logRegModel = LogisticRegression()
logRegModel.fit(X_train, Y_train)
print("Accuracy of logistic regression classifier on test set:", logRegModel.score(X_test, Y_test))
# 86% accurate

# 18. Train a K nearest neighbors classifier and estimate its performance on the test data
from sklearn.neighbors import KNeighborsClassifier
# set number of neighours to 5
knn = KNeighborsClassifier(n_neighbors=5) 
# fit model to data
knn.fit(X_train, Y_train)
print("Accuracy of KNN classifier on test set:", knn.score(X_test, Y_test))