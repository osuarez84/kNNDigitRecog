# Author: Omar Suarez
# Date: 2016.09.02

# k NearestNeighbor from scratch. 
# This is an implementation of a kNN algorithm for digit recognition.
# Dimensionality reduction is used to reduce computational time. The final 
# number of features after reduction is choosen using cross-validation.

import pandas as pd
import numpy as np
from collections import Counter
from sklearn.decomposition import PCA
from sklearn.cross_validation import KFold


# load csv files to numpy arrays
def load_data():
    train_data = pd.read_csv('train.csv')
    test_data = pd.read_csv('test.csv')
    batch = test_data.shape[0]
    Xtrain = train_data.drop('label',axis=1).values[:batch,:]
    Ytrain = train_data['label'].values[:batch]
    Xtest = test_data.values[:batch,:]    
    return Xtrain, Ytrain, Xtest
    
    
def euclideanDistance(instance1, instance2, length):
    distance = 0
    for x in range(length):
        distance += pow((instance1[x] - instance2[x]),2)
    return np.sqrt(distance)
    
# Function to get the distances between the test sample and each of the training set samples    
def getDistances(trainingSet, testInstance):
    distances = np.empty(shape=(0,0))   # Initializing an empty np.array
    length = trainingSet.shape[1]-1
    for x in range(len(trainingSet)):
        dist = euclideanDistance(testInstance, trainingSet[x], length)
        distances = np.append(distances, dist)
    return distances    

# The function we need to call is predict        
def predict(Xtrain,Ytrain, Xtest, k=1):
    dists = getDistances(Xtrain, Xtest)
    num_test = dists.shape[0]
    y_pred = np.zeros(num_test)
    
    # Sorting the traning labels using the distances.
    labels = Ytrain[np.argsort(dists)]
    k_closest_y = labels[:k]

    c = Counter(k_closest_y)
    y_pred = c.most_common(1)[0][0]    
    return y_pred
        
# ---------------------------------------------------------- 
# MAIN
Xtrain, Ytrain, Xtest = load_data()



# Defining some variables
predictions = np.empty(shape=(0,0))
kf = KFold(len(Xtrain), n_folds=5, shuffle=True)
means = []

print("Cross-validating...")
best_PCA = []
best_Acc = 0
n_componets = ['100', '150', '200', '300']
# cross validation testing with 5 folds to select the PCA n_components best value
for n in n_componets:    
    for training, testing in kf:
        for i in range(len(Xtrain[testing])):
            pca = PCA(n_components=int(n), whiten=True)
            pca.fit(Xtrain[training])
            Xt = pca.transform            
            Xcv = Xtrain[testing,:]
            result = predict(Xt[training,:], Ytrain[training], Xcv[i,:])
            predictions = np.append(predictions, result)
        curmean = np.mean(predictions == Ytrain[testing])
        means.append(curmean)

    # print the mean for each n_component tested
    print("Mean accuracy for PCA = {} : {:.1%}".format(n, np.mean(means)))
    # Save the n_component that gives us the best accuracy
    finalAcc = np.means(means)
    if finalAcc > best_Acc:
        best_Acc = finalAcc
        best_PCA = n


# Once cross-validation gives us the best n_components we use the whole training
# set to do dimensionality reduction
print("Reducing dimensionality...")
pca = PCA(n_components=int(best_PCA), whiten=True)
pca.fit(Xtrain)
Xt = pca.transform(Xtrain)

# Predicting using the whole training set
print("Predicting...")
pred = predict(Xtrain, Ytrain, Xtest)


# -------------------------------------------------------------------
# CSV for submit
image_id = np.arange(1, pred.size+1)
submission = pd.DataFrame({"ImageId": image_id, "Label": pred})
submission.to_csv('digit_recog.csv',index=False)

 













   
    
    
    