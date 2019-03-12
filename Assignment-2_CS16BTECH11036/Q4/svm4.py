import numpy as np
from sklearn.svm import SVC
import sys

train_data = np.genfromtxt("4_train.csv", dtype='float', delimiter=' ')
train_data = np.delete(train_data, [1,2,4], 1)
train_data = train_data[[train_data[i][0] == 1 or train_data[i][0] == 5 for i in range(train_data.shape[0])],:]
train_labels = train_data[:,0]
train_labels[train_labels==5] = -1
train_data = train_data[:,1:]



test_data = np.genfromtxt("4_test.csv", dtype='float', delimiter=' ')
test_data = np.delete(test_data, [1,2,4], 1)
test_data= test_data[[test_data[i][0] == 1 or test_data[i][0] == 5 for i in range(test_data.shape[0])],:]
test_labels = test_data[:,0]
test_labels[test_labels == 5] = -1
test_data = test_data[:,1:]

C = 0.01
for c in range (5):
    print "C is ", C
    no_of_vectors = 0
    prediction = []
    prediction_train = []

    clf = SVC(kernel='rbf', C = C)
    clf.fit(train_data, train_labels)
    no_of_vectors = clf.n_support_
    prediction = clf.predict(test_data)
    prediction_train = clf.predict(train_data)

    print "support vector: ", no_of_vectors
    result_array = []
    counter = 0
    for instance in test_labels:
        result_array.append(instance == prediction[counter])
        counter += 1

    base_test = float(result_array.count(True))/float(len(result_array))

    print "Accuracy_test: ", float(result_array.count(True))/float(len(result_array))


    result_array = []
    counter = 0
    for instance in train_labels:
        result_array.append(instance == prediction_train[counter])
        counter += 1

    base_train = float(result_array.count(True))/float(len(result_array))
    print "Accuracy_train : ", float(result_array.count(True))/float(len(result_array))


    C = C*100
