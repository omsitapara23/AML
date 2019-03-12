import numpy as np 
import sys

from sklearn.svm import LinearSVC
from sklearn.svm import SVC


if len(sys.argv) != 2:
    print "Please provide function name"
else:

    train_data = np.genfromtxt("gisette_train_data.csv", dtype= 'int', delimiter= ' ')
    tain_labels = np.genfromtxt("gisette_train_labels.csv", dtype= 'int', delimiter= ' ')
    test_data = np.genfromtxt("gisette_valid_data.csv", dtype= 'int', delimiter= ' ')
    test_labels = np.genfromtxt("gisette_valid.csv", dtype= 'int', delimiter= ' ')


    no_of_vectors = 0
    prediction = []
    prediction_train = []
    if sys.argv[1] == "linear":
        clf = SVC(kernel='linear')
        clf.fit(train_data, tain_labels)
        no_of_vectors = clf.n_support_
        prediction = clf.predict(test_data)
        prediction_train = clf.predict(train_data)
    elif sys.argv[1] == "poly":
        clf = SVC(kernel='poly', degree=2, coef0 = 1)
        clf.fit(train_data, tain_labels)
        no_of_vectors = clf.n_support_
        prediction = clf.predict(test_data)
        prediction_train = clf.predict(train_data)
    elif sys.argv[1] == "gaussian":
        clf = SVC(kernel='rbf', gamma=0.001)
        clf.fit(train_data, tain_labels)
        no_of_vectors = clf.n_support_
        prediction = clf.predict(test_data)
        prediction_train = clf.predict(train_data)
    else:
        print "Please provide correct  function name"

    print prediction
    print "support vector: ", no_of_vectors
    result_array = []
    counter = 0
    for instance in test_labels:
        result_array.append(instance == prediction[counter])
        counter += 1

    print "Accuracy test: ", float(result_array.count(True))/float(len(result_array))


    result_array = []
    counter = 0
    for instance in tain_labels:
        result_array.append(instance == prediction_train[counter])
        counter += 1

    print "Accuracy train: ", float(result_array.count(True))/float(len(result_array))
