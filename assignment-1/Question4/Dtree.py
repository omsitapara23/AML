import numpy as np  
import csv
import json
from sklearn.neighbors import KNeighborsClassifier
from sklearn import preprocessing
from collections import Counter
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import BernoulliNB
from sklearn import tree


def makeTrain(data_train):
    totAttr = set([])
    count=0
    labels = []
    for item in data_train:
        labels.append(item.get("cuisine"))
        ingredits = item.get("ingredients")
        for i in ingredits:
            totAttr.add(i)
        count += 1

    featureVec = []
    for i in totAttr:
        featureVec.append(i)

    data = np.zeros((count, len(totAttr)))
    count =0
    for item in data_train:
        ingredits = item.get("ingredients")
        for i in ingredits:
            if i in featureVec:
                ind = featureVec.index(i)
                data[count,ind] = 1
        count +=1
    
    return data, len(totAttr), labels, featureVec

def makeTest(data_test, totAttr, featureVec):

    no = 0
    for item in data_test:
        no += 1

    ids = []
    data = np.zeros((no, totAttr))
    count = 0 
    for item in data_test:
        ids.append(item.get("id"))
        ingredits = item.get("ingredients")
        for i in ingredits:
            if i in featureVec:
                ind = featureVec.index(i)
                data[count,ind] = 1
        count += 1
    return data, ids

def preprocessing_data(data_train, data_test):
    return preprocessing.scale(data_train), preprocessing.scale(data_test)


def learn(data_train, labels):
    model = tree.DecisionTreeClassifier()
    model.fit(data_train, labels)
    return model

def test(data_test, model):
    output = model.predict(data_test)
    return output

def write_csv(output, ids):
    text_file = open("Output.csv", "w")
    text_file.write("id,cuisine\n")

    counter = 0
    for instance in output:
        text_file.write("%d,%s\n" % (ids[counter] , instance))
        counter += 1

    text_file.close() 


if __name__ == "__main__":
    #opening the files
    with open('train.json') as f:
        data_train = json.load(f)

    with open('test.json') as f1:
        data_test = json.load(f1)

    data_train, totAttr, labels, featureVec = makeTrain(data_train)
    print "Train loaded"
    data_test, ids = makeTest(data_test, totAttr, featureVec)
    print "Test loaded"
    print "Preprocessing..."
    data_train, data_test = preprocessing_data(data_train, data_test)
    print "Preprocessing complete"
    print "Learning..."
    model = learn(data_train, labels)
    print "Model learned"
    print "Predicting..."
    output = test(data_test, model)
    print "Predection complete writing to file..."
    write_csv(output, ids)
    print "Writing success"
