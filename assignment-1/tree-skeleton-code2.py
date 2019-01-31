# CS6510 HW 1 Code Skeleton
# Please use this outline to implement your decision tree. You can add any code around this.

import csv
import pandas as pd
from sklearn.tree import DecisionTreeClassifier 
from sklearn import tree as tt
from sklearn.model_selection import train_test_split
import pydot
import sklearn

# Enter You Name Here
myname = "Om-Sitapara" # or "Doe-Jane-"

# Implement your decision tree below
class DecisionTree():
    tree = {}

    def learn(self, training_set, training_value):
        # implement this function
        self.tree = DecisionTreeClassifier(criterion = "entropy")
        self.tree = self.tree.fit(training_set, training_value)

    # implement this function
    def classify(self, test_instance):
        result = 0 # baseline: always classifies as 0
        result = self.tree.predict(test_instance)
        return result

def run_decision_tree():

    # Load data set
    # with open("wine-dataset.csv") as f:
    #     next(f, None)
    #     data = [tuple(line) for line in csv.reader(f, delimiter=",")]
    data = pd.read_csv("wine-dataset.csv", sep = ',')


    # Split training/test sets
    # You need to modify the following code for cross validation.
    # K = 10
    # training_set = [x for i, x in enumerate(data) if i % K != 9]
    # test_set = [x for i, x in enumerate(data) if i % K == 9]
    X = data.values[:,1:11]
    Y = data.values[:,11]
    print(X)
    print(Y)
    X_train, X_test, y_train, y_test = train_test_split( X, Y, test_size = 0.3, random_state = 100)

    print(X_test)
    tree = DecisionTree()
    # Construct a tree using training set
    tree.learn( X_train, y_train )

    # Classify the test set using the tree we just constructed
    results = []
    count = 0
    # for instance in X_test:
    #     result = tree.classify( instance )
    #     results.append( result == y_test[count])
    #     count = count + 1
    result = tree.classify(X_test)
    print(result)
    count = 0
    for instance in y_test:
        results.append(result[count] == instance)
        count = count + 1


    # Accuracy
    accuracy = float(results.count(True))/float(len(results))
    print "accuracy: %.4f" % accuracy       
    

    # Writing results to a file (DO NOT CHANGE)
    f = open(myname+"result.txt", "w")
    f.write("accuracy: %.4f" % accuracy)
    f.close()


if __name__ == "__main__":
    run_decision_tree()
