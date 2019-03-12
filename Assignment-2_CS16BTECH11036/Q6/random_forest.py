# CS6510 HW 1 Code Skeleton
# Please use this outline to implement your decision tree. You can add any code around this.

#Running guidline:
# python dtree.py gain -> using gain function
# python dtree.py gini -> using gini function
import csv
import numpy
import sys
import pandas
import math as math
import random

# Enter You Name Here
myname = "Om-Sitapara-" # or "Doe-Jane-"

# Implement your decision tree below

#class for making a tree node 
class treeNode:
    def __init__(self, idattr, to_split, parent, leafc, lettr, rettr, rootV, depth):
        self.left = None
        self.right = None
        self.svalue = to_split
        self.parent = parent
        self.leaf = leafc
        self.entropyleft = lettr
        self.entropyright = rettr
        self.attrid = idattr
        self.isroot = rootV
        self.depth = depth
        self.prediction = 0

class DecisionTree():
    tree = {}

    def learn(self, training_set, function = "gain"):
        # implement this function
        self.tree = generate_tree(training_set, function)


    # implement this function
    def classify(self, test_instance):
        result = classification(test_instance, self.tree)
        return result


#function to count the entropy of data set
def entropy_set(data):
    countn = 0
    countp = 0
    for instance in data:
        if instance[-1] == 0:
            countn = countn + 1
        else:
            countp = countp + 1
    summ = countn + countp
    p1 = countn/float(summ)
    p2 = countp/float(summ)
    return calcEntropy(countn, countp)


#helper function to caluclate entropy
def calcEntropy(i,j):
    if i+j == 0:
        return 0
    p1 = float(i)/(i+j)
    p2 = float(j)/(i+j)
    res = 0
    if p1!= 0 and p2 != 0:
        res = -p1*numpy.log2(p1) - p2*numpy.log2(p2)
    return res


#this function finds the best split value for and attr
#to obtain maximum gain
def entropy_attr(attr, values, rows):
    gains=[]
    gains_v = []
    E1 = []
    E2 = []   
    minv = min(attr)
    maxv = max(attr)    
    sprr = 15
    itr = (maxv - minv)/float(sprr)   
    for j in range(sprr):
        to_split = minv + j*itr
        count1p = 0
        count1n = 0
        count2p = 0
        count2n = 0
        summ = 0
        i = 0
        for instance in attr:
            if instance <= to_split:
                if values[i] == 1:
                    count1p = count1p + 1
                else:
                    count1n = count1n + 1
            else:
                if values[i] == 1:
                    count2p = count2p + 1
                else:
                    count2n = count2n + 1
            i = i + 1
        summ = count1n + count1p + count2n + count2p
        p1 = float(count1n + count1p)/summ
        p2 = float(count2n + count2p)/summ
        E1.append(calcEntropy(count1n, count1p))
        E2.append(calcEntropy(count2n, count2p))
        enttr = p1*calcEntropy(count1n, count1p) + p2*calcEntropy(count2n, count2p)
        gains.append(enttr)
        gains_v.append(to_split)
    return min(gains), gains_v[gains.index(min(gains))], E1[gains.index(min(gains))], E2[gains.index(min(gains))]
    

#function to select the root node split
def firstSplit(data, function):
    if function == "gini":
        gini, attr, split_value, lgini, rgini = findAttrWithGini(data)
        root = treeNode(attr, split_value, None, False, lgini, rgini, True, 0)
    else:
        gain, attr, split_value, lettr, rettr = findAttrWithGain(data)
        root = treeNode(attr, split_value, None, False, lettr, rettr, True, 0)
    return root


#splitting the data set into two parts on an attr and split value
def splitter(data, node):
    datal = []
    datar = []
    deleteCol = data[:,node.attrid].copy()
    new_data = data.copy()
    new_data = numpy.delete(new_data, node.attrid, 1)
    count = 0
    for instance in new_data:
        if deleteCol[count] <= node.svalue:
            datal.append(instance)
        else:
            datar.append(instance)
        count = count + 1
    
    datal = numpy.array(datal)
    datar = numpy.array(datar)

    return datal, datar

#recurssive function for building the tree
def preorder(data, node, depth, function):
    curr = node
    if curr.leaf == True:
        return
    datal, datar = splitter(data, node)
    
    #checking leaf condition
    if len(datal) == len(data):
        curr.leaf = True
        curr.prediction = datal[0,-1]
        return
    elif len(datar) == len(data):
        curr.leaf = True
        curr.prediction = datar[0,-1]
        return
    else:
        if curr.entropyleft == 0 or depth == 10:
            new_nodel = treeNode(0,0,curr,True,0,0,False, depth + 1 )
            if len(set(datal[:,-1])) == 1:
                new_nodel.prediction = datal[0,-1]
            else:
                countp = 0
                countn = 0
                for instance in datal:
                    if instance[-1] == 1:
                        countp = countp + 1
                    else:
                        countn = countn + 1
                if countn >= countp:
                    new_nodel.prediction = 0
                else:
                    new_nodel.prediction = 1
            curr.left = new_nodel
        else:
            if function == "gini":
                gini, attr, split_value, lgini, rgini = findAttrWithGini(datal)
                new_nodel = treeNode(attr, split_value, curr, False, lgini, rgini, False, depth + 1)
            else:
                gain, attr, split_value, lettr, rettr = findAttrWithGain(datal)
                new_nodel = treeNode(attr, split_value, curr, False, lettr, rettr, False, depth + 1)
            curr.left = new_nodel
        if curr.entropyright == 0 or depth == 10 :
            new_noder = treeNode(0,0,curr,True,0,0,False, depth + 1 )
            if len(set(datar[:,-1])) == 1:
                new_noder.prediction = datar[0,-1]
            else:
                countp = 0
                countn = 0
                for instance in datar:
                    if instance[-1] == 1:
                        countp = countp + 1
                    else:
                        countn = countn + 1
                if countn >= countp:
                    new_noder.prediction = 0
                else:
                    new_noder.prediction = 1
            curr.right = new_noder
        else:
            if function == "gini":
                gini, attr, split_value, lgini, rgini = findAttrWithGini(datar)
                new_noder = treeNode(attr, split_value, curr, False, lgini, rgini, False, depth + 1)
            else:
                gain, attr, split_value, lettr, rettr = findAttrWithGain(datar)
                new_noder = treeNode(attr, split_value, curr, False, lettr, rettr, False, depth + 1)
            curr.right = new_noder
        #recurssive calls:
        preorder(datal, curr.left, depth+1, function)
        preorder(datar, curr.right, depth+1, function)




#main function to generate the tree
def generate_tree(data, function):
    root = firstSplit(data, function)
    curr = root
    datal, datar = splitter(data,root)
    preorder(data, curr, 0, function)

    return root


#this function gives the attr on which to split to have max gain
def findAttrWithGain(data):
    diff_gains = []
    split_v = []
    E1 = []
    E2 = []
    rows, column = data.shape
    set_ettr = entropy_set(data)
    attributeToSplit = random.sample(range(0, column-1), min(column - 1, 2*int(math.sqrt(column))))
    for i in attributeToSplit:
        a,b,c,d = entropy_attr(data[:,i], data[:,column-1], rows)
        diff_gains.append(set_ettr - a)
        split_v.append(b)
        E1.append(c)
        E2.append(d)
    max_gain = max(diff_gains)
    max_gain_index = diff_gains.index(max_gain)
    attribute_index = attributeToSplit[max_gain_index]
    return max_gain, attribute_index, split_v[max_gain_index], E1[max_gain_index], E2[max_gain_index]


#function for classification of input
def classification(data, node):
    if node.leaf == True:
        return node.prediction
    newdata = data.copy()
    newdata = numpy.delete(newdata, node.attrid)
    if data[node.attrid] <= node.svalue:
        return classification(newdata, node.left)
    else:
        return classification(newdata, node.right)


#function returns the best split value for which 
#gini index of the attr is minimum
def attr_gini(attr, values, rows):
    gini=[]
    gini_v = []
    G1 = []
    G2 = []   
    minv = min(attr)
    maxv = max(attr)    
    sprr = 15
    itr = (maxv - minv)/float(sprr)
    for j in range(sprr):
        to_split = minv + j*itr
        count1p = 0
        count1n = 0
        count2p = 0
        count2n = 0
        summ = 0
        i = 0
        for instance in attr:
            if instance <= to_split:
                if values[i] == 1:
                    count1p = count1p + 1
                else:
                    count1n = count1n + 1
            else:
                if values[i] == 1:
                    count2p = count2p + 1
                else:
                    count2n = count2n + 1
            i = i + 1
        if count1n + count1p != 0:
            p1p = float(count1p)/(count1p + count1n)
            p1n = float(count1n)/(count1n + count1p)
        else:
            p1p = 0
            p1n = 0
        if count2n + count2p != 0:
            p2p = float(count2p)/(count2n + count2p)
            p2n = float(count2n)/(count2n + count2p)
        else:
            p2p = 0
            p2n = 0
        GG1 = 1 - p1p*p1p - p1n*p1n
        GG2 = 1 - p2p*p2p - p2n*p2n
        G1.append(GG1)
        G2.append(GG2)
        summ = count1n + count1p + count2n + count2p
        gini_f = ((count1n + count1p)/float(summ))*GG1 + ((count2n + count2p)/float(summ))*GG2
        gini.append(gini_f)
        gini_v.append(to_split)
    return min(gini), gini_v[gini.index(min(gini))], G1[gini.index(min(gini))], G2[gini.index(min(gini))]


# this function gives the best attr with lowest gini index
def findAttrWithGini(data):
    diff_ginis = []
    split_v = []
    G1 = []
    G2 = []
    rows, column = data.shape
    attributeToSplit = random.sample(range(0, column-1), min(column - 1, int(2*math.sqrt(column))))
    for i in attributeToSplit:
        a,b,c,d = attr_gini(data[:,i], data[:,column-1], rows)
        diff_ginis.append(a)
        split_v.append(b)
        G1.append(c)
        G2.append(d)
    min_gini = min(diff_ginis)
    min_gini_index = diff_ginis.index(min_gini)
    split_attr = attributeToSplit[min_gini_index]
    return min_gini, split_attr, split_v[min_gini_index], G1[min_gini_index], G2[min_gini_index]


        
# main function to run the decision tree
def run_decision_tree(function = "gain"):

    data = pandas.read_csv('spam.csv',delim_whitespace=True)
    print data.shape
    training_set = data.sample(frac=0.7,random_state=200)
    test_set = data.drop(training_set.index)
    test_set = test_set.values
    print test_set.shape[0]
    diff_result = numpy.zeros((test_set.shape[0], 50))
    oob_result = numpy.zeros((data.shape[0], 51))
    data = data.values
    for iteration in range(50):

        training_set_new = training_set.sample(frac = 1, replace = True)
        uniq = training_set_new.copy()
        uniq.drop_duplicates(inplace = True)
        oobset = training_set.drop(uniq.index)
        training_set_new = training_set_new.values
        oobset_index = oobset.index
        oobset = oobset.values
        tree = DecisionTree()
        tree.learn(training_set_new, function)
        counter = 0
        for instance in test_set:
            prediction = tree.classify(instance[:-1])
            diff_result[counter, iteration] = prediction
            counter = counter + 1

        counter = 0
        for instance in oobset:
            index = oobset_index[counter]
            prediction = tree.classify(instance[:-1])
            if prediction == 0 :
                oob_result[index, iteration] = -1
                oob_result[index, -1] = 2
            else:
                oob_result[index, iteration] = 1
                oob_result[index, -1] = 2

            counter += 1

        print "tree ", iteration, " made"
            
        tree = {}
    training_set = training_set.values
    counter = 0
    final_prediction = 0
    accuracy_array = []
    for instance in diff_result:
        prediction1 = numpy.sum(instance, dtype=numpy.int32)
        prediction0 = len(instance) - prediction1
        if prediction0 > prediction1:
            final_prediction = 0
        else:
            final_prediction = 1
        accuracy_array.append(final_prediction == test_set[counter,57])
        counter += 1
    
    counter = 0
    final_prediction = 0
    oobs_array = []
    for instance in oob_result:
        prediction0 = numpy.count_nonzero(instance == -1)
        prediction1 = numpy.count_nonzero(instance == 1)
        if prediction0 > prediction1:
            final_prediction = 0
        else:
            final_prediction = 1
        if instance[-1] == 2:
            oobs_array.append(final_prediction == data[counter, 57])
        else:
            oobs_array.append(1==1)
        counter += 1


    faccuracy = float(accuracy_array.count(True))/float(len(accuracy_array))
    print "Final accu : %.4f" % faccuracy

    faccuracy = float(oobs_array.count(True))/float(len(oobs_array))
    print "Final oob_acc : %.4f" % faccuracy


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print "wrong input run as : python dtree.py gain or python dtree.py gini"
    else:
        arg1 = sys.argv[1]
        run_decision_tree(arg1)