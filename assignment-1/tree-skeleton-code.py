# CS6510 HW 1 Code Skeleton
# Please use this outline to implement your decision tree. You can add any code around this.

import csv
import numpy

# Enter You Name Here
myname = "Om-Sitapara-" # or "Doe-Jane-"

# Implement your decision tree below
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

    def learn(self, training_set):
        # implement this function
        self.tree = {}


    # implement this function
    def classify(self, test_instance):
        result = 0 # baseline: always classifies as 0
        return result

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

def calcEntropy(i,j):
    if i+j == 0:
        return 0
    p1 = float(i)/(i+j)
    p2 = float(j)/(i+j)
    res = 0
    if p1!= 0 and p2 != 0:
        res = -p1*numpy.log2(p1) - p2*numpy.log2(p2)
    return res

def entropy_attr(attr, values, rows):
    # print attr.shape
    gains=[]
    gains_v = []
    E1 = []
    E2 = []          
    split_arr = attr.copy()
    split_arr.sort()
    # print "split", split_arr , split_arr.shape
    print len(set(values))
    for j in range (rows-1):
        to_split = (split_arr[j] + split_arr[j+1])/float(2)
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
    # print count1p ,  " : " , count1n
    # print count2p , " : " , count2n
        summ = count1n + count1p + count2n + count2p
        p1 = float(count1n + count1p)/summ
        p2 = float(count2n + count2p)/summ
        E1.append(calcEntropy(count1n, count1p))
        E2.append(calcEntropy(count2n, count2p))
        enttr = p1*calcEntropy(count1n, count1p) + p2*calcEntropy(count2n, count2p)
        gains.append(enttr)
        gains_v.append(to_split)
    return min(gains), gains_v[gains.index(min(gains))], E1[gains.index(min(gains))], E2[gains.index(min(gains))]
    

def firstSplit(data):
    gain, attr, split_value, lettr, rettr = findAttrWithGain(data)
    root = treeNode(attr, split_value, None, False, lettr, rettr, True, 0)
    return root

def splitter(data, node):
    datal = []
    datar = []
    print node.attrid, " : ", node.svalue
    deleteCol = data[:,node.attrid].copy()
    # print deleteCol, deleteCol.shape
    new_data = data.copy()
    new_data = numpy.delete(new_data, node.attrid, 1)
    # print new_data , new_data.shape
    count = 0
    for instance in new_data:
        if deleteCol[count] <= node.svalue:
            datal.append(instance)
        else:
            datar.append(instance)
        count = count + 1
    
    datal = numpy.array(datal)
    datar = numpy.array(datar)
    print "Node ", data.shape , " is splitted in ", datal.shape, " : ", datar.shape

    return datal, datar

def preorder(data, node, depth):
    curr = node
    # if depth == 11 and curr.leaf == False:
    #     curr.leaf = True
    #     countp = 0
    #     countn = 0
    #     for instance in data:
    #         if data[-1] == 1:
    #             countp = countp + 1
    #         else:
    #             countn = countn + 1
    #     if countn >= countp:
    #         curr.prediction = 0
    #     else:
    #         curr.prediction = 1
    #     return
    if curr.leaf == True:
        return
    datal, datar = splitter(data, node)

    # if len(datal) == len(data):
    #     curr.leaf = True
    #     curr.prediction = data[1,-1]
    # elif len(datar) == len(data):
    #     curr.leaf = True
    #     curr.prediction = data[1,-1]
    # else:
    if curr.entropyleft == 0 or depth == 10:
        new_nodel = treeNode(0,0,curr,True,0,0,False, depth + 1 )
        if len(set(datal[:,-1])) == 1:
            new_nodel.prediction = datal[0,-1]
        else:
            countp = 0
            countn = 0
            for instance in datal:
                if datal[-1] == 1:
                    countp = countp + 1
                else:
                    countn = countn + 1
            if countn >= countp:
                new_nodel.prediction = 0
            else:
                new_nodel.prediction = 1
        print "A leaf node for ", datal.shape, " prediction : ", new_nodel.prediction
        curr.left = new_nodel
    else:
        gain, attr, split_value, lettr, rettr = findAttrWithGain(datal)
        new_nodel = treeNode(attr, split_value, curr, False, lettr, rettr, False, depth + 1)
        curr.left = new_nodel
    if curr.entropyright == 0 or depth == 10:
        new_noder = treeNode(0,0,curr,True,0,0,False, depth + 1 )
        if len(set(datar[:,-1])) == 1:
            new_noder.prediction = datar[0,-1]
        else:
            countp = 0
            countn = 0
            for instance in datar:
                if datar[-1] == 1:
                    countp = countp + 1
                else:
                    countn = countn + 1
            if countn >= countp:
                new_noder.prediction = 0
            else:
                new_noder.prediction = 1
        print "a leaf node generation for ",  datar.shape, " prediction : ", new_noder.prediction 
        curr.right = new_noder
    else:
        gain, attr, split_value, lettr, rettr = findAttrWithGain(datar)
        new_noder = treeNode(attr, split_value, curr, False, lettr, rettr, False, depth + 1)
        curr.right = new_noder
    preorder(datal, curr.left, depth+1)
    preorder(datar, curr.right, depth+1)





def generate_tree(data):
    root = firstSplit(data)
    curr = root
    datal, datar = splitter(data,root)
    print "Datal :", datal.shape, datal
    print "Dalar :", datar.shape, datar
    preorder(data, curr, 0)

    return root


def print_inorder(node):
    print node.attrid, " : ", node.svalue, " : ", node.leaf
    if node.leaf == False:
        print_inorder(node.left)
        print_inorder(node.right)








def findAttrWithGain(data):
    diff_gains = []
    split_v = []
    E1 = []
    E2 = []
    rows, column = data.shape
    print "computing " , data.shape
    set_ettr = entropy_set(data)
    for i in range(column - 1):
        a,b,c,d = entropy_attr(data[:,i], data[:,column-1], rows)
        diff_gains.append(set_ettr - a)
        split_v.append(b)
        E1.append(c)
        E2.append(d)
    # print(diff_gains)
    max_gain = max(diff_gains)
    max_gain_index = diff_gains.index(max_gain)
    # print data
    # print "E1 : " , E1
    # print "E2 : " , E2
    print "Splitter : " , max_gain_index, " : ", split_v[max_gain_index]
    print "E1 : " , E1[max_gain_index], " : ", E2[max_gain_index]
    return max_gain, max_gain_index, split_v[max_gain_index], E1[max_gain_index], E2[max_gain_index]

def classification(data, node):
    if node.leaf == True:
        return node.prediction
    newdata = data.copy()
    newdata = numpy.delete(newdata, node.attrid)
    if data[node.attrid] <= node.svalue:
        return classification(newdata, node.left)
    else:
        return classification(newdata, node.right)
        



def run_decision_tree():

    #Load data set
    with open("wine-dataset.csv") as f:
        next(f, None)
        data = [tuple(line) for line in csv.reader(f, delimiter=",")]
    print "Number of records: %d" % len(data)
    data = numpy.array(list(data)).astype("float")
    # Split training/test sets
    # You need to modify the following code for cross validation.
    K = 10
    faccuracy = 0.0
    for z in range(10):

        training_set = [x for i, x in enumerate(data) if i % K != z]
        test_set = [x for i, x in enumerate(data) if i % K == z]
        training_set = numpy.array(training_set)
        test_set = numpy.array(test_set)
        print training_set.shape
        print(training_set)
        print test_set.shape
        print(test_set)
        root = generate_tree(training_set)
        print_inorder(root)


        # print root.attrid, root.svalue
        # print root.left.attrid, root.left.svalue
        # print root.right.attrid, root.right.svalue
        # tree = DecisionTree()
        # # Construct a tree using training set
        # tree.learn( training_set )

        # Classify the test set using the tree we just constructed
        results = []
        for instance in test_set:
            result = classification( instance[:-1], root )
            print result, " : " , instance[-1] , result == instance[-1]
            results.append( result == instance[-1])

        # Accuracy
        accuracy = float(results.count(True))/float(len(results))
        print "accuracy: %.4f" % accuracy 
        faccuracy = faccuracy + accuracy
    
    print "Final accu : ", faccuracy/10
    # # Writing results to a file (DO NOT CHANGE)
    # f = open(myname+"result.txt", "w")
    # f.write("accuracy: %.4f" % accuracy)
    # f.close()


if __name__ == "__main__":
    run_decision_tree()