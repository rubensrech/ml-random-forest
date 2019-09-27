from graphviz import Digraph
import pandas as pd
from pandas.api.types import is_numeric_dtype, is_string_dtype
from math import *
import numpy as np 
from tree import *

DEBUG = False

def info(x):
    return x * np.log2(1/x)

def infoGain(D, targetAttr, attr):
    # Calculate dataset (D) current entropy
    currEntropy = D.groupby(targetAttr).size().apply(lambda x: info(x/len(D))).agg('sum')

    # Partitionate dataset D using 'attr'
    attrPartitionsCount = D.groupby([attr, targetAttr])[targetAttr].count()
    # Create (auxiliary) table with resultant partitions Dj
    attrPartitions = D.groupby([attr]).agg({ targetAttr: 'count' })
    # Calculate |Dj|/|D| for each partition Dj
    attrPartitions['Prop'] = attrPartitionsCount.groupby(level=0).apply(lambda d: d.sum()/len(D))
    # Calculate entropy Info(Dj) for each parition Dj 
    attrPartitions['PartEntropy'] = attrPartitionsCount.groupby(level=0).agg(lambda x: np.sum(info(x/x.sum())))
    # Calculate |Dj|/|D| * Info(Dj) for each parition Dj
    attrPartitions['Entropy'] = attrPartitions['Prop'] * attrPartitions['PartEntropy']
    # Calculate entropy InfoA(D) of dataset after partitioning with attribute 'currAttr' 
    newEntropy = attrPartitions['Entropy'].agg('sum')

    # Calculate gain Gain(A)
    attrGain = currEntropy - newEntropy
    # Find attribute value that maximizes partition entropy 
    maxEntropyAttrValue = attrPartitions.idxmax()['PartEntropy']

    if DEBUG:
        print("> Resulting paritions for attribute: " + attr)
        print(attrPartitions)
        print("> Resulting gain: " + str(attrGain))
        # print("> Resulting entropy: " + str(newEntropy))
        # print("> Selected attribute value: " + maxEntropyAttrValue)
    
    return (attrGain, maxEntropyAttrValue)

def DecisionTree(D, targetAttr, attrs):
    if DEBUG:
        print("=============================================")
        print("=============================================")

    # If all instances in D has same class
    Dclasses = D[targetAttr].unique()
    if (len(Dclasses) == 1):
        return ClassNode(Dclasses[0])

    # If there are no more attributes
    if (len(attrs) == 0):
        majorityClass = D[targetAttr].value_counts().idxmax()
        return ClassNode(majorityClass)

    # Find attribute with max InfoGain
    maxGain = -1
    maxGainAttr = None
    for attr in attrs:
        if DEBUG: print("===============================")
        (gain, maxEntropyAttrValue) = infoGain(D, targetAttr, attr)
        if (gain > maxGain):
            maxGain = gain
            maxGainAttr = attr
    
    if DEBUG:
        print("\n===============================")
        print("> RESULTS:", maxGainAttr, maxGain)
        
    # Select attribute with max InfoGain
    selectedAttr = maxGainAttr
    selectedAttrGain = maxGain
    # Remove attribute from attributes list
    attrs.remove(selectedAttr)

    node = AttrNode(selectedAttr, selectedAttrGain)

    # Node Split: For each different value of the selected attribute
    if DEBUG: print("> New partitions: ")

    # Selected attribute is numeric
    if (is_numeric_dtype(D[selectedAttr])):
        cutoff = D[selectedAttr].mean()

        # A <= cutoff
        Dv = D[D[selectedAttr] <= cutoff]
        if DEBUG: print("\n", Dv)
        if (len(Dv) == 0):
            # Return leaf node labeled with most frequent class Yi in D
            majorityClass = D[targetAttr].value_counts().idxmax()
            node = ClassNode(majorityClass)
        else:
            node.setChild('<= {0:.2f}'.format(cutoff), DecisionTree(Dv, targetAttr, attrs)) 

        # A > cutoff
        Dv = D[D[selectedAttr] > cutoff]
        if DEBUG: print("\n", Dv)
        if (len(Dv) == 0):
            # Return leaf node labeled with most frequent class Yi in D
            majorityClass = D[targetAttr].value_counts().idxmax()
            node = ClassNode(majorityClass)
        else:
            node.setChild('> {0:.2f}'.format(cutoff), DecisionTree(Dv, targetAttr, attrs)) 

    # Selected attribute is categorical
    else:
        attrValues = D[selectedAttr].unique()
        for v in attrValues:
            # Get resulting partition for each value of the selected attribute
            Dv = D[D[selectedAttr] == v]
            if DEBUG: print("\n", Dv)
            # If partition is empty 
            if (len(Dv) == 0):
                # Return leaf node labeled with most frequent class Yi in D
                majorityClass = D[targetAttr].value_counts().idxmax()
                node = ClassNode(majorityClass)
            else:
                node.setChild(v, DecisionTree(Dv, targetAttr, attrs)) 
    return node

def main():
    D = pd.read_csv('example-num.csv', sep=";")
    targetAttr = 'Joga'

    # Initial state
    attrs = D.keys().tolist()
    attrs.remove(targetAttr)

    n = DecisionTree(D, targetAttr, attrs)
    print(n)
    dot.view(cleanup=True)


if __name__ == "__main__":
    main()