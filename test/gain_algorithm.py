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
    
    return attrGain

def DecisionTree(D, targetAttr, attrs):
    # If all instances in D has same class
    Dclasses = D[targetAttr].unique()
    if (len(Dclasses) == 1):
        # Return leaf node of only class
        return ClassNode(Dclasses[0], len(D))

    # If there are no more attributes
    if (len(attrs) == 0):
        # Return leaf node of majority class
        majorityClass = D[targetAttr].value_counts().idxmax()
        return ClassNode(majorityClass, len(D))

    # Find attribute with max InfoGain
    maxGain = -1
    maxGainAttr = None
    for attr in attrs:
        attrGain = infoGain(D, targetAttr, attr)
        if (gain > maxGain):
            maxGain = attrGain
            maxGainAttr = attr
            
    # Remove attribute from attributes list
    attrs.remove(maxGainAttr)
    # Create node with selected attribute
    node = AttrNode(maxGainAttr, maxGain)

    # > Node Split <
    def nodeSplitAux(Dv, node, edgeValue):
        # If partition is empty
        if (len(Dv) == 0):
            # Return leaf node labeled with most frequent class Yi in D
            majorityClass = D[targetAttr].value_counts().idxmax()
            node = ClassNode(majorityClass, len(D))
        else:
            # Connect node to sub-tree
            node.setChild(edgeValue, DecisionTree(Dv, targetAttr, attrs))
        return node

    # If selected attribute is numeric
    if (is_numeric_dtype(D[maxGainAttr])):
        # Define attribute division cutoff
        cutoff = D[maxGainAttr].mean()
        # A <= cutoff
        Dv = D[D[maxGainAttr] <= cutoff]
        node = nodeSplitAux(Dv, node, '<= {0:.2f}'.format(cutoff))
        # A > cutoff
        Dv = D[D[maxGainAttr] > cutoff]
        node = nodeSplitAux(Dv, node, '> {0:.2f}'.format(cutoff))
    # If selected attribute is categorical
    else:
        attrValues = D[maxGainAttr].unique()
        # For each different value of the selected attribute
        for v in attrValues:
            # Get resulting partition for each value of the selected attribute
            Dv = D[D[maxGainAttr] == v]
            node = nodeSplitAux(Dv, node, v)

    return node

def main():
    D = pd.read_csv('example.csv', sep=";")
    targetAttr = 'Joga'

    attrs = D.keys().tolist()
    attrs.remove(targetAttr)

    n = DecisionTree(D, targetAttr, attrs)
    print(n)
    
    dot.view(cleanup=True)


if __name__ == "__main__":
    main()