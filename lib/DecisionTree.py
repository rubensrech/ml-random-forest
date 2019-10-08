import pandas as pd
import numpy as np
from graphviz import Digraph
from pandas.api.types import is_numeric_dtype, is_string_dtype
from math import ceil, sqrt
import random

from Tree import *

# from datetime import datetime
# random.seed(datetime.now())
random.seed(1)

class DecisionTree:
    uid = 0

    def __init__(self, D, targetAttr):
        self.__createGraph()

        attrs = D.keys().tolist()
        attrs.remove(targetAttr)
        self.tree = self.__induct(D, attrs, targetAttr)

    def __createGraph(self):
        self.filename = "Tree%d" % DecisionTree.uid
        self.graph = Digraph(filename=self.filename, edge_attr={'fontsize':'10.0'}, format="pdf")
        DecisionTree.uid += 1

    def __infoGain(self, D, targetAttr, attr):
        def info(x): return x * np.log2(1/x)

        # Calculate dataset (D) current entropy
        currEntropy = D.groupby(targetAttr).size().apply(lambda x: info(x/len(D))).agg('sum')

        # Partitionate dataset D using 'attr'
        attrPartitionsCount = D.groupby([attr, targetAttr])[targetAttr].count()
        # Create (auxiliary) table with resultant partitions Dj
        attrPartitions = D.groupby([attr]).agg({ targetAttr: 'count' })
        # Calculate |Dj|/|D| for each partition Dj
        attrPartitions['Prop'] = attrPartitions[targetAttr].apply(lambda x: x/len(D))
        # Calculate entropy Info(Dj) for each parition Dj 
        attrPartitions['PartEntropy'] = attrPartitionsCount.groupby(level=0).agg(lambda x: np.sum(info(x/x.sum())))
        # Calculate |Dj|/|D| * Info(Dj) for each parition Dj
        attrPartitions['Entropy'] = attrPartitions['Prop'] * attrPartitions['PartEntropy']
        # Calculate entropy InfoA(D) of dataset after partitioning with attribute 'currAttr' 
        newEntropy = attrPartitions['Entropy'].agg('sum')
        # Calculate gain Gain(A)
        attrGain = currEntropy - newEntropy
        
        return attrGain

    def __induct(self, D, attrs, targetAttr):
        # If all instances in D has same class
        Dclasses = D[targetAttr].unique()
        if (len(Dclasses) == 1):
            # Return leaf node of only class
            return ClassNode(Dclasses[0], len(D), self.graph)
        
        # If there are no more attributes for splitting
        if (len(attrs) == 0):
            # Return leaf node of majority class
            majorityClass = D[targetAttr].value_counts().idxmax()
            return ClassNode(majorityClass, len(D), self.graph)

        # Find attribute with max info gain
        maxGain = -1
        maxGainAttr = None
        # Attributes sampling
        m = ceil(sqrt(len(attrs)))
        attrsSample = random.sample(attrs, m)
        for attr in attrsSample:
            attrGain = self.__infoGain(D, targetAttr, attr)
            if (attrGain > maxGain):
                maxGain = attrGain
                maxGainAttr = attr

        # Remove attribute from attributes list
        attrs.remove(maxGainAttr)

        # > Node Split <
        node = None

        def numNodeSplit(Dv, node, leftOrRight):
            # If partition is empty
            if (len(Dv) == 0):
                # Return leaf node of majority class
                majorityClass = D[targetAttr].value_counts().idxmax()
                node = ClassNode(majorityClass, len(D), self.graph)
            else:
                # Connect node to sub-tree
                if leftOrRight == 'left':
                    node.setLeftChild(self.__induct(Dv, attrs, targetAttr))
                else:
                    node.setRightChild(self.__induct(Dv, attrs, targetAttr))
            return node

        # If selected attribute is numeric
        if (is_numeric_dtype(D[maxGainAttr])):
            # Define attribute division cutoff
            cutoff = D.groupby([targetAttr])[maxGainAttr].mean().agg('mean') # D[maxGainAttr].mean()
            # Create node of selected numerical attribute
            node = NumAttrNode(maxGainAttr, maxGain, self.graph, cutoff)
            # A <= cutoff
            Dv = D[D[maxGainAttr] <= cutoff]
            # If partition is empty
            if (len(Dv) == 0):
                # Return leaf node of majority class
                majorityClass = D[targetAttr].value_counts().idxmax()
                return ClassNode(majorityClass, len(D), self.graph)
            else:
                node.setLeftChild(self.__induct(Dv, attrs, targetAttr))
            # A > cutoff
            Dv = D[D[maxGainAttr] > cutoff]
            # If partition is empty
            if (len(Dv) == 0):
                # Return leaf node of majority class
                majorityClass = D[targetAttr].value_counts().idxmax()
                return ClassNode(majorityClass, len(D), self.graph)
            else:
                node.setRightChild(self.__induct(Dv, attrs, targetAttr))
            node = numNodeSplit(Dv, node, 'right')
        # If selected attribute is categorical
        else:
            # Create node of selected categorical attribute
            node = AttrNode(maxGainAttr, maxGain, self.graph)
            attrValues = D[maxGainAttr].unique()
            # For each different value of the selected attribute
            for v in attrValues:
                # Get resulting partition for each value of the selected attribute
                Dv = D[D[maxGainAttr] == v]
                # If partition is empty
                if (len(Dv) == 0):
                    # Return leaf node of majority class
                    majorityClass = D[targetAttr].value_counts().idxmax()
                    return ClassNode(majorityClass, len(D), self.graph)
                else:
                    # Connect node to sub-tree
                    node.setChild(v, self.__induct(Dv, attrs, targetAttr))

        return node

    @classmethod
    def fromData(cls, D, targetAttr):
        return cls(D, targetAttr)

    def classify(self, instance):
        prediction = None
        currNode = self.tree
        while prediction is None:
            if TreeNode.isClassNode(currNode):
                prediction = currNode.value
            else:
                instVal = instance[currNode.attr]
                currNode = currNode.getChild(instVal)

        return prediction

    def render(self):
        self.graph.view(cleanup=True)