import pandas as pd
import numpy as np
from graphviz import Digraph
from pandas.api.types import is_numeric_dtype, is_string_dtype

from Tree import *

class DecisionTree():
    def __init__(self, D, attrs, targetAttr):
        self.graph = Digraph(edge_attr={'fontsize':'10.0'})
        self.tree = self.__induct(D, attrs, targetAttr)

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
        for attr in attrs:
            attrGain = self.__infoGain(D, targetAttr, attr)
            if (attrGain > maxGain):
                maxGain = attrGain
                maxGainAttr = attr

        # Remove attribute from attributes list
        attrs.remove(maxGainAttr)
        # Create node with selected attribute
        node = AttrNode(maxGainAttr, maxGain, self.graph)

        # > Node Split <
        def nodeSplitAux(Dv, node, edgeValue):
            # If partition is empty
            if (len(Dv) == 0):
                # Return leaf node of majority class
                majorityClass = D[targetAttr].value_counts().idxmax()
                node = ClassNode(majorityClass, len(D), self.graph)
            else:
                # Connect node to sub-tree
                node.setChild(edgeValue, self.__induct(Dv, attrs, targetAttr))
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

    @classmethod
    def fromData(cls, D, attrs, targetAttr):
        return cls(D, attrs, targetAttr)

    def show(self):
        self.graph.view(cleanup=True)