import pandas as pd
import numpy as np
from graphviz import Digraph
from pandas.api.types import is_numeric_dtype, is_string_dtype

np.random.seed(1)

from Tree import *
import ValidationTools

class DecisionTree:
    uid = 0

    def __init__(self, D, targetAttr, attrsNVals, attrsSampleFn=None, graph=True):
        self.graph = self.__createGraph() if graph else None

        self.targetAttr = targetAttr
        self.attrsNVals = attrsNVals
        self.attrsSampleFn = attrsSampleFn

        attrs = D.keys().tolist()
        attrs.remove(targetAttr)
        self.tree = self.__induct(D, attrs)

    def __createGraph(self):
        self.filename = "Tree%d" % DecisionTree.uid
        DecisionTree.uid += 1
        return Digraph(filename=self.filename, edge_attr={'fontsize':'10.0'}, format="pdf")
        
    def __infoGain(self, D, attr):
        targetAttr = self.targetAttr
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

    def __induct(self, D, attrs):
        targetAttr = self.targetAttr
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
        attrsSample = ValidationTools.attrsSample(attrs, self.attrsSampleFn) if (self.attrsSampleFn is not None) else attrs
        for attr in attrsSample:
            attrGain = self.__infoGain(D, attr)
            if (attrGain > maxGain):
                maxGain = attrGain
                maxGainAttr = attr

        # Remove attribute from attributes list
        attrs.remove(maxGainAttr)

        # > Node Split <
        node = None
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
                node.setLeftChild(self.__induct(Dv, attrs))
            # A > cutoff
            Dv = D[D[maxGainAttr] > cutoff]
            # If partition is empty
            if (len(Dv) == 0):
                # Return leaf node of majority class
                majorityClass = D[targetAttr].value_counts().idxmax()
                return ClassNode(majorityClass, len(D), self.graph)
            else:
                node.setRightChild(self.__induct(Dv, attrs))
        # If selected attribute is categorical
        else:
            attrValues = D[maxGainAttr].unique()
            # Prevent creation of node without edge for each possible attribute value
            if len(attrValues) < self.attrsNVals[maxGainAttr]:
                majorityClass = D[targetAttr].value_counts().idxmax()
                return ClassNode(majorityClass, len(D), self.graph)
                
            # Create node of selected categorical attribute
            node = AttrNode(maxGainAttr, maxGain, self.graph)
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
                    node.setChild(v, self.__induct(Dv, attrs))

        return node

    def classify(self, instance):
        prediction = None
        currNode = self.tree
        while prediction is None:
            if TreeNode.isClassNode(currNode):
                prediction = currNode.value
            else:
                instVal = instance[currNode.attr]
                nextNode = currNode.getChild(instVal)
                if nextNode is None:
                    currNode = nextNode
                else:
                    currNode = nextNode

        return prediction

    def render(self):
        if self.graph is not None:
            self.graph.view(cleanup=True)
        else:
            raise Exception('Nothing to render')