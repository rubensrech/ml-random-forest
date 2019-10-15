import pandas as pd
import numpy as np
from graphviz import Digraph
from pandas.api.types import is_numeric_dtype, is_string_dtype

import numba as nb

from Tree import *
import ValidationTools

class DecisionTree:
    uid = 0

    def __init__(self, D, targetAttr, attrsNVals, attrsSampleFn=None, graph=True):
        self.graph = self.createGraph() if graph else None

        self.targetAttr = targetAttr
        self.attrsNVals = attrsNVals
        self.attrsSampleFn = attrsSampleFn

        attrs = D.keys().tolist()
        attrs.remove(targetAttr)
        self.tree = self.induct(D, attrs)

    def createGraph(self):
        self.filename = "Tree%d" % DecisionTree.uid
        DecisionTree.uid += 1
        return Digraph(filename=self.filename, edge_attr={'fontsize':'10.0'}, format="pdf")
        
    def calcAttrsInfoGain(self, D, attrs):
        targetAttr = self.targetAttr

        def info(x):
            return x * np.log2(1/x)

        def attrEntropy(attr):
            # Partitionate dataset D using 'attr'
            attrPartitionsCount = D.groupby([attr, targetAttr])[targetAttr].count()
            # Count instances in each partition
            partitionsSize = D.groupby(attr)[targetAttr].agg('count')
            # Calculate proportion |Dj|/|D| for each partition Dj
            prop = partitionsSize / len(D)
            # Calculate entropy Info(Dj) for each parition Dj 
            partEntropy = attrPartitionsCount.groupby(level=0).agg(lambda x: np.sum(info(x/x.sum())))
            # Calculate entropy InfoA(D) of dataset after partitioning with attribute 'currAttr'
            newEntropy = np.sum(prop * partEntropy)
            return newEntropy

        # Calculate entropy for each attr
        attrsEntropy = np.array([attrEntropy(attr) for attr in attrs])
        # Calculate dataset (D) current entropy
        currEntropy = D.groupby(targetAttr).size().apply(lambda x: info(x/len(D))).agg('sum')
        # Calculate info gain
        attrsInfoGain = currEntropy - attrsEntropy
        return attrsInfoGain
        
    def induct(self, D, attrs):
        targetAttr = self.targetAttr
        sampleFn = self.attrsSampleFn

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
        attrsSample = ValidationTools.attrsSample(attrs, sampleFn) if (sampleFn is not None) else attrs
        attrsInfoGain = self.calcAttrsInfoGain(D, attrsSample)
        maxGainAttrIndex = np.argmax(attrsInfoGain)
        maxGain = attrsInfoGain[maxGainAttrIndex]
        maxGainAttr = attrsSample[maxGainAttrIndex]

        attrs.remove(maxGainAttr)

        # > Node Split <
        node = None
        # If selected attribute is numeric
        if (is_numeric_dtype(D[maxGainAttr])):
            # Define attribute division cutoff
            cutoff = D[maxGainAttr].mean()
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
                node.setLeftChild(self.induct(Dv, attrs))
            # A > cutoff
            Dv = D[D[maxGainAttr] > cutoff]
            # If partition is empty
            if (len(Dv) == 0):
                # Return leaf node of majority class
                majorityClass = D[targetAttr].value_counts().idxmax()
                return ClassNode(majorityClass, len(D), self.graph)
            else:
                node.setRightChild(self.induct(Dv, attrs))
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
                    node.setChild(v, self.induct(Dv, attrs))

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