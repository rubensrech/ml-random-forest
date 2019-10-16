import pandas as pd
import numpy as np
from graphviz import Digraph
from pandas.api.types import is_numeric_dtype, is_string_dtype

from Tree import *
import ValidationTools as vt

class DecisionTree:
    uid = 0

    def __init__(self, D, targetAttr, attrsNVals, attrsSampleFn=None, graph=True):
        self.graph = self.__createPlot() if graph else None

        self.targetAttr = targetAttr
        self.attrsNVals = attrsNVals
        self.attrsSampleFn = attrsSampleFn

        attrs = D.keys().tolist()
        attrs.remove(targetAttr)
        self.tree = self.__build(D, np.array(attrs))

    def __createPlot(self):
        self.filename = "Tree%d" % DecisionTree.uid
        DecisionTree.uid += 1
        return Digraph(filename=self.filename, edge_attr={'fontsize':'10.0'}, format="pdf")
        
    def __calculateInfoGains(self, D, attrs):
        targetAttr = self.targetAttr
        classValues = D[targetAttr].nunique()

        def info(x):
            return np.nan_to_num(-x * np.log2(x, where=(x>0)))

        def vectorizeAttrPart(attr):
            partData = D.groupby([attr, targetAttr])[targetAttr].count().unstack().fillna(0).stack()
            attrValues = D[attr].nunique()
            return partData.to_numpy().reshape((attrValues, classValues))
        
        def attrEntropy(attr):
            # Vectorization
            partVectorizedData = vectorizeAttrPart(attr)
            # Count instances in each partition
            partitionsSize = np.sum(partVectorizedData, axis=1)
            # Calculate proportion |Dj|/|D| for each partition Dj
            prop = partitionsSize / len(D)
            # Calculate entropy Info(Dj) for each parition Dj
            partEntropy = np.sum(info(partVectorizedData / partitionsSize[:, None]), axis=1)
            # Calculate entropy InfoA(D) of dataset after partitioning with attribute 'currAttr'
            newEntropy = np.sum(prop * partEntropy)
            return newEntropy

        # Calculate entropy for each attr
        attrsEntropy = np.array([attrEntropy(attr) for attr in attrs])
        # Calculate dataset (D) current entropy
        currEntropy = np.sum(info(D.groupby(targetAttr).size().to_numpy() / len(D)))
        # Calculate info gain
        attrsInfoGain = currEntropy - attrsEntropy
        return attrsInfoGain
        
    def __build(self, D, attrs):
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
        attrsSample = vt.attrsSample(attrs, sampleFn) if (sampleFn is not None) else attrs
        attrsInfoGain = self.__calculateInfoGains(D, attrsSample)
        maxGainAttrIndex = np.argmax(attrsInfoGain)
        maxGain = attrsInfoGain[maxGainAttrIndex]
        maxGainAttr = attrsSample[maxGainAttrIndex]

        attrs = attrs[attrs != maxGainAttr]
        
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
                node.setLeftChild(self.__build(Dv, attrs))
            # A > cutoff
            Dv = D[D[maxGainAttr] > cutoff]
            # If partition is empty
            if (len(Dv) == 0):
                # Return leaf node of majority class
                majorityClass = D[targetAttr].value_counts().idxmax()
                return ClassNode(majorityClass, len(D), self.graph)
            else:
                node.setRightChild(self.__build(Dv, attrs))
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
                    node.setChild(v, self.__build(Dv, attrs))

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