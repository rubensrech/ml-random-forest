from graphviz import Digraph
import pandas as pd
from math import *
import numpy as np 

### Pandas Examples ###
# > MAX
#   <DataFrame>.reset_index().max()['<column>']

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

    print("> Resulting paritions for attribute: " + attr)
    print(attrPartitions)
    print("> Resulting entropy: " + str(newEntropy))
    print("> Resulting gain: " + str(attrGain))
    print("> Selected attribute value: " + maxEntropyAttrValue)
    
    return (attrGain, maxEntropyAttrValue)


def main():
    D = pd.read_csv('example.csv', sep=";")
    targetAttr = 'Joga'
    
    # Calculate entropy after partitioning with attribute 'currAttr'
    currAttr = 'Tempo'
    
    (gain, maxEntropyAttrValue) = infoGain(D, targetAttr, currAttr)
    
    print("###################################################")
    selectedAttr = currAttr
    selectedAttrValue = maxEntropyAttrValue
    newD = D[D[selectedAttr] == selectedAttrValue]


if __name__ == "__main__":
    main()