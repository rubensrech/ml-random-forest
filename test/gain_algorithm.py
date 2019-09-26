from graphviz import Digraph
import pandas as pd
from math import *
import numpy as np 

### Pandas Examples ###
# > MAX
#   <DataFrame>.reset_index().max()['<column>']

def info(x):
    return x * np.log2(1/x)

def main():
    D = pd.read_csv('example.csv', sep=";")
    targetAttr = 'Joga'

    # Dataset (D) original entropy
    totalEntropy = D.groupby(targetAttr).size().apply(lambda x: info(x/len(D))).agg('sum')
    
    # Calculate entropy after partitioning with attribute 'currAttr'
    currAttr = 'Tempo'
    # Partitionate dataset D using 'currAttr'
    attrPartitionsCount = D.groupby([currAttr, targetAttr])[targetAttr].count()
    # Create table with resultant partitions Dj
    attrPartitions = D.groupby([currAttr]).agg({ targetAttr: 'count' })
    # Calculate |Dj|/|D| for each partition Dj
    attrPartitions['Prop'] = attrPartitionsCount.groupby(level=0).apply(lambda x: x.sum()/len(D))
    # Calculate entropy Info(Dj) for each parition Dj 
    attrPartitions['EntropyTerm'] = attrPartitionsCount.groupby(level=0).agg(lambda x: np.sum(info(x/x.sum())))
    # Calculate |Dj|/|D| * Info(Dj) for each parition Dj
    attrPartitions['Entropy'] = attrPartitions['Prop'] * attrPartitions['EntropyTerm']
    # Calculate entropy InfoA(D) of dataset after partitioning with attribute 'currAttr' 
    currAttrEntropy = attrPartitions['Entropy'].agg('sum')
    # Calculate gain Gain(A)
    currAttrGain = totalEntropy - currAttrEntropy

    print("> Resulting paritions for attribute: " + currAttr)
    print(attrPartitions)
    print("> Resulting entropy: " + str(currAttrEntropy))
    print("> Resulting gain: " + str(currAttrGain))


if __name__ == "__main__":
    main()