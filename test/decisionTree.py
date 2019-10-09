import pandas as pd

import os
import sys
LIB_PATH = os.path.join(os.path.dirname(__file__), '../lib')
sys.path.append(LIB_PATH)

from DecisionTree import *

def main():
    if (len(sys.argv) < 4):
        print("Usage: python3 %s <dataset-csv> <target-attr> <separator>" % sys.argv[0])
        exit(-1)

    datasetFile = sys.argv[1]
    targetAttr = sys.argv[2]
    separator = sys.argv[3]

    # Read dataset
    D = pd.read_csv(datasetFile, sep=separator)

    # Tree induction
    attrsNVals = D.nunique()
    tree = DecisionTree(D, targetAttr, attrsNVals)
    tree.render()

    # Test classification
    instance = D.iloc[4]
    print(instance)
    print(tree.classify(instance))

if __name__ == "__main__":
    main()