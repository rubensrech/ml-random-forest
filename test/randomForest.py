import pandas as pd
from math import sqrt
import random

import os
import sys
LIB_PATH = os.path.join(os.path.dirname(__file__), '../lib')
sys.path.append(LIB_PATH)

from RandomForest import *

def main():
    if (len(sys.argv) < 5):
        print("Usage: python3 %s <dataset-csv> <separator> <target-attr> <ntree>" % sys.argv[0])
        exit(-1)

    datasetFile = sys.argv[1]
    separator = sys.argv[2]
    targetAttr = sys.argv[3]
    ntree = int(sys.argv[4])

    random.seed(1)

    # Read dataset
    D = pd.read_csv(datasetFile, sep=separator)
    # Get the number of possible values for each attribute in dataset
    attrsNVals = D.nunique()
    # Build Random Forest
    forest = RandomForest(D, targetAttr, attrsNVals, ntree, attrsSampleFn=sqrt, graph=True)

    if ntree <= 6:
        forest.render()

    # Test classification
    instance = D.iloc[200]
    print("> Test instance:")
    print(instance)
    print("> Prediction: %s" % forest.classify(instance))
    

if __name__ == "__main__":
    main()