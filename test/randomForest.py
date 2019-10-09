import pandas as pd
import numpy as np

import os
import sys
LIB_PATH = os.path.join(os.path.dirname(__file__), '../lib')
sys.path.append(LIB_PATH)

from RandomForest import *
from ValidationTools import *

def main():
    if (len(sys.argv) < 4):
        print("Usage: python3 %s <dataset-csv> <target-attr> <separator> <ntree>" % sys.argv[0])
        exit(-1)

    datasetFile = sys.argv[1]
    targetAttr = sys.argv[2]
    separator = sys.argv[3]
    ntree = int(sys.argv[4])

    # Read dataset
    D = pd.read_csv(datasetFile, sep=separator)

    # Build Random Forest
    forest = RandomForest(D, targetAttr, ntree, graph=True)

    if ntree <= 6:
        forest.render()

    # Test classification
    instance = D.iloc[4]
    print("> Test instance:")
    print(instance)
    print("> Prediction: %s" % forest.classify(instance))

    

if __name__ == "__main__":
    main()