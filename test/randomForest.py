import pandas as pd
import numpy as np

import os
import sys
LIB_PATH = os.path.join(os.path.dirname(__file__), '../lib')
sys.path.append(LIB_PATH)

from RandomForest import *
from ValidationTools import *

def crossValidation(D, targetAttr, ntree, K):
    print(">> Cross Validation <<")
    attrsNVals = D.nunique()
    kfolds = ValidationTools.getKFolds(D, targetAttr, K)
    for i in range(1): # for i in range(K):
        # Select test fold
        testFold = kfolds[i]
        # Select training folds (merge K-1 folds)
        trainFolds = pd.concat([f for j,f in enumerate(kfolds) if j != i])
        # Create Random Forest with training folds
        print("> Training Random Forest (%d/%d)" % (i+1, K))
        forest = RandomForest(trainFolds, targetAttr, attrsNVals, ntree, graph=False)
        # Evaluate Random Forest with test fold
        print("> Evaluating Random Forest (%d/%d)" % (i+1, K))
        forest.evaluate(testFold)
        

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

    K = 10
    crossValidation(D, targetAttr, ntree, K)

    # Build Random Forest
    # forest = RandomForest(D, targetAttr, ntree, graph=True)

    # if ntree <= 6:
    #     forest.render()

    # # Test classification
    # instance = D.iloc[4]
    # print("> Test instance:")
    # print(instance)
    # print("> Prediction: %s" % forest.classify(instance))
    

if __name__ == "__main__":
    main()