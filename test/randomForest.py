import pandas as pd
import numpy as np

import os
import sys
LIB_PATH = os.path.join(os.path.dirname(__file__), '../lib')
sys.path.append(LIB_PATH)

from RandomForest import *

def getFolds(D, targetAttr, K):
    DClasses = D.groupby([targetAttr]).agg({ targetAttr: 'count' })
    DClasses['InstsPerFold'] = DClasses.apply(lambda x: x/K)

    classSets = {}
    classes = D[targetAttr].unique()
    for c in classes:
        classSets[c] = D[D[targetAttr] == c]

    folds = {}
    for i in range(K):
        for j, c in enumerate(classes):
            classInstsCount = int(DClasses.loc[c]['InstsPerFold'])
            classSample = classSets[c].sample(n=classInstsCount)
            if i not in folds:
                folds[i] = classSample
            else:
                folds[i] = folds[i].append(classSample, sort=False)
            classSets[c] = classSets[c].drop(classSample.index)

    folds = list(folds.values())
    print("%d folds" % len(folds))
    print(folds)
    for i in range(K):
        print(">>> Fold %d <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<" % i)
        print(folds[i][targetAttr])
        print(folds[i].groupby([targetAttr]).agg({ targetAttr: 'count' }).apply(lambda x: x/K))
    
    # print(classSets)
    # print(DClasses)
    # print(len(D)/K)

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

    getFolds(D, targetAttr, 10)
    return

    # Build Random Forest
    forest = RandomForest(D, targetAttr, ntree, graph=False)

    if ntree <= 6:
        forest.render()

    # Test classification
    instance = D.iloc[4]
    print("> Test instance:")
    print(instance)
    print("> Prediction: %s" % forest.classify(instance))

    

if __name__ == "__main__":
    main()