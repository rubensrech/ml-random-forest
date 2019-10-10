import pandas as pd
from math import floor
from statistics import mean, stdev
from math import sqrt, ceil
import random
import time

random.seed(1)

from RandomForest import *

def dataSample(D, n):
    indexes = random.sample(list(D.index.values), n)
    return D.loc[indexes]

def attrsSample(attrs, fn):
    m = ceil(fn(len(attrs)))
    return random.sample(attrs, m)

def bootstrap(D, frac=1):
    trainIndexes = random.choices(D.index, k=ceil(len(D)*frac))
    trainSet = D.loc[trainIndexes]
    testSet = D.drop(trainIndexes)
    return (trainSet, testSet)

def getKFolds(D, targetAttr, K):
    kfolds = {}

    # Calculate number of instances of each class to be in each fold
    classesInstsCount = D.groupby([targetAttr]).agg({ targetAttr: 'count' })
    classesInstsCount['InstsPerFold'] = classesInstsCount.apply(lambda x: x/K)

    classes = D[targetAttr].unique()
    for c in classes:
        # Get instances of class 'c' in 'D'
        classSet = D[D[targetAttr] == c]
        # Get number of instances of class 'c' to be in each fold
        classInstsCount = floor(classesInstsCount.loc[c]['InstsPerFold'])
        # For each fold
        for i in range(K):
            # Sample instances of class 'c'
            classSample = dataSample(classSet, classInstsCount)
            # Add sample to fold
            if i not in kfolds:
                kfolds[i] = classSample
            else:
                kfolds[i] = kfolds[i].append(classSample, sort=False)
            # Remove sampled instances from class 'c' instances
            classSet = classSet.drop(classSample.index)
    
    return list(kfolds.values())

def crossValidation(D, targetAttr, K, ntree, attrsSampleFn=sqrt):
    print("============ CROSS VALIDATION ============")
    startTime = time.time()
    # Get the number of possible values for each attribute in dataset
    attrsNVals = D.nunique()
    kfolds = getKFolds(D, targetAttr, K)
    F1s = []
    precisionAcc = 0
    recallAcc = 0
    for i in range(K):
        # Select test fold
        testFold = kfolds[i]
        # Select training folds (merge K-1 folds)
        trainFolds = pd.concat([f for j,f in enumerate(kfolds) if j != i])
        # Create Random Forest with training folds
        print("> TRAINING Random Forest (%d/%d)" % (i+1, K))
        forest = RandomForest(trainFolds, targetAttr, attrsNVals, ntree, attrsSampleFn=attrsSampleFn, graph=False)
        # # Evaluate Random Forest with test fold
        print("> EVALUATING Random Forest (%d/%d)" % (i+1, K))
        (precision, recall, F1) = forest.evaluate(testFold)
        F1s.append(F1)
        precisionAcc += precision
        recallAcc += recall
    # Calculate average performance
    avgPerf = mean(F1s)
    stdevPerf = stdev(F1s)
    avgPrecision = precisionAcc / K
    avgRecall = recallAcc / K
    totalTime = time.time() - startTime
    # Print results
    print("------------ x ------------")
    print("RESULTS")
    print("- K = %d" % K)
    print("- NTREE = %d" % ntree)
    print("- Total duration: %f s" % totalTime)
    print("- Average Performance: %f" % avgPerf)
    print("- Performance Std Deviation: %f" % stdevPerf)
    print("- Average Precision: %f" % avgPrecision)
    print("- Average Recall: %f" % avgRecall)
    print("==========================================")
    return (avgPerf, stdevPerf, avgPrecision, avgRecall)