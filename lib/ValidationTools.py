import pandas as pd
from math import floor
from statistics import mean, stdev
from math import sqrt

from RandomForest import *

def bootstrap(D, frac=1):
    train = D.sample(frac=frac, replace=True, random_state=1)
    test = D.drop(train.index)
    return (train, test)

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
            classSample = classSet.sample(n=classInstsCount, random_state=1)
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
    # Get the number of possible values for each attribute in dataset
    attrsNVals = D.nunique()
    kfolds = getKFolds(D, targetAttr, K)
    performances = []
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
        performance = forest.evaluate(testFold)
        performances.append(performance)
        print(">> Test fold %d => Performance %f" % (i, performance))
    # Calculate average performance
    avgPerf = mean(performances)
    stdevPerf = stdev(performances)
    # Print results
    print("------------ x ------------")
    print("RESULTS")
    print("- K = %d" % K)
    print("- NTREE = %d" % ntree)
    print("- Average Performance: %f" % avgPerf)
    print("- Performance Std Deviation: %f" % stdevPerf)
    print("==========================================")
    return (avgPerf, stdevPerf)