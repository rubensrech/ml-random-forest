import time
import numpy as np
from DecisionTree import *
from ValidationTools import *

class RandomForest:
    def __init__(self, D, targetAttr, attrsNVals, ntree, attrsSampleFn=None, graph=True):
        startTime = time.time()
        self.targetAttr = targetAttr
        self.trees = []
        # Training 'ntree' trees with different bootstraps
        for i in range(ntree):
            print(">> Training trees... (%d/%d)" % (i+1, ntree), end='\r')
            # Get bootstraps
            (trainSet, testSet) = ValidationTools.bootstrap(D)
            # Create Decision Tree from training set
            tree = DecisionTree(trainSet, targetAttr, attrsNVals, attrsSampleFn=attrsSampleFn, graph=graph)
            # Add tree to the ensemble
            self.trees.append(tree)
        trainingTime = time.time() - startTime
        print(">> Training duration: %f s" % trainingTime)

    def classify(self, instance):
        preds = []
        # Classify using each tree individually
        for tree in self.trees:
            preds.append(tree.classify(instance))
        # Get the majority class
        return max(set(preds), key=preds.count)

    def evaluate(self, Dt):
        targetAttr = self.targetAttr
        # Classify all instances in test set
        preds = Dt.apply(lambda x: self.classify(x), axis=1)

        # cm = pd.crosstab(Dt[targetAttr], Dt['Preds'], rownames=['Actual'], colnames=['Predicted'], dropna=False)
        # print(cm)

        # Calculate TP, FP and FN for each class
        classes = Dt[targetAttr].unique()
        classesData = { 'TP':[], 'FP':[], 'FN':[] }
        for c in classes:
            tp = len(Dt[(Dt[targetAttr] == c) & (preds == c)])
            fp = len(Dt[(preds == c) & (Dt[targetAttr] != c)])
            fn = len(Dt[(Dt[targetAttr] == c) & (preds != c)])
            classesData['TP'].append(tp)
            classesData['FP'].append(fp)
            classesData['FN'].append(fn)
            # print("Class: %s (TP = %d, FP = %d, FN = %d)" % (c, tp, fp, fn))

        classesMetrics = pd.DataFrame(classesData, index=classes)

        # Calculate macro metrics (precision and recall)
        Prec = lambda c: c['TP']/(c['TP']+c['FP']) if c['TP']+c['FP'] > 0 else 0
        Rec = lambda c: c['TP']/(c['TP']+c['FN']) if c['TP']+c['FN'] > 0 else 0
        classesMetrics['Precision'] = classesMetrics.apply(Prec, axis=1)
        classesMetrics['Recall'] = classesMetrics.apply(Rec, axis=1)
        prec = classesMetrics['Precision'].mean()
        rec = classesMetrics['Recall'].mean()

        # Calculate F1-measure
        F1 = 2 * (prec * rec)/(prec + rec) if (prec + rec) > 0 else 0
        # print(classesMetrics)
        # print(F1)
        return F1

    def render(self):
        for tree in self.trees:
            if tree.graph is not None:
                tree.render()