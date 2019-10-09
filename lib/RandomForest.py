import time
from DecisionTree import *
from ValidationTools import *

class RandomForest:
    def __init__(self, D, targetAttr, ntree, graph=True):
        startTime = time.time()
        self.trees = []
        # Training 'ntree' trees with different bootstraps
        for i in range(ntree):
            print("> Training trees... (%d/%d)" % (i+1, ntree), end='\r')
            # Get bootstraps
            (trainSet, testSet) = ValidationTools.bootstrap(D)
            # Create Decision Tree from training set
            tree = DecisionTree(trainSet, targetAttr, graph=graph)
            # Add tree to the ensemble
            self.trees.append(tree)
        trainingTime = time.time() - startTime
        print("> Training duration: %f s" % trainingTime)

    def classify(self, instance):
        preds = []
        # Classify using each tree individually
        for tree in self.trees:
            preds.append(tree.classify(instance))
        # Get the majority class
        return max(set(preds), key=preds.count)

    def render(self):
        for tree in self.trees:
            if tree.graph is not None:
                tree.render()