from DecisionTree import *

class RandomForest:
    def __init__(self, D, targetAttr, ntree, graph=True):
        self.trees = []
        for i in range(ntree):
            print("> Training trees... (%d/%d)" % (i+1, ntree), end='\r')
            # Get bootstrap sets
            (trainSet, testSet) = self.bootstrap(D)
            # Create Decision Tree from training set
            tree = DecisionTree(trainSet, targetAttr, graph=graph)
            # Add tree to the ensemble
            self.trees.append(tree)

    def bootstrap(self, D, frac=0.8):
        train = D.sample(frac=frac, replace=True)
        test = D.drop(train.index)
        return (train, test)

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