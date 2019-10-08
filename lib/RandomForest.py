from DecisionTree import *

class RandomForest:
    def __init__(self, D, targetAttr, ntree):
        self.trees = []
        for i in range(ntree):
            (trainSet, testSet) = self.bootstrap(D)
            tree = DecisionTree.fromData(D, targetAttr)
            tree.show()
            self.trees.append(tree)

    def bootstrap(self, D, frac=0.8):
        train = D.sample(frac=frac, replace=True)
        test = D.drop(train.index)
        return (train, test)