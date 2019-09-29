import pandas as pd
import sys
sys.path.append('../lib')

from DecisionTree import *

def main():
    if (len(sys.argv) < 3):
        print("Usage: python3 %s <dataset-csv> <target-attr>" % sys.argv[0])
        exit(-1)

    datasetFile = sys.argv[1]
    targetAttr = sys.argv[2]

    # Read dataset
    D = pd.read_csv(datasetFile, sep=";")    

    # Tree induction
    tree = DecisionTree.fromData(D, targetAttr)
    tree.show()

    # Test classification
    instance = D.iloc[4]
    print(instance)
    print(tree.classify(instance))

if __name__ == "__main__":
    main()