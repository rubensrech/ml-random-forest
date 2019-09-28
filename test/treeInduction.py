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

    D = pd.read_csv(datasetFile, sep=";")
    attrs = D.keys().tolist()
    attrs.remove(targetAttr)

    tree = DecisionTree.fromData(D, attrs, targetAttr)
    tree.show()

if __name__ == "__main__":
    main()