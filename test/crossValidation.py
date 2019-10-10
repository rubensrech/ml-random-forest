import pandas as pd
from math import sqrt

import os
import sys
LIB_PATH = os.path.join(os.path.dirname(__file__), '../lib')
sys.path.append(LIB_PATH)

import ValidationTools

def main():
    if (len(sys.argv) < 4):
        print("Usage: python3 %s <dataset-csv> <separator> <target-attr>" % sys.argv[0])
        exit(-1)

    datasetFile = sys.argv[1]
    separator = sys.argv[2]
    targetAttr = sys.argv[3]
    # K = int(sys.argv[4])
    # fstNtree = int(sys.argv[5])
    # lstNtree = int(sys.argv[6])
    # stepNtree = int(sys.argv[7])

    # Read dataset
    D = pd.read_csv(datasetFile, sep=separator)

    # Run Cross Validation
    K = 10
    ntrees = [1, 3, 5, 7, 9, 13, 17, 27, 39]
    ValidationTools.crossValidation(D, targetAttr, K, ntrees, attrsSampleFn=sqrt)

if __name__ == "__main__":
    main()