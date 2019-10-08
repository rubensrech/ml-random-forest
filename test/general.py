import pandas as pd
import numpy as np

import os
import sys
LIB_PATH = os.path.join(os.path.dirname(__file__), '../lib')
sys.path.append(LIB_PATH)

from RandomForest import *

def main():
    if (len(sys.argv) < 3):
        print("Usage: python3 %s <dataset-csv> <target-attr> <ntree>" % sys.argv[0])
        exit(-1)

    datasetFile = sys.argv[1]
    targetAttr = sys.argv[2]
    ntree = int(sys.argv[3])

    D = pd.read_csv(datasetFile, sep=";")

    forest = RandomForest(D, targetAttr, ntree)

if __name__ == "__main__":
    main()