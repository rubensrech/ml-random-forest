import random
import time
import numpy as np

import pandas as pd
from math import sqrt

import os
import sys
LIB_PATH = os.path.join(os.path.dirname(__file__), '../lib')
sys.path.append(LIB_PATH)

import ValidationTools
from RandomForest import RandomForest

def main():
    if (len(sys.argv) < 3):
        print("Usage: python3 %s <dataset-csv> <target-attr>" % sys.argv[0])
        exit(-1)

    datasetFile = sys.argv[1]
    targetAttr = sys.argv[2]
    separator = ','

    random.seed(0)
    np.random.seed(0)

    # Read dataset
    D = pd.read_csv(datasetFile, sep=separator)

    t0 = time.time()

    forest = RandomForest(D, targetAttr, D.nunique(), 10, sqrt, False)

    print("%f" % (time.time() - t0))

if __name__ == "__main__":
    main()