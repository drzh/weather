#!/usr/bin/python3

# Usage: prog FILE_prediction FILE_true

import sys
import numpy as np
import lzma
import pickle

_, fp, ft = sys.argv

with lzma.open(fp, 'rb') as f:
    ypred = pickle.load(f)

with lzma.open(ft, 'rb') as f:
    data = pickle.load(f)
    ytrain = data[2]

ypred = np.round(ypred / 100)
ytrain = np.round(ytrain / 100)

for i in range(ypred.shape[0]):
    print(ypred[i], ytrain[i], ypred[i] - ytrain[i], sep = '\t')

