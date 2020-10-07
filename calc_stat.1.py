#!/usr/bin/python3

# Usage: prog FILE_pickle_xz | <STDOUT>
# Usage: Calculate the statistics

import sys
import lzma
import pickle
import numpy as np

_, fi = sys.argv

with lzma.open(fi, 'rb') as f :
    data = pickle.load(f)

print(len(data))
