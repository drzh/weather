#!/usr/bin/python3

import sys
import lzma
import numpy as np
from netCDF4 import Dataset

_, fi, fo = sys.argv

# Read the data
g16nc = Dataset(fi, 'r')
var_names = [ii for ii in g16nc.variables]
var_name = var_names[0]
data = g16nc.variables[var_name][:]

# close file when finished
g16nc.close()
g16nc = None

mydata = np.concatenate(data)

with lzma.open(fo, 'wb') as f:
    np.save(f, mydata)
