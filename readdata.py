"""
    This module provide function to read data according to the file list
"""

import lzma
import pickle

# Function to read file and return matrix
def readfi(fi):
    with lzma.open(fi, 'rb') as f:
        data = pickle.load(f)
        tem = []
        if len(data) == 3:
            d1 = len(data[0])
            d2 = len(data[1])
            tem = data[2].reshape(d1, d2)
        return tem

# def readdata(flist, mean = 0, sd = 1):
def readdata(flist):
    with open(flist, 'r') as f:
        for line in f:
            line = line.strip()
            e = line.split('\t')
            ftset = e[:-1]
            fp = e[-1]
            train = []
            pred = []

            # Process training files
            for ft in ftset:
                tc = []
                # Process channels in the same time slot
                for f in ft.split(','):
                    dmat = readfi(f)
                    tc.append(dmat)
                train.append(tc)

            # Process prediction files
            for f in fp.split(','):
                dmat = readfi(f)
                pred.append(dmat)

            # yield output
            yield [train, pred]

