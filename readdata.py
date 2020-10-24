"""
    This module provide function to read data according to the file list
"""

import lzma
import pickle

# Function to read file and return matrix
def readfi(fi, mean, sd):
    with lzma.open(fi, 'rb') as f:
        data = pickle.load(f)
        tem = []
        if len(data) == 3:
            d1 = len(data[0])
            d2 = len(data[1])
            tem = data[2].reshape(d1, d2)
            # Scale the data
            if mean != 0:
                tem = tem - mean
            if sd > 0 and sd != 1:
                tem = tem / sd
        return tem

def readdata(flist, mean = 0, sd = 1):
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
                    dmat = readfi(f, mean, sd)
                    tc.append(dmat)
                train.append(tc)

            # Process prediction files
            for f in fp.split(','):
                dmat = readfi(f, mean, sd)
                pred.append(dmat)

            # yield output
            yield [train, pred]

