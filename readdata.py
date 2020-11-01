"""
    This module provide function to read data according to the file list
"""

import lzma
import pickle
import math
import multiprocessing

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

# Process each line of files
def readline(line):
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

    return [train, pred]

# Process lines
def readlines(lines):
    dout = []
    for line in lines:
        d = readline(line)
        dout.append(d)
    return dout

# worker
def worker(idx, lines, return_dict):
    d = readlines(lines)
    return_dict[idx] = d

# Process linespt
def readlinespt(linespt):
    manager = multiprocessing.Manager()
    return_dict = manager.dict()
    jobs = []
    dim1 = len(linespt)
    for i in range(dim1):
        lines = linespt[i]
        p = multiprocessing.Process(target=worker, args=(i, lines, return_dict))
        jobs.append(p)
        p.start()

    for p in jobs:
        p.join()
    
    dout = []
    idxs = list(return_dict.keys())
    idxs.sort()
    for i in idxs:
        dout += return_dict[i]
    return dout

# Process file list
def readdata(flist, n = 1, threads = 1):
    npt = math.ceil(n / threads) if threads > 1 else 1
    with open(flist, 'r') as f:
        i = 0
        dout = []
        lines = []
        linespt = []  # list of lines in each thread
        proc = []
        for line in f:
            lines.append(line)
            i += 1
            if i % npt == 0:
                # Push npt lines to linespt
                linespt.append(lines)
                lines = []

            if i % n == 0:
                # Process n lines in threads
                if lines:
                    linespt.append(lines)
                    lines = []
                dout = readlinespt(linespt)
                yield dout
                linespt = []
