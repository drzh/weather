#!/home2/s167968/bin/python3

# Usage: <STDIN_file_list> | porg -m FLOAT_mean -s FLOAT_sd -o FILE_out.xz

import sys
import numpy as np
import lzma
import pickle
import getopt

# Function to print error
def eprint(msg):
    print(msg, file = sys.stderr)

# Parsing parameters
try :
    opts, args = getopt.getopt(sys.argv[1:],
                               'm:s:o:',
                               ['mean=', 'sd=', 'out='
                               ]
    )
except getopt.GetoptError as err :
    eprint(str(err))
    sys.exit(2)

mean = 0
sd = 1
fo = ''

for o, a in opts:
    if o in ('-m', '--mean'):
        mean = float(a)
    if o in ('-s', '--sd'):
        sd = float(a)
    if o in ('-o', '--out'):
        fo = a

# Check fo
if fo == '':
    eprint('No output file')
    sys.exit(1)

# Function to read file and return matrix
def readfi(fi):
    with lzma.open(fi, 'rb') as f:
        data = pickle.load(f)
        tem = []
        if len(data) == 3:
            d1 = len(data[0])
            d2 = len(data[1])
            tem = data[2].reshape(d1, d2)
            if mean != 0:
                tem = tem - mean
            if sd > 0 and sd != 1:
                tem = tem / sd
        return tem

dout = []
f = sys.stdin
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

    # Round
    # train = np.around(train, 3)
    # pref = np.around(pred, 3)
    
    # Combine output
    dout.append([train, pred])

# Output
with lzma.open(fo, 'wb') as f:
    pickle.dump(dout, f)
