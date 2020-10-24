#!/home2/s167968/bin/python3

# Usage: <STDIN_time_group> | prog | <STDOUT>
#     --step1 : step between adjacent time slots of training
#     --step2 : step between last time slot of training and the time slot of prediction
#     --step3 : step between the first time slot of the training group and the first time slot of the next training group
#     --ns : number of slices of training

import sys
import getopt
import numpy as np

# define channels
chs = ['C07', 'C08', 'C09', 'C10', 'C11', 'C12', 'C13', 'C14', 'C15', 'C16']

# Parsing parameters
try:
    opts, args = getopt.getopt(sys.argv[1:],
                               'n:o:t:p:',
                               ['s1=', 's2=', 's3=', 'ns=',
                               'tprefix=', 'pprefix=',
                                'tsuffix=', 'psuffix=']
    )
except getopt.GetoptError as err:
    print(str(err))
    sys.exit(2)

s1 = 3
s2 = 12
s3 = 3
ns = 17
tprefix = ''
pprefix = ''
tsuffix = ''
psuffix = ''

for o, a in opts:
    if o == '-o':
        fo = a
    elif o == '--s1':
        s1 = int(a)
    elif o == '--s2':
        s2 = int(a)
    elif o == '--s3':
        s3 = int(a)
    elif o in ['-n', '--ns']:
        ns = int(a)
    elif o == '--tprefix':
        tprefix = a
    elif o == '--pprefix':
        pprefix = a
    elif o == '--tsuffix':
        tsuffix = a
    elif o == '--psuffix':
        psuffix = a
    else:
        assert False, 'unhandled option'

# Error printing
def eprint(msg):
    print(msg, file = sys.stderr)

# Read in the file list
f = sys.stdin
for line in f:
    line = line.strip()
    e = line.split('\t')
    tlen = len(e)
    t0 = 0
    tpred = t0 + (ns - 1) * s1 + s2
    while tpred < tlen:
        train = []
        pred = []
        t = t0
        n = 0
        # add training
        while n < ns:
            fs = ','.join([tprefix + e[t] + '.' + c + tsuffix for c in chs])
            train.append(fs)
            n += 1
            t += s1
        # add prediction
        fs = ','.join([pprefix + e[tpred] + '.' + c + psuffix for c in chs])
        pred = fs
        # output
        print('\t'.join(train), pred, sep='\t')
        t0 += s3
        tpred += s3
