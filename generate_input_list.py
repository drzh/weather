#!/home2/s167968/bin/python3

# Usage: <STDIN_file_channel_order> | prog -o FILE_out
#     --step1 : step between adjacent time slots of training
#     --step2 : step between last time slot of training and the time slot of prediction
#     --step3 : step between the first time slot of the training group and the first time slot of the next training group
#     --ns : number of slices of training
#     -o : output file

import sys
import getopt
import numpy as np

# define channels
chs = ['C07', 'C08', 'C09', 'C10', 'C11', 'C12', 'C13', 'C14', 'C15', 'C16']

# Parsing parameters
try :
    opts, args = getopt.getopt(sys.argv[1:],
                               'n:o:',
                               ['s1=', 's2=', 's3=', 'ns=']
    )
except getopt.GetoptError as err :
    print(str(err))
    sys.exit(2)

s1 = 6
s2 = 12
s3 = 24
ns = 16
fo = ''
for o, a in opts :
    if o == '-o' :
        fo = a
    elif o == '--s1' :
        s1 = int(a)
    elif o == '--s2' :
        s2 = int(a)
    elif o == '--s3' :
        s3 = int(a)
    elif o in ['-n', '--ns'] :
        ns = int(a)
    else :
        assert False, 'unhandled option'

# Error printing
def eprint(msg) :
    print(msg, file = sys.stderr)

if fo == '' :
    eprint('No output file')
    
# Read in the file list
rec = {}
f = sys.stdin
for line in f :
    line = line.strip()
    [fn, ch, ti] = line.split('\t')
    ti = int(ti)
    if ti not in rec :
        rec[ti] = {}
    rec[ti][ch] = fn

# generate training and prediction list
dataout = []
tis = list(rec.keys())
tis.sort()
tmax = max(tis)
t1 = tis[0]
tpred = t1 + (ns - 1) * s1 + s2

# print(s1, s2, s3)
# print(tis[0], '-', tis[-1])

while t1 < tis[-1] and tpred <= tis[-1] :
    # print(t1, tpred)
    train = []
    pred = []
    t = t1
    n = 0
    # add traning
    while n < ns :
        vec = []
        for c in chs :
            if c in rec[t] :
                vec.append(rec[t][c])
            else :
                vec.append('NA')
                eprint('No time slot:', t, 'channel:', c)
        train.append(vec)
        n += 1
        t += s1
    # add prediction
    for c in chs :
        if c in rec[t] :
            pred.append(rec[t][c])
        else :
            pred.append('NA')
            eprint('No time slot:', t, 'channel:', c)
    # output
    dataout.append([pred, train])
    t1 += s3
    tpred += s3

print(dataout[1][0])
print('----------')
print(dataout[1][1])

# with open
