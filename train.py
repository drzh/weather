#!/usr/bin/python3

# Usage: porg -m FLOAT_mean -s FLOAT_sd -l FILE_list -o FILE_out.xz

import sys
import numpy as np
import lzma
import pickle
import getopt
from readdata import readdata

# Function to print error
def eprint(msg):
    print(msg, file = sys.stderr)

# Parsing parameters
try :
    opts, args = getopt.getopt(sys.argv[1:],
                               'm:s:l:o:',
                               ['mean=', 'sd=', 'list=', 'out='
                               ]
    )
except getopt.GetoptError as err :
    eprint(str(err))
    sys.exit(2)

mean = 0
sd = 1
fl = '/dev/stdin'
fo = ''

for o, a in opts:
    if o in ('-m', '--mean'):
        mean = float(a)
    elif o in ('-s', '--sd'):
        sd = float(a)
    elif o in ('-o', '--out'):
        fo = a
    elif o in ('-o', '--out'):
        fo = a
    else:
        assert False, 'unhandled option'

# # Check fo
# if fo == '':
#     eprint('No output file')
#     sys.exit(1)

dout = []
for d in readdata(fl, mean, sd):
    
    sys.exit(0)
