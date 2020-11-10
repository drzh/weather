#!/usr/bin/python3

import sys
import getopt
import lzma
import numpy as np
# import pandas as pd
# import itertools
import pickle

# Parsing parameters
try:
    opts, args = getopt.getopt(sys.argv[1:],
                               'i:o:',
                               ['latmin=', 'latmax=', 'lonmin=', 'lonmax=',
                                'dmin=', 'dmax=', 'latlon=', 'input=',
                                'step=', 'scale=', 'out=', 'list='
                               ]
    )
except getopt.GetoptError as err:
    print(str(err), file = sys.stderr)
    sys.exit(2)

latmin = -90
latmax = 90
lonmin = -180
lonmax = 180
dmin = -10000
dmax = 60000
step = 0.1
dscale = 1
fi = '/dev/stdin'
fo = ''

for o, a in opts:
    if o == '--latmin':
        latmin = float(a)
    elif o == '--latmax':
        latmax = float(a)
    elif o == '--lonmin':
        lonmin = float(a)
    elif o == '--lonmax':
        lonmax = float(a)
    elif o == '--dmin':
        dmin = float(a)
    elif o == '--dmax':
        dmax = float(a)
    elif o == '--step':
        step = float(a)
    elif o == '--scale':
        dscale = float(a)
    elif o in ('-i', '--input'):
        fdata = a
    elif o in ('-o', '--out'):
        fo = a
    else:
        assert False, 'unhandled option'

if fo == '':
    print('No output file', file = sys.stderr)
    sys.exit(1)

latbs = [round(x, 1) for x in np.arange(latmin, latmax, step)]
lonbs = [round(x, 1) for x in np.arange(lonmin, lonmax, step)]
dsum = {}
dn = {}
# Initiate dsum and dn
for lat in latbs:
    dsum[lat] = {}
    dn[lat] = {}
    for lon in lonbs:
        dsum[lat][lon] = 0
        dn[lat][lon] = 0
        
with open(fi, 'r') as f:
    for line in f:
        lat, lon, elev = line.strip().split('\t')
        lat = np.round(float(lat), 1)
        lon = np.round(float(lon), 1)
        elev = float(elev)
        if lat >= latmin and lat < latmax and lon >= lonmin and lon < lonmax:
            # Re-scale the data
            elev = elev * dscale
            
            dsum[lat][lon] += elev
            dn[lat][lon] += 1

dout = []
for lat in latbs:
    for lon in lonbs:
        dout.append(dsum[lat][lon] / dn[lat][lon] if dn[lat][lon] > 0 else 0)

dout = np.array(dout, dtype = np.float32)
with lzma.open(fo, 'wb') as f:
    pickle.dump([latbs, lonbs, dout], f)
