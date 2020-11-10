#!/usr/bin/python3

import sys
import getopt
import lzma
import numpy as np
import pandas as pd
import itertools
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

with open(fi, 'r') as f:
    dataall = pd.read_csv(f, sep='\t', header=None)
    dataall.columns = ['lat', 'lon', 'data']
    dataall['lat'] = np.floor(dataall['lat'] / step) * step
    dataall['lat'] = np.round(dataall['lat'])
    dataall['lon'] = np.floor(dataall['lon'] / step) * step
    dataall['lon'] = np.round(dataall['lon'])
    datamean = pd.DataFrame(dataall.groupby(['lat', 'lon'])['data'].mean().reset_index())
    latbs = [round(x, 1) for x in np.arange(latmin, latmax, step)]
    lonbs = [round(x, 1) for x in np.arange(lonmin, lonmax, step)]
    latlon = pd.DataFrame(list(itertools.product(latbs, lonbs)), columns = ['lat', 'lon'])

    # Merge latlon and datamean to ensure the datamean was ordered by latlon
    datamean = pd.merge(latlon, datamean, on=['lat', 'lon'], how='left')
    datamean = datamean.fillna(0)
    if latlon.shape[0] == datamean.shape[0] :
        vec = np.array(datamean['data'], dtype=np.float32)
        dataout = [latbs, lonbs, vec]
        with lzma.open(fo, 'wb') as f :
            pickle.dump(dataout, f)
    else :
        print(fi, 'is NA ', datamean.shape, file = sys.stderr)
