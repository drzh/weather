#!/usr/bin/python3

# Usage: prog lat_min lat_max lon_min lon_max data_min data_max FLOAT_step_in_lat_lon lat_lon.xz data.xz out.xz

import sys
import getopt
import lzma
import numpy as np
import pandas as pd
import itertools
import pickle

# Parsing parameters
try :
    opts, args = getopt.getopt(sys.argv[1:],
                               'l:d:o:',
                               ['latmin=', 'latmax=', 'lonmin=', 'lonmax=',
                                'dmin=', 'dmax=', 'latlon=', 'data=',
                                'step=', 'scale=', 'out='
                               ]
    )
except getopt.GetoptError as err :
    print(str(err))
    sys.exit(2)

latmin = -90
latmax = 90
lonmin = -180
lonmax = 180
dmin = 0
dmax = 60000
step = 0.1
dscale = 1
flat = ''
fdata = ''
fo = ''

for o, a in opts :
    if o == '--latmin' :
        latmin = float(a)
    elif o == '--latmax' :
        latmax = float(a)
    elif o == '--lonmin' :
        lonmin = float(a)
    elif o == '--lonmax' :
        lonmax = float(a)
    elif o == '--dmin' :
        dmin = float(a)
    elif o == '--dmax' :
        dmax = float(a)
    elif o == '--step' :
        step = float(a)
    elif o == '--scale' :
        dscale = float(a)
    elif o in ('-l', '--latlon') :
        flat = a
    elif o in ('-d', '--data') :
        fdata = a
    elif o in ('-o', '--out') :
        fo = a
    else :
        assert False, 'unhandled option'

if (flat == '' or
    fdata == '' or
    fo == ''
    ) :
    print('No enough input and output files', flat, fdata, fo, file = sys.stderr)
    sys.exit(1)

# read in lon, lat and data
with lzma.open(flat, 'rb') as f :
    lat_lon = np.load(f)
with lzma.open(fdata, 'rb') as f :
    data = np.load(f)

lat = lat_lon[0]
lon = lat_lon[1]

# Re-scale the data
if dscale != 1 and dscale != 0 :
    data = data * dscale

if (len(lat.shape) != len(lon.shape) or
    lat.shape != lon.shape or
    len(lat.shape) != len(data.shape) or
    lat.shape != data.shape
) :
    print(fdata, 'lat, lon and data are not in the same length', file = sys.stderr)
    sys.exit(1)

dataall = pd.DataFrame(zip(lat, lon, data), columns = ['lat', 'lon', 'data'])
dataall = dataall[(dataall['lat'] >= latmin) & (dataall['lat'] < latmax) &
                  (dataall['lon'] >= lonmin) & (dataall['lon'] < lonmax) &
                  (dataall['data'] >= dmin) & (dataall['data'] < dmax)]
dataall['lat'] = np.floor(dataall['lat'] / step) * step
dataall['lat'] = [round(x, 1) for x in dataall['lat']]
dataall['lon'] = np.floor(dataall['lon'] / step) * step
dataall['lon'] = [round(x, 1) for x in dataall['lon']]

datamean = pd.DataFrame(dataall.groupby(['lat', 'lon'])['data'].mean().reset_index())
    
latbs = [round(x, 1) for x in np.arange(latmin, latmax, step)]
lonbs = [round(x, 1) for x in np.arange(lonmin, lonmax, step)]
latlon = pd.DataFrame(list(itertools.product(latbs, lonbs)), columns = ['lat', 'lon'])

if latlon.shape[0] == datamean.shape[0] :
    # Merge latlon and datamean to ensure the datamean was ordered by latlon
    datamean = pd.merge(latlon, datamean, on=['lat', 'lon'])
    if latlon.shape[0] == datamean.shape[0] :
        vec = np.array(datamean['data'], dtype=np.uint16)
        dataout = [latbs, lonbs, vec]
        with lzma.open(fo, 'wb') as f :
            pickle.dump(dataout, f)
    else :
        print(fdata, 'is NA ', datamean.shape, file = sys.stderr)
else :
    print(fdata, 'is NA ', datamean.shape, file = sys.stderr)
