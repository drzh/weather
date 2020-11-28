#!/usr/bin/python3

import sys
import lzma
import pickle
import numpy as np

_, fi, folat, folon = sys.argv

with lzma.open(fi, 'rb') as f :
    data = pickle.load(f)

lat = data[0]
lon = data[1]

dlat = len(lat)
dlon = len(lon)

latmat = np.squeeze(np.repeat([lat], dlon, axis = 1))
lonmat = np.squeeze(np.repeat([lon], dlat, axis = 0).reshape(1, -1))
latmat = np.array(latmat * 100, dtype = np.int)
lonmat = np.array(lonmat * 100, dtype = np.int)
lonmat = np.negative(lonmat)

with lzma.open(folat, 'wb') as f:
    pickle.dump([lat, lon, latmat], f)

with lzma.open(folon, 'wb') as f:
    pickle.dump([lat, lon, lonmat], f)
