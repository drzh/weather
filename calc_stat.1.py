#!/usr/bin/python3

# Usage: <STDIN_file_list> | prog | <STDOUT>
# Usage: Calculate the statistics

import sys
import lzma
import pickle
import numpy as np

for line in sys.stdin :
    fn = line.strip()
    with lzma.open(fn, 'rb') as f :
        data = pickle.load(f)
        if len(data) >= 3 :
            tem = data[2]
            print(fn,
                  int(round(np.mean(tem))),
                  int(round(np.std(tem))),
                  np.amin(tem),
                  np.amax(tem),
                  sep = '\t'
            )
