#!/usr/bin/python3

# Usage: <STDIN_file_list> | prog FLOAT_start FLOAT_end FLOAT_step | <STDOUT>
# Usage: Calculate the statistics

import sys
import lzma
import pickle
import numpy as np
from collections import Counter

_, start, end, step = sys.argv

start = int(start)
end = int(end)
step = int(step)

r = list(range(start, end, step))

print('id', '\t'.join([str(x) for x in r]), sep='\t')

for line in sys.stdin :
    fn = line.strip()
    with lzma.open(fn, 'rb') as f :
        data = pickle.load(f)
        if len(data) >= 3 :
            tem = data[2]
            tem = tem / 100
            tem = tem.astype(int)
            cnt = Counter(tem)
            print(fn, end='')
            for n in r:
                print('\t', cnt[n] if n in cnt else 0, sep='', end='')
            print()
