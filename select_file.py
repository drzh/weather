#!/home2/s167968/bin/python3

import sys
import datetime

_, fo = sys.argv

# Number of files contained for each time slot
nt = 10

rec = {}
cnt = {}
for line in sys.stdin:
    line = line.strip()
    e = line.split('\t')
    e[3] = int(e[3])
    if e[3] not in rec:
        rec[e[3]] = []
        cnt[e[3]] = 0
    rec[e[3]].append(e[0])
    cnt[e[3]] += 1

def output(fo, x):
    with open(fo, 'w') as f:
        for e in x:
            print('\t'.join(e), file=f)
            
k0 = -1
out = []
i = 0;
for k in sorted(rec.keys()):
    # print(k, out)
    if cnt[k] == nt:
        if k0 > 0:
            if k - k0 < 270 or k - k0 > 330:
                outf = fo + '.' + str(i)
                output(outf, out)
                out = []
                i += 1
        out.append(rec[k])
        k0 = k

outf = fo + '.' + str(i)
output(outf, out)
