#!/home2/s167968/bin/python3

import sys
import datetime

_, fo = sys.argv

# Number of files contained for each time slot
nt = 10

rec = {}
cnt = {}
for line in sys.stdin :
    line = line.strip()
    e = line.split('\t')
    e[5] = int(e[5])
    if e[5] not in rec :
        rec[e[5]] = []
        cnt[e[5]] = 0
    rec[e[5]].append(line)
    cnt[e[5]] += 1

def output(fo, x) :
    with open(fo, 'w') as f :
        i = 0
        for e1 in x :
            i += 1
            for e2 in e1 :
                print(e2, i, sep='\t', file=f)
    f.close()
            
k0 = -1
out = []
i = 0;
for k in rec :
    # print(k, out)
    if cnt[k] == nt :
        if k0 > 0 :
            if k - k0 < 270 or k - k0 > 330 :
                outf = fo + '.' + str(i)
                output(outf, out)
                out = []
                i += 1
        out.append(rec[k])
        k0 = k

outf = fo + '.' + str(i)
output(outf, out)
