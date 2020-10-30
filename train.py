#!/usr/bin/python3

# Usage: porg -m FLOAT_mean -s FLOAT_sd -l FILE_list -o FILE_out_prefix

import sys
import numpy as np
import lzma
import pickle
import getopt
import torch
import torch.nn as nn
import torch.nn.functional as F
from datetime import datetime
from readdata import readdata
from wlstm import WLSTM

# Function to print error
def eprint(msg):
    print(msg, file = sys.stderr)

# Parsing parameters
try :
    opts, args = getopt.getopt(sys.argv[1:],
                               'm:s:l:o:n:',
                               ['mean=', 'sd=', 'list=', 'out=',
                                'lr=', 'model=', 'dev='
                               ]
    )
except getopt.GetoptError as err :
    eprint(str(err))
    sys.exit(2)

mean = 0
sd = 1
lr = 0.001
n = 1000
dev = 'cuda:0'
fm = ''
fl = '/dev/stdin'
fo = ''

for o, a in opts:
    if o in ('-m', '--mean'):
        mean = float(a)
    elif o in ('-s', '--sd'):
        sd = float(a)
    elif o in ('--lr'):
        lr = float(a)
    elif o in ('-n'):
        n = int(a)
    elif o in ('--dev'):
        dev = a
    elif o in ('--model'):
        fm = a
    elif o in ('-l', '--list'):
        fl = a
    elif o in ('-o', '--out'):
        fo = a
    else:
        assert False, 'unhandled option'

# Prepare GPU
device = torch.device(dev if torch.cuda.is_available() else "cpu")
print('Using', device, file = sys.stderr)

# Function to scale temperature (181 ~ 330) to group (0 ~ 149)
def scale_group(x):
    x = torch.round(x / 100 - 181)
    if x < 0:
        x = 0
    elif x > 149:
        x = 149
    return x

model = WLSTM()

# Create a model
if fm != '':
    model.load_state_dict(torch.load(fm))
    model.eval()

# Send data to GPU
model.to(device)

# Create loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr = lr)

# Train the model
i = 1
# for xtrain, ytrain in readdata(fl, mean, sd):
for xtrain, ytrain in readdata(fl):
    # Scale the data
    xtrain = np.asarray(xtrain)
    if mean != 0:
        xtrain = xtrain - mean
    if sd > 0 and sd != 1:
        xtrain = xtrain / sd

    xtrain = torch.Tensor([xtrain]).to(device)
    ytrain = torch.Tensor(ytrain)
    
    optimizer.zero_grad()
    ypred = model(xtrain)

    # sys.exit(0)

    # Reshape ypred
    ypred = torch.flatten(ypred, 2)
    ypred = torch.transpose(ypred, 1, 2)
    ypred = torch.flatten(ypred, 0, 1)

    # Reshape ytrain
    ytrain = torch.flatten(ytrain, 1)
    ytrain = torch.flatten(ytrain, 0, 1)

    # Label ytrain from (181 ~ 330) to group (0 ~ 149)
    ylabel = torch.Tensor([scale_group(x) for x in ytrain]).long()
    ylabel = ylabel.to(device)
    
    loss = criterion(ypred, ylabel)
    loss.backward()
    optimizer.step()

    # print(loss)
    # print(ytrain)
    # print(ylabel)
    # print(ypred)
    # sys.exit(0)

    # Print epoch information
    print(datetime.now().strftime('%H:%M:%S'), 'epoch', i, 'loss:', '%.8f' % loss.item())
    sys.stdout.flush()
    
    # Export model state dict every n steps
    if i % n == 0:
        fout = fo + '.state_dict.' + str(i) + '.pt'
        torch.save(model.state_dict(), fout)
    
    i += 1
    
# Export model state dict in final step
fout = fo + '.state_dict.' + str(i) + '.pt'
torch.save(model.state_dict(), fout)
