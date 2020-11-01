#!/usr/bin/python3

# Usage: porg -m FLOAT_mean -s FLOAT_sd -b INT_minibatch_size -l FILE_list -o FILE_out_prefix

import sys
import numpy as np
import lzma
import pickle
import getopt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR
from datetime import datetime
from readdata import readdata
from wlstm import WLSTM

# Function to print error
def eprint(msg):
    print(msg, file = sys.stderr)

# Parsing parameters
try :
    opts, args = getopt.getopt(sys.argv[1:],
                               'm:s:l:o:n:b:e:t:',
                               ['mean=', 'sd=', 'list=', 'out=',
                                'lr=', 'model=', 'dev=', 'ln=',
                                'ld=', 'batch=', 'epoch=', 'threads='
                               ]
    )
except getopt.GetoptError as err :
    eprint(str(err))
    sys.exit(2)

mean = 0
sd = 1
bat = 1    # number of samples in each minibatch
epo = 1    # number of epoch for each minibatch
th = 1
lrt = 0.001
ld = 0.5
ln = 0
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
    elif o in ('-b', '--batch'):
        bat = int(a)
    elif o in ('-t', '--threads'):
        th = int(a)
    elif o in ('-3', '--epoch'):
        epo = int(a)
    elif o in ('--lr'):
        lrt = float(a)
    elif o in ('--ln'):
        ln = int(a)
    elif o in ('--ld'):
        ld = float(a)
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
# print('Using', device, file = sys.stderr)

# Function to scale temperature (200 ~ 299K) to group (0 ~ 19)
def scale_group(x):
    x = torch.round((x / 100 - 200) / 5)
    if x < 0:
        x = 0
    elif x > 19:
        x = 19
    return x

model = WLSTM()

# Create a model
if fm != '':
    model.load_state_dict(torch.load(fm))
    model.eval()

# Send data to GPU
model.to(device)

# Create loss function and optimizer
# criterion = nn.CrossEntropyLoss()
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr = lrt)
# optimizer = torch.optim.SGD(model.parameters(), lr=lrt, momentum=0.9)

# Decays the learning rate according to epoch
scheduler = StepLR(optimizer, step_size = 1, gamma = ld)

# Train the model
i = 0
ibat = 0
istep = 0
loss_sum = 0
for din in readdata(fl, bat):
    ibat += 1
    for ep in range(epo):
        for xtrain, ytrain in din:
            i += 1
            # Scale the data
            xtrain = np.asarray(xtrain)
            ytrain = np.asarray(ytrain)
            xtrain = xtrain.astype(float)
            ytrain = ytrain.astype(float)
            if mean != 0:
                xtrain = xtrain - mean
                ytrain = ytrain -mean
            if sd > 0 and sd != 1:
                xtrain = xtrain / sd
                ytrain = ytrain / sd
            
            xtrain = torch.Tensor([xtrain]).to(device)
            ytrain = torch.Tensor(ytrain).to(device)
        
            ypred = model(xtrain)

            # Reshape ypred
            ypred = torch.flatten(ypred)

            # Reshape ytrain
            ytrain = torch.flatten(ytrain)

            loss = criterion(ypred, ytrain) / bat
            loss.backward()
            loss_sum += loss

        # Process the epoch of minibatch
        optimizer.step()
        optimizer.zero_grad()

        # Print epoch and loss information
        print(datetime.now().strftime('%H:%M:%S'), ' input=', i, ' batch=', ibat, ' epoch=', ep + 1, ' lr=', optimizer.param_groups[0]['lr'], ' loss=', '%.8f' % loss_sum, sep='')
        sys.stdout.flush()
        loss_sum = 0

        # Export model state dict every n inputs
        if i % n == 0:
            fout = fo + '.state_dict.b_' + str(ibat) + '.e_' + str(ep + 1) + '.pt'
            torch.save(model.state_dict(), fout)

        # Decay the learning rate
        if ln > 0 and i - istep >= ln:
            scheduler.step()
            istep = i

