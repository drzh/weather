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
                               'm:s:l:o:n:b:e:',
                               ['mean=', 'sd=', 'list=', 'out=',
                                'lr=', 'model=', 'dev=', 'ln=',
                                'ld=', 'batch=', 'epoch='
                               ]
    )
except getopt.GetoptError as err :
    eprint(str(err))
    sys.exit(2)

mean = 0
sd = 1
bat = 1    # number of samples in each minibatch
epo = 1    # number of epoch for each minibatch
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
print('Using', device, file = sys.stderr)

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
scheduler = StepLR(optimizer, step_size = ln, gamma = ld)

# Train the model
i = 0
ibat = 0
loss_sum = 0
# for xtrain, ytrain in readdata(fl, mean, sd):
for din in readdata(fl, bat):
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

        # Process minibatch
        if i % bat == 0:
            ibat += 1
            optimizer.step()
            optimizer.zero_grad()
            print(datetime.now().strftime('%H:%M:%S'), ' input=', i, ' batch=', ibat, ' loss=', '%.8f' % loss_sum, sep='')
            sys.stdout.flush()
            loss_sum = 0
            # Export model state dict every n minibatchs
            if ibat % n == 0:
                fout = fo + '.state_dict.' + str(ibat) + '.pt'
                torch.save(model.state_dict(), fout)
                # Print learning rate
                if ln > 0:
                    for param_group in optimizer.param_groups:
                        print('lr =', param_group['lr'])

        # Decays the learning rate
        if ln > 0:
            scheduler.step()


