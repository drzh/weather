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
                               'm:s:l:o:n:',
                               ['mean=', 'sd=', 'list=', 'out=',
                                'lr=', 'model=', 'dev=', 'ln=',
                                'ld='
                               ]
    )
except getopt.GetoptError as err :
    eprint(str(err))
    sys.exit(2)

mean = 0
sd = 1
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

# # Function to scale temperature (181 ~ 330) to group (0 ~ 149)
# def scale_group(x):
#     x = torch.round(x / 100 - 181)
#     if x < 0:
#         x = 0
#     elif x > 149:
#         x = 149
#     return x

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
i = 1
# for xtrain, ytrain in readdata(fl, mean, sd):
for xtrain, ytrain in readdata(fl):
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
    
    optimizer.zero_grad()
    ypred = model(xtrain)

    # print(ypred.size())
    # print(ytrain.size())
    # sys.exit(0)

    # Reshape ypred
    # ypred = torch.flatten(ypred, 2)
    # ypred = torch.transpose(ypred, 1, 2)
    # ypred = torch.flatten(ypred, 0, 1)
    ypred = torch.flatten(ypred)

    # Clamp the min and max value for ypred
    # ypred = torch.clamp(ypred, 19000, 32000)

    # Reshape ytrain
    # ytrain = torch.flatten(ytrain, 1)
    # ytrain = torch.flatten(ytrain, 0, 1)
    ytrain = torch.flatten(ytrain)

    # # Label ytrain from (181 ~ 330) to group (0 ~ 149)
    # ylabel = torch.Tensor([scale_group(x) for x in ytrain]).long()
    # ylabel = ylabel.to(device)
    
    # loss = criterion(ypred, ylabel)
    loss = criterion(ypred, ytrain)
    loss.backward()
    optimizer.step()

    # Print epoch information
    print(datetime.now().strftime('%H:%M:%S'), 'epoch', i, 'loss:', '%.8f' % loss.item())
    sys.stdout.flush()

    # # Update learning rate
    # if ln > 0 and i % ln == 0:
    #     lrt = lrt * ld
    #     optimizer = torch.optim.Adam(model.parameters(), lr = lrt)
    
    # Decays the learning rate
    scheduler.step()
    
    # Export model state dict every n steps
    if i % n == 0:
        fout = fo + '.state_dict.' + str(i) + '.pt'
        torch.save(model.state_dict(), fout)
        # Print learning rate
        for param_group in optimizer.param_groups:
            print('lr =', param_group['lr'])

    i += 1
    
# Export model state dict in final step
fout = fo + '.state_dict.' + str(i) + '.pt'
torch.save(model.state_dict(), fout)
