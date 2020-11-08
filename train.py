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
from torch.optim.lr_scheduler import ReduceLROnPlateau
from datetime import datetime
from readdata import readdata
from wlstm import WLSTM

# Function to print error
def eprint(*argv):
    print(*argv, file = sys.stderr)

# Initiate default parameters
param = {
    'mean': 0,
    'sd': 1,
    'bat': 1,    # number of samples in each minibatch
    'epo': 1,    # number of epoch for each minibatch
    'th': 1,
    'lrt': 0.001,    # initial learning rate
    'ld': 0.5,     # decay rate of learning
    'ln': 0,    # number of inputs to execute a decay
    'lp': 0,    # number of patience before decay
    'opt': 'adam',    # optimizer
    'mmt': 1,    # momentum for SGD
    'n': 1000,    # number of inputs to output model
    'dev': 'cuda:0',
    'recv': 0,    # Recover the learning rate (1) or use new learning rate (0)
    'fm': '',
    'fl': '/dev/stdin',
    'fo': '',
}

# Parsing parameters
try :
    opts, args = getopt.getopt(sys.argv[1:],
                               'n:b:t:r',
                               ['mean=', 'sd=', 'list=', 'out=',
                                'lr=', 'model=', 'dev=', 'ln=',
                                'ld=', 'batch=', 'epoch=', 'threads=',
                                'recover', 'lp=', 'opt=', 'mmt='
                               ]
    )
except getopt.GetoptError as err :
    eprint(str(err))
    sys.exit(2)

for o, a in opts:
    if o in ('--mean'):
        param['mean'] = float(a)
    elif o in ('--sd'):
        param['sd'] = float(a)
    elif o in ('-b', '--batch'):
        param['bat'] = int(a)
    elif o in ('-t', '--threads'):
        param['th'] = int(a)
    elif o in ('--epoch'):
        param['epo'] = int(a)
    elif o in ('--lr'):
        param['lrt'] = float(a)
    elif o in ('--ld'):
        param['ld'] = float(a)
    elif o in ('--ln'):
        param['ln'] = int(a)
    elif o in ('--lp'):
        param['lp'] = int(a)
    elif o in ('--opt'):
        param['opt'] = a
    elif o in ('--mmt'):
        param['mmt'] = float(a)
    elif o in ('-n'):
        param['n'] = int(a)
    elif o in ('--dev'):
        param['dev'] = a
    elif o in ('-r', '--recover'):
        param['recv'] = 1
    elif o in ('--model'):
        param['fm'] = a
    elif o in ('--list'):
        param['fl'] = a
    elif o in ('--out'):
        param['fo'] = a
    else:
        assert False, 'unhandled option'

# Check parameters
# Confirm no more than one schedualer was set
if param['ln'] > 0 and param['lp'] > 0:
    eprint('Selected two schedualer')
    sys.exit(2)
# check optimizer
if param['opt'] not in ('adam', 'sgd'):
    eprint('Optimizer:', param['opt'])
    eprint('Optimizer must be: adam, sgd')
    sys.exit(2)
    
# Prepare GPU
device = torch.device(param['dev'] if torch.cuda.is_available() else "cpu")

# Function to scale temperature (200 ~ 299K) to group (0 ~ 19)
def scale_group(x):
    x = torch.round((x / 100 - 200) / 5)
    if x < 0:
        x = 0
    elif x > 19:
        x = 19
    return x

# Create a model
model = WLSTM(input_dim = 12)

# Send data to GPU
model.to(device)

# Creater optimizer
if param['opt'] == 'adam':
    optimizer = torch.optim.Adam(model.parameters(), lr = param['lrt'])
elif param['opt'] == 'sgd':
    optimizer = torch.optim.SGD(model.parameters(), lr = param['lrt'],
                                momentum = param['mmt'], nesterov = True)
else:
    eprint('Optimizer:', param['opt'])
    eprint('Optimizer must be: adam, sgd')
    sys.exit(2)

# Create loss function and optimizer
criterion = nn.MSELoss()

# Read a saved model
if param['fm'] != '':
    checkpoint = torch.load(param['fm'])
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    
    if param['recv'] == 0:
        # Set new learning rate
        for g in optimizer.param_groups:
            g['lr'] = param['lrt']

    model.eval()

# Set the schedualer to decays the learning rate
if param['ln'] > 0:
    scheduler = StepLR(optimizer, step_size = 1, gamma = param['ld'])
elif param['lp'] > 0:
    scheduler = ReduceLROnPlateau(optimizer, factor = param['ld'], patience = param['lp'])

# Train the model
i = 0
ibat = 0
istep = 0
loss_sum = 0
for din in readdata(param['fl'], param['bat']):
    ibat += 1
    for ep in range(1, param['epo'] + 1):
        for xtrain, ytrain in din:
            i += 1
            # Scale the data
            xtrain = np.asarray(xtrain)
            ytrain = np.asarray(ytrain)
            xtrain = xtrain.astype(float)
            ytrain = ytrain.astype(float)
            if param['mean'] != 0:
                xtrain = xtrain - param['mean']
                ytrain = ytrain - param['mean']
            if param['sd'] > 0 and param['sd'] != 1:
                xtrain = xtrain / param['sd']
                ytrain = ytrain / param['sd']
            
            xtrain = torch.Tensor([xtrain]).to(device)
            ytrain = torch.Tensor(ytrain).to(device)
        
            ypred = model(xtrain)

            # # Reshape ypred
            # ypred = torch.flatten(ypred)

            # Reshape ytrain
            # ytrain = torch.flatten(ytrain)
            ytrain = ytrain.view(-1)

            loss = criterion(ypred, ytrain) / param['bat']
            loss.backward()
            loss_sum += loss

        # Process the epoch of minibatch
        optimizer.step()
        optimizer.zero_grad()

        # Print epoch and loss information
        print(datetime.now().strftime('%H:%M:%S'), ' input=', i, ' batch=', ibat, ' epoch=', ep, ' lr=', optimizer.param_groups[0]['lr'], ' loss=', '%.8f' % loss_sum, sep='')
        sys.stdout.flush()

        # Export checkpoint every n inputs
        if i % param['n'] == 0:
            fout = param['fo'] + '.b_' + str(ibat) + '.e_' + str(ep) + '.pt'
            torch.save({
                'i': i,
                'bat': ibat,
                'epoch': ep,
                'istep': istep,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss
            }, fout
            )

        # Decay the learning rate
        if param['ln'] > 0:
            if i - istep >= param['ln']:
                scheduler.step()
                istep = i
        elif param['lp'] > 0:
            scheduler.step(loss_sum)

        # Reset loss_sum
        loss_sum = 0
