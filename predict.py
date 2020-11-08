#!/usr/bin/python3

# Usage: porg -m FLOAT_mean -s FLOAT_sd -b INT_minibatch_size -l FILE_list -o FILE_out_prefix

import sys
import numpy as np
import lzma
import pickle
import getopt
import torch
from readdata import readdata
from wlstm import WLSTM

# Function to print error
def eprint(*argv):
    print(*argv, file = sys.stderr)

# Initiate default parameters
param = {
    'mean': 0,
    'sd': 1,
    'th': 1,
    'dev': 'cpu',
    'fm': '',
    'fl': '/dev/stdin',
    'fo': '',
}

# Parsing parameters
try :
    opts, args = getopt.getopt(sys.argv[1:],
                               't:',
                               ['mean=', 'sd=', 'list=', 'out=',
                                'model=', 'dev=', 'threads=',
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
    elif o in ('-t', '--threads'):
        param['th'] = int(a)
    elif o in ('--dev'):
        param['dev'] = a
    elif o in ('--model'):
        param['fm'] = a
    elif o in ('--list'):
        param['fl'] = a
    elif o in ('--out'):
        param['fo'] = a
    else:
        assert False, 'unhandled option'

# Prepare GPU
device = torch.device(param['dev'] if torch.cuda.is_available() else "cpu")

# Create a model
model = WLSTM(input_dim = 12)

# Send data to GPU
model.to(device)

# Read a saved model
if param['fm'] != '':
    checkpoint = torch.load(param['fm'], map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

# Predict
for din in readdata(param['fl']):
    for xtrain in din:
        # Scale the data
        xtrain = np.asarray(xtrain)
        xtrain = xtrain.astype(float)
        if param['mean'] != 0:
            xtrain = xtrain - param['mean']
        if param['sd'] > 0 and param['sd'] != 1:
            xtrain = xtrain / param['sd']
            
        xtrain = torch.Tensor([xtrain]).to(device)
        ypred = model(xtrain)
        ypred = ypred.detach().numpy()

        if param['fo'] != '':
            with lzma.open(param['fo'], 'wb') as f:
                pickle.dump(ypred, f)

        print(ypred)
        print('min:', np.amin(ypred))
        print('max:', np.amax(ypred))
        print('mean:', np.mean(ypred))
