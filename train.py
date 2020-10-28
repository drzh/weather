#!/usr/bin/python3

# Usage: porg -m FLOAT_mean -s FLOAT_sd -l FILE_list -o FILE_out.xz

import sys
import numpy as np
import lzma
import pickle
import getopt
import torch
import torch.nn as nn
import torch.nn.functional as F
from readdata import readdata
from convlstm import ConvLSTM
from axial_attention import AxialAttention

# Function to print error
def eprint(msg):
    print(msg, file = sys.stderr)

# Parsing parameters
try :
    opts, args = getopt.getopt(sys.argv[1:],
                               'm:s:l:o:',
                               ['mean=', 'sd=', 'list=', 'out='
                               ]
    )
except getopt.GetoptError as err :
    eprint(str(err))
    sys.exit(2)

mean = 0
sd = 1
fl = '/dev/stdin'
fo = ''

for o, a in opts:
    if o in ('-m', '--mean'):
        mean = float(a)
    elif o in ('-s', '--sd'):
        sd = float(a)
    elif o in ('-l', '--list'):
        fl = a
    elif o in ('-o', '--out'):
        fo = a
    else:
        assert False, 'unhandled option'

# Define a model
class WLSTM(nn.Module):
    def __init__(self,
                 input_dim = 10, hidden_dim = 64, kernel_size = (3, 3),
                 num_layers = 1, batch_first = True, bias = True,
                 return_all_layers = False):

        super(WLSTM, self).__init__()

        # Conv and pooling model
        self.conv1 = nn.Conv2d(input_dim, 160, 3, padding = (1, 1))
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(160, 256, 3, padding = (1, 1))
        self.conv3 = nn.Conv2d(256, 256, 3, padding = (1, 1))
        self.conv4 = nn.Conv2d(256, 256, 3, padding = (1, 1))
        self.pool2 = nn.MaxPool2d(2, 2)

        # ConvLSTM model
        self.convlstm = ConvLSTM(input_dim = 256,
                                 hidden_dim = 384,
                                 kernel_size = (3, 3),
                                 num_layers = 1,
                                 batch_first = True,
                                 bias = True,
                                 return_all_layers = False)
    """
    Input: (b, t, c, h, w)
    """
    def forward(self, x):
        xb = []
        for b in x:
            # Convlution and pooling to get 128x64x64 (c x h x w)
            y = b
            y = F.relu(self.conv1(y))
            y = F.max_pool2d(y, 2)
            y = F.relu(self.conv2(y))
            y = F.relu(self.conv3(y))
            y = F.relu(self.conv4(y))
            y = F.max_pool2d(y, 2)
            xb.append(y)

        # b x t x 128 x 64 x 64 to get through ConvLSTM
        xb = torch.stack(xb, dim = 0)
        res = self.convlstm(xb)
        return res

# Create a model
model = WLSTM()

# Create loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

for xtrain, ytrain in readdata(fl, mean, sd):
    xtrain, ytrain = torch.Tensor([xtrain]), torch.Tensor(ytrain)
    optimizer.zero_grad()
    layer_output, last_state = model(xtrain)
    ypred = last_state[0][0]
    print(ypred.size())
    print(ytrain.size())
    # print(layer_output[0].size())
    # print('---------')
    # print(last_state[0][0].size())
    # print('---------')
    # print(last_state[0][1].size())
    sys.exit(0)
