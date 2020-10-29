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

# Function to scale temperature (181 ~ 330) to group (0 ~ 149)
def scale_group(x):
    x = torch.round(x / 100 - 181)
    if x < 0:
        x = 0
    elif x > 149:
        x = 149
    return x

# Define a model
class WLSTM(nn.Module):
    def __init__(self,
                 input_dim = 10, hidden_dim = 64, kernel_size = (3, 3),
                 num_layers = 1, batch_first = True, bias = True,
                 return_all_layers = False):

        super(WLSTM, self).__init__()

        # Conv and pooling model
        self.conv1 = nn.Conv2d(input_dim, 160, 3, padding = (1, 1))
        self.conv2 = nn.Conv2d(160, 256, 3, padding = (1, 1))
        self.conv3 = nn.Conv2d(256, 256, 3, padding = (1, 1))
        self.conv4 = nn.Conv2d(256, 256, 3, padding = (1, 1))
        self.conv5 = nn.Conv2d(384, 2048, 1)
        self.conv6 = nn.Conv2d(2048, 150, 1)

        # ConvLSTM model
        self.convlstm = ConvLSTM(input_dim = 256,
                                 hidden_dim = 384,
                                 kernel_size = (3, 3),
                                 num_layers = 1,
                                 batch_first = True,
                                 bias = True,
                                 return_all_layers = False)

        # Axial self-attention model
        self.attn1 = AxialAttention(dim = 64,
                                    dim_index = 2,
                                    dim_heads = 16,
                                    num_dimensions = 2,
                                    sum_axial_out = True)
        
        self.attn2 = AxialAttention(dim = 64,
                                    dim_index = 3,
                                    dim_heads = 16,
                                    num_dimensions = 2,
                                    sum_axial_out = True)
        
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
        layer_output, last_state = self.convlstm(xb)
        h = last_state[0][0]

        # convert b x 384 x 64 x 64 to b x 384 x 64 x 64
        y = F.relu(self.conv5(h))
                                    
        # Axial self attension
        y = self.attn1(y)
        y = self.attn2(y)
        y = self.attn1(y)
        y = self.attn2(y)
        y = self.attn1(y)
        y = self.attn2(y)
        y = self.attn1(y)
        y = self.attn2(y)

        # Convert to 150 x 64 x 64 to represent probability for each unit
        y = self.conv6(y)
        
        return y

# Create a model
model = WLSTM()

# Create loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

i = 0
for xtrain, ytrain in readdata(fl, mean, sd):
    xtrain, ytrain = torch.Tensor([xtrain]), torch.Tensor(ytrain)
    optimizer.zero_grad()
    ypred = model(xtrain)

    # Reshape ypred
    ypred = torch.flatten(ypred, 2)
    ypred = torch.transpose(ypred, 1, 2)
    ypred = torch.flatten(ypred, 0, 1)

    # Reshape ytrain
    ytrain = torch.flatten(ytrain, 1)
    ytrain = torch.flatten(ytrain, 0, 1)

    # Label ytrain from (181 ~ 330) to group (0 ~ 149)
    ylabel = torch.Tensor([scale_group(x) for x in ytrain]).long()

    loss = criterion(ypred, ylabel)
    loss.backward()
    optimizer.step()
    print('echoch', i, 'loss:', loss.item())
    i += 1

    # print(ypred.size())
    # print(ytrain.size())
    # print(layer_output[0].size())
    # print('---------')
    # print(last_state[0][0].size())
    # print('---------')
    # print(last_state[0][1].size())
    # sys.exit(0)
