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
from datetime import datetime
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

# # Function to scale temperature (223 ~ 322) to group (0 ~ 99)
# def scale_group(x):
#     x = torch.round(x / 100 - 223)
#     if x < 0:
#         x = 0
#     elif x > 99:
#         x = 99
#     return x

# Function to scale temperature (181 ~ 330) to group (0 ~ 149)
def scale_group(x):
    x = torch.round(x / 100 - 181)
    if x < 0:
        x = 0
    elif x > 149:
        x = 149
    return x

# # Function to scale temperature (181 ~ 330) to group (0 ~ 49)
# def scale_group(x):
#     x = torch.round((x / 100 - 181) / 3)
#     if x < 0:
#         x = 0
#     elif x > 49:
#         x = 49
#     return x

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

        # self.conv1 = nn.Conv2d(input_dim, 128, 3, padding = (1, 1))
        # self.conv2 = nn.Conv2d(128, 128, 3, padding = (1, 1))
        # self.conv3 = nn.Conv2d(128, 128, 3, padding = (1, 1))
        # self.conv4 = nn.Conv2d(128, 128, 3, padding = (1, 1))

        self.conv5 = nn.Conv2d(384, 512, 1)
        self.conv6 = nn.Conv2d(512, 150, 1)

        self.conv7 = nn.Conv2d(384, 150, 1)
        
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

        # b x t x 256 x 64 x 64 to get through ConvLSTM
        xb = torch.stack(xb, dim = 0)
        layer_output, last_state = self.convlstm(xb)
        h = last_state[0][0]

        # # convert b x 384 x 64 x 64 to b x 2048 x 64 x 64
        y = F.relu(self.conv5(h))
        # y = h
        
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
        # y = self.conv7(y)
        
        return y

if fm == '':
    # Create a model
    model = WLSTM()
else:
    with lzma.open(fm, 'rb') as f:
        model = pickle.load(f)

# Send data to GPU
model.to(device)

# Create loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr = lr)

# Train the model
i = 0
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
    i += 1

# Export model
if fo != '':
    with lzma.open(fo, 'wb') as f:
        pickle.dump(model, f)
