import torch
import torch.nn as nn
import torch.nn.functional as F
from convlstm import ConvLSTM
from axial_attention import AxialAttention

# Define the major model
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
        self.conv5 = nn.Conv2d(384, 512, 1)
        self.conv6 = nn.Conv2d(512, 1, 1)
        # self.linear = nn.Linear(512 * 64 * 64, 64 * 64)

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
            y = torch.tanh(self.conv1(y))
            # y = F.max_pool2d(y, 2)
            y = F.avg_pool2d(y, 2)
            y = torch.tanh(self.conv2(y))
            y = torch.tanh(self.conv3(y))
            y = torch.tanh(self.conv4(y))
            # y = F.max_pool2d(y, 2)
            y = F.avg_pool2d(y, 2)
            xb.append(y)

        # b x t x 256 x 64 x 64 to get through ConvLSTM
        xb = torch.stack(xb, dim = 0)
        layer_output, last_state = self.convlstm(xb)
        h = last_state[0][0]

        # # convert b x 384 x 64 x 64 to b x 2048 x 64 x 64
        y = torch.tanh(self.conv5(h))
        # y = h
        
        # Axial self attension
        y = torch.tanh(self.attn1(y))
        y = torch.tanh(self.attn2(y))
        y = torch.tanh(self.attn1(y))
        y = torch.tanh(self.attn2(y))
        y = torch.tanh(self.attn1(y))
        y = torch.tanh(self.attn2(y))
        y = torch.tanh(self.attn1(y))
        y = torch.tanh(self.attn2(y))

        # Convert to b x 20 x 64 x 64 to represent probability
        y = self.conv6(y)
        y = y.view(-1)
        # y = y.flatten(start_dim = 2)
        # y = y.transpose(1, 2)
        # y = y.flatten(start_dim = 0, end_dim = 1)
        # y = self.linear(y)
        
        return y
