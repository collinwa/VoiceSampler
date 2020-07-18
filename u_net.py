import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import ModuleList, Tanh, Linear, MaxPool1d, Conv1d, LeakyReLU, BatchNorm1d, GRU

def compute_mse(y_pred, y):
    return (y_pred - y).pow(2).sum()


class UNet(nn.Module):
    def __init__(self, depth=3, audio_ch = 2, ch_factor=2, relu_slope=0.01, kern_size=5, max_pool=2, eps=1e-05, momentum=0.1, gru=True):
        super(UNet, self).__init__()

        self.down_conv_dict = {'kern_size': kern_size, 
                                'max_pool': max_pool,
                                'eps': eps,
                                'momentum': momentum,
                                'slope': relu_slope
                            }
        self.depth = depth
        self.gru = gru
        self.ch_factor = ch_factor
        self.kern_size = kern_size
        self.max_pool = max_pool
        self.eps = eps
        self.momentum = momentum

        self.first_channels = ch_factor ** 6
        self.down_layers = nn.ModuleList()
        self.up_layers = nn.ModuleList()
        
        channels = lambda base, scale, depth: base * (scale ** depth)

        padding = (kern_size) // 2
        self.conv1 = DoubleConvAndPool(audio_ch, self.first_channels, **self.down_conv_dict, use_pool=False) 
        self.down_layers.append(self.conv1)
        
        for i in range(0, depth):
            self.down_layers.append(DoubleConvAndPool( channels(self.first_channels, ch_factor, i), channels(self.first_channels, ch_factor, i + 1), **self.down_conv_dict))

        if self.gru:
            self.gru_layer = GRULayer( channels(self.first_channels, ch_factor, i + 1) )

        self.first_up = DoubleConvConcatAndDilate( channels(self.first_channels, ch_factor, i + 1), channels(self.first_channels, ch_factor, i), **self.down_conv_dict, use_skip=False)

        for i in reversed(range(0, depth-1)):
            self.up_layers.append(DoubleConvConcatAndDilate( channels(self.first_channels, ch_factor, i + 1), channels(self.first_channels, ch_factor, i), **self.down_conv_dict ))


        self.final_out = DoubleConvAndPool(self.first_channels, audio_ch, **self.down_conv_dict, use_pool=False) 

        self.tanh = Tanh()

        self.max_channels = self.first_channels * (ch_factor ** (depth - 1))

    def forward(self, x):
        assert x.shape[2] % (self.ch_factor ** self.depth) == 0
        
        skip_list = []
        for (idx, layer) in enumerate(self.down_layers):
            x = layer(x)
            skip_list.append(x)

        print([t.shape for t in skip_list])
        # apply recurrent unit
 
        if self.gru:
            x = self.gru_layer(x)

        # upsample 1 layer to match dimensions
        x = self.first_up(x)

        for (idx, layer) in enumerate(self.up_layers):
            x = self.up_layers[idx](x, skip_connection=skip_list[-2-idx])
        
        x = self.tanh(self.final_out(x))

        return x




class DoubleConvAndPool(nn.Module):
    def __init__(self, in_channels, out_channels, slope=0.01, kern_size=5, max_pool=2, eps=1e-05, momentum=0.1, use_pool=True):
        super(DoubleConvAndPool, self).__init__()
        padding = (kern_size) // 2 # just a step size of 1 and 'same' padding
        self.conv1 = Conv1d(in_channels, out_channels, kernel_size=kern_size, padding=padding)
        self.batch_norm_1 = BatchNorm1d(out_channels)
        self.relu_1 = LeakyReLU(negative_slope=slope)
        
        self.conv2 = Conv1d(out_channels, out_channels, kernel_size=kern_size, padding=padding)
        self.batch_norm_2 = BatchNorm1d(out_channels)
        self.relu_2 = LeakyReLU(negative_slope=slope)
        self.pool = MaxPool1d(kern_size, stride=max_pool, padding=padding)

        self.use_pool = use_pool

    def forward(self, x):
        # first convolution
        # pool first
        
        if self.use_pool:
            x = self.pool(x)
        x = self.conv1(x)
        x = self.batch_norm_1(x)
        x = self.relu_1(x)
        
        # second convlution
        x = self.conv2(x)
        x = self.batch_norm_2(x)
        x = self.relu_2(x)

        return x


class DoubleConvConcatAndDilate(nn.Module):
    def __init__(self, in_channels, out_channels, slope=0.01, kern_size=5, max_pool=2, eps=1e-05, momentum=0.1, use_skip=True):
        super(DoubleConvConcatAndDilate, self).__init__()
        padding = kern_size // 2
        transpose_padding = padding - 1

        # we need to account for the skip connection
        if use_skip:
            in_channels += in_channels

        self.conv1 = ConvTranspose1d(in_channels, out_channels, stride=max_pool, kernel_size=kern_size, padding=transpose_padding)

        self.batch_norm_1 = BatchNorm1d(out_channels)
        self.relu_1 = LeakyReLU(negative_slope=slope)
        
        self.conv2 = Conv1d(out_channels, out_channels, kernel_size=kern_size, padding=padding)
    
        self.batch_norm_2 = BatchNorm1d(out_channels)
        self.relu_2 = LeakyReLU(negative_slope=slope)

    def forward(self, x, skip_connection=None):
        if (skip_connection is not None):
            to_trim = x.shape[2] - skip_connection.shape[2]

            to_trim_left = to_trim // 2
            to_trim_right = to_trim_left + 1 if to_trim % 2 == 1 else to_trim_left

            print(x.shape)
            print(skip_connection.shape)
            if to_trim != 0:
                skip_connection = skip_connection[to_trim_left:-to_trim_right]
            x = torch.cat((x, skip_connection), 1) # concat along the channels dimension
            print(x.shape)
        x = self.conv1(x)[:,:,:-1] # NOTE: sort of hacky fix here
        x = self.batch_norm_1(x)
        x = self.relu_1(x)
        
        # second convlution
        x = self.conv2(x)
        x = self.batch_norm_2(x)
        x = self.relu_2(x)

        return x

class GRULayer(nn.Module):
    def __init__(self, input_ch, bidirectional=True):
        super(GRULayer, self).__init__()
        self.gru = GRU(input_ch, input_ch // 2, bidirectional=bidirectional, batch_first=True)
    
    def forward(self, x):
        print(x.shape)
        x = torch.transpose(x, 1, 2)
        x, _ = self.gru(x)
        x = torch.transpose(x, 1, 2)

        return x
