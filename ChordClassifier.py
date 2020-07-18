import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Conv1d, BatchNorm1d, ReLU, Linear, Dropout, MaxPool1d

class ConvTwin(nn.Module):
	def __init__(self, in_channels=2, 
		num_layers=5,
		filter_width=11,
		channels=[16, 32, 64, 128, 256],
		dropout=0.3,
		rate=44100,
		duration=0.5,
		flat_dim = 512 
		):

		assert num_layers == len(channels)
		super(ConvTwin, self).__init__()

		extra_px = filter_width // 2
		channels = [in_channels] + channels		

		total_len = int(2 ** np.floor(np.log2(rate * duration)))

		conv_block = lambda in_ch, out_ch, p=dropout, fw=filter_width, pd=extra_px: nn.Sequential(
			Conv1d(in_ch, out_ch, kernel_size=fw, padding=pd),
			BatchNorm1d(out_ch),
			ReLU(),
			Dropout(p=p),
			MaxPool1d(kernel_size=2)
		)

		convs = []

		for i in range(1, len(channels)):
			convs.append(conv_block(channels[i-1], channels[i]))

		self.dense = Linear(total_len * channels[-1] // (2 ** len(convs)), flat_dim)
		self.L = total_len

		self.convs = convs
		self.convs = nn.Sequential(*self.convs)

	def forward(self, x):
		assert x.shape[-1] >= self.L

		if x.shape[-1] > self.L:
			all_px = (x.shape[-1] - self.L) 
			to_trim_right = all_px // 2
			to_trim_left = all_px // 2 + 1 if all_px % 2 == 1 else all_px // 2
			x = x[:,:,to_trim_left:-to_trim_right] 

		for i in range(0, len(self.convs)):
			x = self.convs[i](x)
			# print(x.size())

		x = x.contiguous().view(x.shape[0], -1)
		x = self.dense(x)
		return F.relu(x)

class SiameseArchitecture(nn.Module):
	def __init__(self, **kwargs):	
		super(SiameseArchitecture, self).__init__()
		kwargs_dict = {**kwargs}
		self.ref_net = ConvTwin(**kwargs)
		self.comp_net = ConvTwin(**kwargs)
		self.dense = Linear(kwargs_dict['flat_dim'], 1)

	def forward(self, x_ref, x_new):
		out_1 = self.ref_net(x_ref)
		out_2 = self.comp_net(x_new)
		out = out_2 - out_1
		out = self.dense(out)	
		return torch.sigmoid(out)
