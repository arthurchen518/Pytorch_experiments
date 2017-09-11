import torch
import torch.nn as nn
from torch.nn.init import xavier_normal, kaiming_normal

def ConvBlock(n_in, n_fb, kernel_size = 3, stride = 1, padding = 1, downsample = False):
	layers = [
			nn.Conv2d(n_in, n_fb, kernel_size = 3, padding = 1),
			nn.BatchNorm2d(n_fb),
			nn.ReLU(inplace = True)
		]
	if downsample:
		layers.append(nn.MaxPool2d(kernel_size = 2))
		
	return nn.Sequential(*layers)

class vgg16(nn.Module):
	def __init__(self, input_dim, n_out):
		super(vgg16, self).__init__()
		h, w = input_dim[0]/32, input_dim[1]/32
		self.convnet = nn.Sequential(
				ConvBlock(3, 64),
				ConvBlock(64, 64, downsample = True),
				ConvBlock(64, 128),
				ConvBlock(128, 128, downsample = True),
				ConvBlock(128, 256),
				ConvBlock(256, 256),
				ConvBlock(256, 256, downsample = True),
				ConvBlock(256, 512),
				ConvBlock(512, 512),
				ConvBlock(512, 512, downsample = True),
				ConvBlock(512, 512),
				ConvBlock(512, 512),
				ConvBlock(512, 512, downsample = True)
			)
		self.fully_connected = nn.Sequential(
				nn.Linear(512*h*w, 1024),
				nn.ReLU(inplace = True),
				nn.Dropout(p = 0.5),
				nn.Linear(1024, 1024),
				nn.ReLU(inplace = True),
				nn.Dropout(p = 0.5),
				nn.Linear(1024, n_out)
			)
		self._init_weight()

	def forward(self, x):
		y = self.convnet(x)
		y = y.view(y.size(0), -1)
		y = self.fully_connected(y)
		return y

	def _init_weight(self):
		for m in self.modules():
			if isinstance(m, nn.Conv2d):
				kaiming_normal(m.weight.data)
				m.bias.data.zero_()
			elif isinstance(m, nn.Linear):
				xavier_normal(m.weight.data)
				m.bias.data.zero_()
			elif isinstance(m, nn.BatchNorm2d):
				m.weight.data.fill_(1)
				m.bias.data.zero_()
