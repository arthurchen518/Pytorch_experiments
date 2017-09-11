import torch
import torch.nn as nn
from torch.nn.init import xavier_normal, kaiming_normal

class DenseBlock(nn.Module):
	def __init__(self, n_in, n_layers, growth_rate, drop_rate):
		super(DenseBlock, self).__init__()
		layers = []
		for i in range(n_layers):
			layers.append(self._make_layer(n_in, growth_rate, drop_rate))
		self.block = nn.Sequential(*layers)

	def _make_layer(self, n_in, growth_rate, drop_rate):
		return nn.Sequential(
				nn.BatchNorm2d(n_in),
				nn.ReLU(inplace = True),
				nn.Conv2d(n_in, , kernel_size = 1, stride = 1, bias = False),
				nn.BatchNorm2d(),
				nn.Conv2d(, growth_rate, kernel_size = 3, stride = 1, 
								padding = 1, bias = False),
				nn.Dropout(p = drop_rate, inplace = True)
			)

	def forward(self, x):
		y = block(x)
		return y

class DenseNet(nn.Module):
	def __init__(self, n_out, input_dim, growth_rate = 32, n_layers = [6, 12, 24, 16], drop_rate = 0.2):
		super(DenseNet, self).__init__()
		self.conv1 = nn.Conv2d(3, 64, kernel_size = 7, stride = 2, padding = 3, bias = False)
		self.bn1 = nn.BatchNorm2d(64)
		self.relu = nn.ReLU(inplace = True)
		self.global_avgpool = nn.AvgPool2d((,))
		self.fc = nn.Linear(, n_out)
		self.denseblocks = nn.Sequential()

		n_in = 64
		for i, n_layer in enumerate(n_layers):
			self.denseblocks.add_module('denseblock%d'%(i+1), 
										DenseBlock(n_in, n_layer, growth_rate, drop_rate))
			n_in = n_in + growth_rate*n_layer
			if i != len(n_layers)-1:
				self.denseblocks.add_module('transition%d'%(i+1), 
											self._transition_layer(n_in, n_in//2))

		self.final_bn = nn.BatchNorm2d(n_in)

	def forward(self, x):
		y = self.conv1(x)
		y = self.bn1(y)
		y = self.relu(y)
		y = self.denseblocks(y)
		y = self.final_bn(y)
		y = self.relu(y)
		y = self.global_avgpool(y)
		y = y.view(y.size(0), -1)
		y = self.fc(y)
		return y

	def _transition_layer(self, n_in, n_out):
		return nn.Sequential(
				nn.BatchNorm2d(n_in),
				nn.ReLU(inplace = True),
				nn.Conv2d(n_in, n_out, kernel_size = 1, bias = False),
				nn.AvgPool2d(kernel_size = 2)
			)

	def _init_weight(self):
		for m in self.modules():
			if isinstance(m, nn.Conv2d):
				kaiming_normal(m.weight.data)
			elif isinstance(m, nn.BatchNorm2d):
				m.weight.data.fill_(1)
				m.bias.data.zero_()
			elif isinstance(m, nn.Linear):
				xavier_normal(m.weight.data)
				m.bias.data.zero_()