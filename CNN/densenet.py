import torch
import torch.nn as nn
from torch.nn.init import xavier_normal, kaiming_normal

class DenseBottleNeckLayer(nn.Module):
	def __init__(self, n_in, growth_rate, drop_rate):
		super(DenseBottleNeckLayer, self).__init__()
		self.bn1 = nn.BatchNorm2d(n_in)
		self.relu = nn.ReLU(inplace = True)
		self.conv1 = nn.Conv2d(n_in, 4*growth_rate, kernel_size = 1, stride = 1, bias = False)
		self.bn2 = nn.BatchNorm2d(4*growth_rate)
		self.conv2 = nn.Conv2d(4*growth_rate, growth_rate, kernel_size = 3, stride = 1, 
								padding = 1, bias = False)
		self.dropout = nn.Dropout(p = drop_rate, inplace = True)

	def forward(self, x):
		y = self.bn1(x)
		y = self.relu(y)
		y = self.conv1(y)

		y = self.bn2(y)
		y = self.relu(y)
		y = self.conv2(y)

		y = self.dropout(y)

		return torch.cat([x, y], 1)

class DenseBlock(nn.Sequential):
	def __init__(self, n_in, n_layers, growth_rate, drop_rate):
		super(DenseBlock, self).__init__()
		for i in range(n_layers):
			self.add_module('DenseBottleNeckLayer%d'%(i + 1), 
								DenseBottleNeckLayer(n_in+i*growth_rate, growth_rate, drop_rate))

class DenseNet(nn.Module):
	def __init__(self, n_out, input_dim, growth_rate = 32, n_layers = [6, 12, 24, 16], drop_rate = 0.2):
		super(DenseNet, self).__init__()
		self.conv1 = nn.Conv2d(3, 64, kernel_size = 7, stride = 2, padding = 3, bias = False)
		self.bn1 = nn.BatchNorm2d(64)
		self.relu = nn.ReLU(inplace = True)
		self.denseblocks = nn.Sequential()

		n_in = 64
		for i, n_layer in enumerate(n_layers):
			self.denseblocks.add_module('denseblock%d'%(i+1), 
										DenseBlock(n_in, n_layer, growth_rate, drop_rate))
			n_in = n_in + growth_rate*n_layer
			if i != len(n_layers)-1:
				self.denseblocks.add_module('transition%d'%(i+1), 
											self._transition_layer(n_in, n_in//2))
				n_in = n_in//2

		self.final_bn = nn.BatchNorm2d(n_in)
		h, w = input_dim[0]/16, input_dim[1]/16
		self.global_avgpool = nn.AvgPool2d((h, w))
		self.fc = nn.Linear(n_in, n_out)

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


def Dense121(n_out, input_dim, growth_rate = 32, drop_rate = 0.2):
	model =  DenseNet(n_out, input_dim, growth_rate = growth_rate, 
					n_layers = [6, 12, 24, 16], drop_rate = drop_rate)
	return model

