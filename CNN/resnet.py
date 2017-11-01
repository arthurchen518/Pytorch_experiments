import torch 
import torch.nn as nn
from torch.nn.init import xavier_normal, kaiming_normal

class ResBlock(nn.Module):
	expansion = 1
	name = 'ResBasic'
	def __init__(self, n_in, n_fb, stride = 1, cardinality = 1, 
					downsample = None, preactivation = False):
		super(ResBlock, self).__init__()
		self.downsample = downsample
		self.preactivation = preactivation
		self.n_in = n_in
		self.bn1 = nn.BatchNorm2d(n_in)
		self.relu = nn.ReLU(inplace = True)
		self.conv1 = nn.Conv2d(n_in, n_fb, kernel_size = 3, stride = stride, 
								padding = 1, bias = False)
		self.bn2 = nn.BatchNorm2d(n_fb)
		self.conv2 = nn.Conv2d(n_fb, self.expansion*n_fb, kernel_size = 3, stride = 1, 
								padding = 1, bias = False)

	def forward(self, x_in):
		if self.preactivation:
			y_out = self.bn1(x_in)
			y_out = self.relu(y_out)
			identity = y_out
		else:
			identity = x_in
			y_out = self.bn1(x_in)
			y_out = self.relu(y_out)

		y_out = self.conv1(y_out)

		y_out = self.bn2(y_out)
		y_out = self.relu(y_out)
		y_out = self.conv2(y_out)

		if self.downsample is not None:
			identity = self.downsample(identity)

		y_out += identity

		return y_out



class ResBottleNeck(nn.Module):
	expansion = 4
	name = 'ResBottleNeck'
	def __init__(self, n_in, n_fb, stride = 1, cardinality = 1, 
					downsample = None, preactivation = False):
		super(ResBottleNeck, self).__init__()
		self.downsample = downsample
		self.preactivation = preactivation
		self.n_in = n_in
		self.bn1 = nn.BatchNorm2d(n_in)
		self.relu = nn.ReLU(inplace = True)
		self.conv1 = nn.Conv2d(n_in, n_fb, kernel_size = 1, stride = 1, 
								padding = 0, bias = False)
		self.bn2 = nn.BatchNorm2d(n_fb)
		self.conv2 = nn.Conv2d(n_fb, n_fb, kernel_size = 3, stride = stride, 
								padding = 1, bias = False)
		self.conv3 = nn.Conv2d(n_fb, self.expansion*n_fb, kernel_size = 1, stride = 1, 
								padding = 0, bias = False)

	def forward(self, x_in):
		if self.preactivation:
			y_out = self.bn1(x_in)
			y_out = self.relu(y_out)
			identity = y_out
		else:
			identity = x_in
			y_out = self.bn1(x_in)
			y_out = self.relu(y_out)

		y_out = self.conv1(y_out)

		y_out = self.bn2(y_out)
		y_out = self.relu(y_out)
		y_out = self.conv2(y_out)

		y_out = self.bn2(y_out)
		y_out = self.relu(y_out)
		y_out = self.conv3(y_out)

		if self.downsample is not None:
			identity = self.downsample(identity)

		y_out += identity

		return y_out


class ResNeXtBottleNeck(nn.Module):
	expansion = 2
	name = 'ResNeXt'
	def __init__(self, n_in, n_fb, stride = 1, cardinality = 32, 
					downsample = None, preactivation = False):
		super(ResNeXtBottleNeck, self).__init__()
		self.downsample = downsample
		self.preactivation = preactivation
		self.n_in = n_in
		self.bn1 = nn.BatchNorm2d(n_in)
		self.relu = nn.ReLU(inplace = True)
		self.conv1 = nn.Conv2d(n_in, n_fb, kernel_size = 1, stride = 1, 
								padding = 0, bias = False)
		self.bn2 = nn.BatchNorm2d(n_fb)
		self.conv2 = nn.Conv2d(n_fb, n_fb, kernel_size = 3, stride = stride, 
								padding = 1, groups = cardinality, bias = False)
		self.conv3 = nn.Conv2d(n_fb, self.expansion*n_fb, kernel_size = 1, stride = 1, 
								padding = 0, bias = False)

	def forward(self, x_in):
		if self.preactivation:
			y_out = self.bn1(x_in)
			y_out = self.relu(y_out)
			identity = y_out
		else:
			identity = x_in
			y_out = self.bn1(x_in)
			y_out = self.relu(y_out)

		y_out = self.conv1(y_out)

		y_out = self.bn2(y_out)
		y_out = self.relu(y_out)
		y_out = self.conv2(y_out)

		y_out = self.bn2(y_out)
		y_out = self.relu(y_out)
		y_out = self.conv3(y_out)

		if self.downsample is not None:
			identity = self.downsample(identity)

		y_out += identity

		return y_out


class ResNet(nn.Module):
	def __init__(self, block, layers, input_dim, n_out, cardinality = 1):
		super(ResNet, self).__init__()
		self.n_in = 64
		if block.name == 'ResNeXt':
			n_fb_start = 128
		else:
			n_fb_start = 64
		self.conv1 = nn.Conv2d(3, 64, kernel_size = 7, stride = 2, padding = 3)
		self.conv2_x = self._make_layer(block, 1*n_fb_start, layers[0], cardinality, preactivation = True)
		self.conv3_x = self._make_layer(block, 2*n_fb_start, layers[1], cardinality, stride = 2)
		self.conv4_x = self._make_layer(block, 4*n_fb_start, layers[2], cardinality, stride = 2)
		self.conv5_x = self._make_layer(block, 8*n_fb_start, layers[3], cardinality, stride = 2)
		self.bn = nn.BatchNorm2d(8*n_fb_start*block.expansion)
		self.relu = nn.ReLU()
		h, w = input_dim[0]/16, input_dim[1]/16
		self.avgpool = nn.AvgPool2d((h, w))	
		self.fc = nn.Linear(8*n_fb_start*block.expansion, n_out)
		self._init_weight()

	def _make_layer(self, block, n_fb, n_layers, cardinality = 1, stride = 1, preactivation = False):
		layers = []
		downsample = None
		if stride != 1 or self.n_in != n_fb*block.expansion:
			downsample = nn.Conv2d(self.n_in, n_fb*block.expansion, kernel_size = 1, 
									stride = stride, bias = False)

		layers.append(block(self.n_in, n_fb, stride, cardinality, downsample, preactivation))
		self.n_in = n_fb*block.expansion
		for _ in range(n_layers-1):
			layers.append(block(self.n_in, n_fb))

		return nn.Sequential(*layers)

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

	def forward(self, x_in):
		y_out = self.conv1(x_in)

		y_out = self.conv2_x(y_out)
		y_out = self.conv3_x(y_out)
		y_out = self.conv4_x(y_out)
		y_out = self.conv5_x(y_out)
		y_out = self.bn(y_out)
		y_out = self.relu(y_out)
		y_out = self.avgpool(y_out)
		y_out = y_out.view(y_out.size(0), -1)
		y_out = self.fc(y_out)
		return y_out


def ResNet18(input_dim, n_out):
	model = ResNet(ResBlock, [2, 2, 2, 2], input_dim, n_out)
	return model

def ResNet50(input_dim, n_out):
	model = ResNet(ResBottleNeck, [3, 4, 6, 3], input_dim, n_out)
	return model

def ResNet101(input_dim, n_out):
	model = ResNet(ResBottleNeck, [3, 4, 23, 3], input_dim, n_out)
	return model

def ResNet152(input_dim, n_out):
	model = ResNet(ResBottleNeck, [3, 8, 36, 3], input_dim, n_out)
	return model

def ResNeXt50(input_dim, n_out, cardinality = 32):
	model = ResNet(ResNeXtBottleNeck, [3, 4, 6, 3], input_dim, n_out, cardinality)
	return model

def ResNeXt101(input_dim, n_out, cardinality = 32):
	model = ResNet(ResNeXtBottleNeck, [3, 4, 23, 3], input_dim, n_out, cardinality)
	return model

def ResNeXt152(input_dim, n_out, cardinality = 32):
	model = ResNet(ResNeXtBottleNeck, [3, 8, 36, 3], input_dim, n_out, cardinality)
	return model
	
