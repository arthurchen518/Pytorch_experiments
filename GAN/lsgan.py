# 1. Remove the last sigmoid layer in the discriminator 
# 2. Use L2 loss instead of log loss

import os 
import torch
import torch.nn as nn 
import torch.optim as optim
from torch.nn.init import xavier_normal
from torch.autograd import Variable 
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from tensorflow.examples.tutorials.mnist import input_data

class Generator(nn.Module):
	def __init__(self, z_dim, h_dim, x_dim):
		super(Generator, self).__init__()
		self.net = nn.Sequential(
			nn.Linear(z_dim, h_dim),
			nn.ReLU(),
			nn.Linear(h_dim, x_dim),
			nn.Sigmoid()
		)

	def forward(self, input):
		output = self.net(input)
		return output


class Discriminator(nn.Module):
	def __init__(self, x_dim, h_dim):
		super(Discriminator, self).__init__()
		self.net = nn.Sequential(
			nn.Linear(x_dim, h_dim),
			nn.ReLU(),
			nn.Linear(h_dim, 1)
		)

	def forward(self, input):
		output = self.net(input)
		return output


# weight initializer for all layers
def init_weights(layer):
	if isinstance(layer, nn.Linear):
		xavier_normal(layer.weight.data)

z_dim = 10
x_dim = 28*28
h_dim = 128

def build_graph(gpu = False):

	G = Generator(z_dim, h_dim, x_dim)
	D = Discriminator(x_dim, h_dim)

	G.apply(init_weights)
	D.apply(init_weights)

	loss = nn.BCELoss()

	if gpu:
		G.cuda()
		D.cuda()
		loss.cuda()

	optimizerG = optim.Adam(G.parameters(), lr = 1e-3)
	optimizerD = optim.Adam(D.parameters(), lr = 1e-3)

	return G, D, loss, optimizerG, optimizerD

def train(n_step = 10000, batch_size = 64, gpu = False):

	# load input dataset
	mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

	# build model
	G, D, loss, optimizerG, optimizerD = build_graph(gpu = gpu)

	# testing input
	test_noise = torch.randn(1, z_dim)
	if gpu:
		test_noise.cuda()
	test_in = Variable(test_noise)
	
	ones_label = Variable(torch.ones(batch_size, 1))
	zeros_label = Variable(torch.zeros(batch_size, 1))
	if gpu:
		ones_label=ones_label.cuda()
		zeros_label=zeros_label.cuda()
	count = 0

	for step_idx in range(n_step):
		data, _ = mnist.train.next_batch(batch_size)
		data = torch.from_numpy(data)
		noise = torch.randn(batch_size, z_dim)
		if gpu:
			noise = noise.cuda()
			data = data.cuda()
		z = Variable(noise)
		x = Variable(data)
		g_sample = G(z)
		real = D(x)
		fake = D(g_sample.detach())

		# When b-c = 1 and b-a = 2, the loss is tantamount to Pearson x^2 divergence.
		# train the discriminator --minimize L(D) = 0.5*[(D(x)-a)^2+(D(G(z))-b)^2]
		a = -1
		b = 1
		c = 0
		D.zero_grad()
		D_loss = 0.5*(torch.mean((real-a)**2)+torch.mean((fake-b)**2))
		D_loss.backward()
		optimizerD.step()

		noise = torch.randn(batch_size, z_dim)
		if gpu:
			noise = noise.cuda()
		z = Variable(noise)
		g_sample = G(z)
		# train the generator -- minimize L(G) = 0.5*(D(G(z))-c)^2
		G.zero_grad()
		out = D(g_sample)
		G_loss = 0.5*torch.mean((out-c)**2)
		G_loss.backward()
		optimizerG.step()

		# save the model weights and the testing sample
		if step_idx % 1000 == 0:
			print 'Iter-{}; D_loss: {:.4}; G_loss: {:.4}'.format(step_idx, D_loss.data[0], G_loss.data[0])
			samples = G(z).data.cpu().numpy()[:16]
			fig = plt.figure(figsize=(4, 4))
			gs = gridspec.GridSpec(4, 4)
			gs.update(wspace=0.05, hspace=0.05)

			for i, sample in enumerate(samples):
				ax = plt.subplot(gs[i])
				plt.axis('off')
				ax.set_xticklabels([])
				ax.set_yticklabels([])
				ax.set_aspect('equal')
				plt.imshow(sample.reshape(28, 28), cmap='Greys_r')

			if not os.path.exists('out/'):
				os.makedirs('out/')

			plt.savefig('out/{}.png'.format(str(count).zfill(3)), bbox_inches='tight')
			count += 1
			plt.close(fig)


if __name__ == "__main__":
	train(10000, 64, torch.cuda.is_available())








			

			
