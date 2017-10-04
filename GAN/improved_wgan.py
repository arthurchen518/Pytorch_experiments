# 1. Remove the last sigmoid in the discriminator
# 2. Clip the weights of the discriminator
# 3. Train the discriminator more than the generator
# 4. Use RMSprop instead of any momentum-based optimizer such as Adam and use lower learning rate
# 5. Use Wasserstein loss instead of JS divergence

import os 
import numpy as np
import torch
import torch.nn as nn 
import torch.optim as optim
from torch.nn.init import xavier_normal
from torch.autograd import Variable, grad
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

	optimizerG = optim.Adam(G.parameters(), lr = 1e-4, betas = (0.1, 0.9))
	optimizerD = optim.Adam(D.parameters(), lr = 1e-4, betas = (0.1, 0.9))

	return G, D, loss, optimizerG, optimizerD

def train(step, batch_size, gpu = False):
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
	w = 10
	for step_idx in range(step):
		for _ in range(5):
			data, _ = mnist.train.next_batch(batch_size)
			data = torch.from_numpy(data)
			noise = torch.randn(batch_size, z_dim)
			e = torch.randn(batch_size, 1)
			e = e.expand(data.size())
			if gpu:
				noise = noise.cuda()
				data = data.cuda()
				e = e.cuda()
			z = Variable(noise)
			x = Variable(data)
			g_sample = G(z)
			interpolate = e*x.data+(1-e)*g_sample.data
			interpolate = Variable(interpolate, requires_grad = True)
			D_interpolate = D(interpolate)
			grad_outputs = torch.ones(D_interpolate.size()).cuda() if gpu else torch.ones(D_interpolate.size())
			gradients = grad(outputs = D_interpolate, 
							inputs = interpolate, 
							grad_outputs = grad_outputs, 
							create_graph = True, 
							retain_graph = True, 
							only_inputs = True)[0]
			real = D(x)
			fake = D(g_sample.detach())
			# train the discriminator --minimize L(D) = -[f(D(x)) - f(D(G(z)))]
			D.zero_grad()
			D_loss = torch.mean(fake) - torch.mean(real) + w*torch.mean((gradients.norm(2, dim = 1)-1)**2)
			D_loss.backward()
			optimizerD.step()


		noise = torch.randn(batch_size, z_dim)
		if gpu:
			noise = noise.cuda()
		z = Variable(noise)
		g_sample = G(z)
		# train the generator -- minimize L(G) = -f(D(G(z)))
		G.zero_grad()
		out = D(g_sample)
		G_loss = -torch.mean(out)
		G_loss.backward()
		optimizerG.step()

		# save the model weights and the testing sample
		if step_idx % 1000 == 0:
			print 'Iter-{}; Wasserstein_loss: {:.4};'.format(step_idx, -D_loss.data[0])
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
	train(100000, 64, torch.cuda.is_available())