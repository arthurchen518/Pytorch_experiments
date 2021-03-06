import os 
import numpy as np
import torch
import torch.nn as nn 
import torch.optim as optim
from torch.nn.init import xavier_normal
from torch.autograd import Variable 
from torchvision import datasets, transforms
from PIL import Image

class Generator(nn.Module):
	def __init__(self, z_dim, h_dim, x_dim):
		super(Generator, self).__init__()
		self.net = nn.Sequential(
			
		)

	def forward(self, input):
		output = self.net(input)
		return output


class Discriminator(nn.Module):
	def __init__(self, x_dim, h_dim):
		super(Discriminator, self).__init__()
		self.net = nn.Sequential(
			
		)

	def forward(self, input):
		output = self.net(input)
		return output


# weight initializer for all layers
def init_weights(layer):
	if isinstance(layer, nn.Linear):
		xavier_normal(layer.weight.data)

z_dim = 100
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

def train(n_epoch, batch_size, gpu = False, save_path = './dcgan/'):
	# check path
	if not os.path.exists(save_path+'G/'):
		os.makedirs(save_path+'G/')
	if not os.path.exists(save_path+'D/'):
		os.makedirs(save_path+'D/')
	if not os.path.exists(save_path+'test/'):
		os.makedirs(save_path+'test/')

	# load input dataset
	mnist = datasets.MNIST('mnist/', train = True, download = True, 
							transform = transforms.Compose([
								transforms.ToTensor(), 
								transforms.Normalize((0.1307,), (0.3801,))]))

	train_loader = torch.utils.data.DataLoader(mnist, 
												batch_size = batch_size, 
												shuffle = True, num_workers = 4, 
												pin_memory = gpu, 
												drop_last = True)

	# build model
	G, D, loss, optimizerG, optimizerD = build_graph(gpu = gpu)


	# testing input
	test_noise = torch.randn(1, z_dim)
	if gpu:
		test_noise.cuda()
	test_in = Variable(test_noise)
	
	ones_label = Variable(torch.ones(batch_size, 1))
	zeros_label = Variable(torch.zeros(batch_size, 1))

	for epoch_idx in range(n_epoch):
		for batch_idx, (data, _) in enumerate(train_loader):
			noise = torch.randn(batch_size, z_dim)
			data.resize_(batch_size, x_dim)
			if gpu:
				noise.cuda()
				data.cuda()
			z = Variable(noise)
			x = Variable(data)
			g_sample = G(z)
			real = D(x)
			fake = D(g_sample.detach())

			# train the discriminator --minimize L(D) = -[log(D(x)) + log(1-D(G(z)))]
			D.zero_grad()
			D_loss_real = loss(real, ones_label)
			D_loss_fake = loss(fake, zeros_label)
			D_loss = D_loss_real + D_loss_fake
			D_loss.backward()
			optimizerD.step()

			# train the generator -- minimize L(G) = -log(D(G(z)))
			G.zero_grad()
			out = D(g_sample)
			G_loss = loss(out, ones_label)
			G_loss.backward()
			optimizerG.step()

			print "Epoch: \033[95m%d\033[0m batch: \033[95m%d\033[0m" % (epoch_idx, batch_idx)

		# save the model weights and the testing sample
		if epoch_idx%1 == 0:
			torch.save(G.state_dict(), save_path+'G/modelG_%d.pth'%epoch_idx)
			torch.save(D.state_dict(), save_path+'D/modelD_%d.pth'%epoch_idx)

			test_out = G(test_in)
			test_out = test_out.data.numpy()
			test_out = test_out.reshape(28,28)
			test_out -= test_out.min()
			test_out = np.uint8(255*test_out/test_out.max())
			# save the output image
			Image.fromarray(test_out, 'L').save(save_path+'test/%d.png'%epoch_idx,'png')



if __name__ == "__main__":
	train(10, 64, False)