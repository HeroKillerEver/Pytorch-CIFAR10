import torch
from torch import nn


cfg = {
    'vgg11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'vgg13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'vgg16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'vgg19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}



class Flatten(nn.Module):
    def forward(self, x):
        N = x.size(0) # read in N, C, H, W
        return x.view(N, -1) 
		



class VGG(nn.Module):
	"""docstring for VGG"""
	def __init__(self, vgg_type):
		super(VGG, self).__init__()
		self.type = type
		self.vgg = self.make_layers(cfg[vgg_type])
		self.fc = nn.Linear(512, 10)

	def forward(self, x):
		net = self.vgg(x)
		net = self.fc(net)
		return net


	def make_layers(self, model):
		layers = []
		in_channels = 3
		for x in model:
		    if x == 'M':
		        layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
		    else:
		        layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1), # (H - f + 2*P) / S + 1
		                        nn.BatchNorm2d(x),
		                        nn.ReLU(inplace=True)]
		        in_channels = x
		layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
		layers += [Flatten()]
		
		return nn.Sequential(*layers)






		