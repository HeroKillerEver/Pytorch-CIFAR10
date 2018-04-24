import torch
from torch import nn
from torch.autograd import Variable
import torchvision
import torchvision.transforms as T
from torch.utils.data import DataLoader, Dataset



import argparse
import os
from models import vgg, resnet, densenet




parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training', epilog='#' * 60)
parser.add_argument('--lr', default=0.1, type=float, help='learning rate, Default: 0.1', dest='lr')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint', dest='resume')
parser.add_argument('--model', '-m', type=str, help='model to use: vgg16, vgg19, \
													resnet18, resnet34, resnet50, resnet101, resnet152, \
													densenet121, densenet161, densenet169, densenet201', dest='model')
parser.add_argument('--gpu', default=0, type=int, help='gpu to use: 0, 1, 2, 3, 4, 5, Default: 0', dest='gpu')
parser.add_argument('--epoch', default=400, type=int, help='how many epochs, Default: 400')
parser.add_argument('--visible', type=str, help='visible gpus')
args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = args.visible

print "loading cifar10 dataset...."

T_train = T.Compose([
    T.RandomCrop(32, padding=4),
    T.RandomHorizontalFlip(),
    T.ToTensor(),
    T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

T_test = T.Compose([
    T.ToTensor(),
    T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

cifar_train = torchvision.datasets.CIFAR10(root='./data', download=True, train=True, transform=T_train)
cifar_test = torchvision.datasets.CIFAR10(root='./data', download=True, train=False, transform=T_train)

cifar_train_loader = DataLoader(cifar_train, batch_size=100, shuffle=True, num_workers=10)
cifar_test_loader = DataLoader(cifar_test, batch_size=100, shuffle=False, num_workers=10)

def train(epoch, net, criterion, optim, lr_scheduler):

	net.train()
	lr_scheduler.step()
	correct = 0
	total = 0
	for idx, (images, labels) in enumerate(cifar_train_loader):
		optim.zero_grad()
		x, y = Variable(images.cuda(args.gpu)), Variable(labels.cuda(args.gpu))
		logits = net(x)
		loss = criterion(logits, y)
		loss.backward()
		optim.step()
		_, labels_pred = torch.max(logits.data, 1)

		total += labels.size(0)
		correct += torch.eq(labels_pred, y.data).cpu().sum()

	print 'epoch: [{0:d}],  loss: [{1:.4f}], acc: [{2:.4f}%]'.format(epoch, loss.data.cpu()[0], 100.*correct/total)

def test(epoch, net):

	net.eval()
	correct = 0
	total = 0
	for idx, (images, labels) in enumerate(cifar_test_loader):
		x, y = Variable(images.cuda(args.gpu)), Variable(labels.cuda(args.gpu))
		logits = net(x)
		_, labels_pred = torch.max(logits.data, 1)
		total += labels.size(0)
		correct += torch.eq(labels_pred, y.data).cpu().sum()

	print 'epoch: [{0:d}],  acc: [{1:.4f}%]'.format(epoch, 100.*correct/total)

	if not os.path.isdir('checkpoint'):
		os.mkdir('checkpoint')

	torch.save(net, 'checkpoint/{}.pkl'.format(args.model))
	print 'model saved...!'












def main():
	
	
	if args.resume:
		if not os.path.isfile('./checkpoint/{}.pkl'.format(args.model)):
			raise ValueError('no models saved....!!!!')
		print 'resume from checkpoint....'
		net = torch.load('./checkpoint/{}.pkl'.format(args.model))
	else:
		if args.model == 'vgg16':
			net = vgg.VGG(args.model)
		elif args.model == 'vgg19':
			net = vgg.VGG(args.model)
		elif args.model == 'resnet18':
			net = resnet.ResNet18()
		elif args.model == 'resnet34':
			net = resnet.ResNet34()
		elif args.model == 'resnet50':
			net = resnet.ResNet50()
		elif args.model == 'resnet101':
			net = resnet.ResNet101()
		elif args.model == 'resnet152':
			net = resnet.ResNet152()
		elif args.model == 'densenet121':
			net = densenet.DenseNet121()
		elif args.model == 'densenet161':
			net = densenet.DenseNet161()
		elif args.model == 'densenet169':
			net = densenet.DenseNet169()
		elif args.model == 'densenet201':
			net = densenet.DenseNet201()
		else:
			raise ValueError('model not implemented...!!')


	net.cuda(args.gpu)
	net = nn.DataParallel(net, device_ids = range(torch.cuda.device_count()))
	criterion = nn.CrossEntropyLoss().cuda(args.gpu)
	optim = torch.optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-4)
	lr_scheduler = torch.optim.lr_scheduler.StepLR(optim, step_size=100, gamma=0.1)



	for e in xrange(args.epoch):
		train(e, net, criterion, optim, lr_scheduler)
		test(e, net)

if __name__ == '__main__':
	main()
