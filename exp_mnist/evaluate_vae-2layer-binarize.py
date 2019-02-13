import sys
sys.path.append("..")
import torch
from torch.autograd import Variable
import torch.utils.data
from torchvision import datasets, transforms
import numpy as np
from lib.vae import VAE
from lib.datasets import MNISTBinarized
import argparse

parser = argparse.ArgumentParser(description='LTVAE MNIST Example')
parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('--samples', type=int, default=100, metavar='N',
                    help='number of samples for importance sampling (default: 10)')
parser.add_argument('--model', type=str, default="", metavar='N',
                    help='save model checkpoint')
args = parser.parse_args()
use_cuda = torch.cuda.is_available()
train_loader = torch.utils.data.DataLoader(
    MNISTBinarized('../dataset/mnist', train=True, download=True),
    batch_size=args.batch_size, shuffle=True, num_workers=0)
test_loader = torch.utils.data.DataLoader(
    MNISTBinarized('../dataset/mnist', train=False),
    batch_size=args.batch_size, shuffle=False, num_workers=0)

z_dim = 20
vae = VAE(input_dim=784, z_dim=z_dim, binary=True,
        encodeLayer=[400,400], decodeLayer=[400,400])
print("Loading model from %s..." % args.model)
vae.load_model(args.model)
if use_cuda:
    vae.cuda()

# validate
vae.train()
loglikelihood = 0.0
for batch_idx, (inputs, _) in enumerate(test_loader):
    inputs = inputs.view(inputs.size(0), -1).float()
    if use_cuda:
        inputs = inputs.cuda()
    inputs = Variable(inputs)
    loglikelihood += torch.sum(vae.log_marginal_likelihood_estimate(inputs, args.samples))
ave = loglikelihood/len(test_loader.dataset)
print("mean loglikelihood: ", ave)
