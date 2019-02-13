import sys
sys.path.append("..")
import torch
import torch.nn as nn
import torch.utils.data
from torchvision import datasets, transforms
import numpy as np
import scipy.io
from lib.autoencoder import Autoencoder
import argparse
from lib.utils import Dataset
from lib.initializer import variance_scaling_initializer
from lib.datasets import MNISTBinarized

parser = argparse.ArgumentParser(description='VAE MNIST Example')
parser.add_argument('--lr', type=float, default=0.001, metavar='N',
                    help='learning rate for training (default: 0.001)')
parser.add_argument('--batch-size', type=int, default=256, metavar='N',
                    help='input batch size for training (default: 256)')
parser.add_argument('--epochs', type=int, default=10, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--pretrain', type=str, default="", metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--save', type=str, default="", metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--dataset', type=str, default="mnist", metavar='N',
                    help='number of epochs to train (default: 10)')
args = parser.parse_args()

train_loader = torch.utils.data.DataLoader(
    MNISTBinarized('../dataset/mnist', train=True, download=True),
    batch_size=args.batch_size, shuffle=True, num_workers=0)
test_loader = torch.utils.data.DataLoader(
    MNISTBinarized('../dataset/mnist', train=False),
    batch_size=args.batch_size, shuffle=False, num_workers=0)

lr = 0.001
optimizer = "adam"
init = "xavier"
binary = True
if args.dataset == "mnist":
    input_dim = 784
    epochs = 300
    # optimizer = "sgd"
    # lr = 1.
    # init = "variance_scaling"

ae = Autoencoder(input_dim=input_dim, z_dim=20, binary=binary,
        encodeLayer=[400,400], decodeLayer=[400,400])

def weights_xavier_init(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform(m.weight.data)
        nn.init.constant(m.bias.data, 0)
def weights_variance_scaling_init(m):
    if isinstance(m, nn.Linear):
        variance_scaling_initializer(m.weight.data, scale=1. / 3., mode='fan_in',
                               distribution='uniform')
        nn.init.constant(m.bias.data, 0)
        
if init=="xavier":
    ae.apply(weights_xavier_init)
elif init == "variance_scaling":
    ae.apply(weights_variance_scaling_init)

if args.pretrain != "":
    print("Loading model from %s..." % args.pretrain)
    ae.load_model(args.pretrain)

print(ae)
ae.fit(train_loader, test_loader, lr=lr, num_epochs=epochs, anneal=False, optimizer=optimizer)
if args.save != "":
    ae.save_model(args.save)

