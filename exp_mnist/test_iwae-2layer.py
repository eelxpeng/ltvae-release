import sys
sys.path.append("..")
import torch
import torch.utils.data
from torchvision import datasets, transforms
import numpy as np
from lib.iwae import IWAE
from lib.datasets import MNISTBinarized
import argparse

parser = argparse.ArgumentParser(description='VAE MNIST Example')
parser.add_argument('--lr', type=float, default=0.001, metavar='N',
                    help='learning rate for training (default: 0.001)')
parser.add_argument('--batch-size', type=int, default=256, metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('--epochs', type=int, default=10, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--pretrain', type=str, default="", metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--save', type=str, default="", metavar='N',
                    help='number of epochs to train (default: 10)')
args = parser.parse_args()

train_loader = torch.utils.data.DataLoader(
    MNISTBinarized('../dataset/mnist', train=True, download=True),
    batch_size=args.batch_size, shuffle=True, num_workers=0)
test_loader = torch.utils.data.DataLoader(
    MNISTBinarized('../dataset/mnist', train=False),
    batch_size=args.batch_size, shuffle=False, num_workers=0)

iwae = IWAE(input_dim=784, z_dim=20, binary=True,
        encodeLayer=[400,400], decodeLayer=[400,400], num_samples=50)
if args.pretrain != "":
    print("Loading model from %s..." % args.pretrain)
    iwae.load_model(args.pretrain)
iwae.fit(train_loader, test_loader, lr=args.lr, batch_size=args.batch_size,
    num_epochs=args.epochs, anneal=True)
if args.save != "":
	iwae.save_model(args.save)
