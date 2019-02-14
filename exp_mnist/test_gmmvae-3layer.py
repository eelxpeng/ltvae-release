"""
python test_gmmvae-3layer.py --lr 0.0001 --lr-stepwise 0.01 --alpha 0.1 --epochs 300 --pretrain model/sdae.pt --save model/gmmvae.pt
On MNIST achieves
acc: 0.8677, nmi: 0.8256 when alpha=1.0
acc: 0.8865, nmi: 0.8609 when alpha=0.1
"""
import sys
sys.path.append("..")
import torch
import torch.utils.data
from torchvision import datasets, transforms
import numpy as np
import argparse
from lib.gmmvae_fixed_var import GMMVAE
from lib.datasets import MNIST
from lib.utils import init_logging
import logging
import time

parser = argparse.ArgumentParser(description='VAE MNIST Example')
parser.add_argument('--lr', type=float, default=0.001, metavar='N',
                    help='learning rate for training (default: 0.001)')
parser.add_argument('--lr-stepwise', type=float, default=0.01, metavar='N',
                    help='learning rate for training (default: 0.001)')
parser.add_argument('--alpha', type=float, default=1., metavar='N',
                    help='set value of alpha')
parser.add_argument('--batch-size', type=int, default=256, metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('--update-interval', type=int, default=10, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--epochs', type=int, default=10, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--pretrain', type=str, default="", metavar='N',
                help='number of epochs to train (default: 10)')
parser.add_argument('--save', type=str, default="", metavar='N',
                help='path to save learned model')
parser.add_argument('--name', type=str, default="gmmvae", metavar='N',
                help='number of epochs to train (default: 10)')
args = parser.parse_args()

timestr = time.strftime("%Y%m%d-%H%M%S")
init_logging("logs/"+timestr+"-"+args.name+".log")

# according to the released code, mnist data is multiplied by 0.02
# 255*0.02 = 5.1. transforms.ToTensor() coverts 255 -> 1.0
# so add a customized Scale transform to multiple by 5.1
mnist_train = MNIST('../dataset/mnist', train=True, download=True)
mnist_test = MNIST('../dataset/mnist', train=False)
train_loader = torch.utils.data.DataLoader(mnist_train, 
    batch_size=args.batch_size, shuffle=True, num_workers=0)
test_loader = torch.utils.data.DataLoader(mnist_test,
    batch_size=args.batch_size, shuffle=False, num_workers=0)

gmmae = GMMVAE(input_dim=784, z_dim=10, n_components=10, binary=False, alpha=args.alpha,
    encodeLayer=[500,500,2000], decodeLayer=[2000,500,500], activation="relu", dropout=0)
print(gmmae)
gmmae.load_pretrain(args.pretrain)

gmmae.fit(train_loader, train_loader, lr=args.lr, lr_stepwise=args.lr_stepwise, batch_size=args.batch_size, num_epochs=args.epochs)
if args.save!="":
    gmmae.save_model(args.save)
