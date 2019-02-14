'''
python test_sdae --lr 0.0001 --pretrainepochs 300 --epochs 300
'''
import sys
sys.path.append("..")
import torch
import torch.utils.data
from torchvision import datasets, transforms
import numpy as np
import argparse
from lib.stackedDAE import StackedDAE
from lib.datasets import MNIST
from lib.utils import init_logging
import logging
import time

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='VAE MNIST Example')
    parser.add_argument('--lr', type=float, default=0.1, metavar='N',
                        help='learning rate for training (default: 0.001)')
    parser.add_argument('--batch-size', type=int, default=256, metavar='N',
                        help='input batch size for training (default: 128)')
    parser.add_argument('--pretrainepochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--save', type=str, default="", metavar='N',
                        help='path to save learned model')
    parser.add_argument('--name', type=str, default="sdae", metavar='N',
                help='number of epochs to train (default: 10)')
    args = parser.parse_args()

    timestr = time.strftime("%Y%m%d-%H%M%S")
    init_logging("logs/"+timestr+"-"+args.name+".log")
    
    # according to the released code, mnist data is multiplied by 0.02
    # 255*0.02 = 5.1. transforms.ToTensor() coverts 255 -> 1.0
    # so add a customized Scale transform to multiple by 5.1
    train_loader = torch.utils.data.DataLoader(
        MNIST('../dataset/mnist', train=True, download=True),
        batch_size=args.batch_size, shuffle=True, num_workers=0)
    test_loader = torch.utils.data.DataLoader(
        MNIST('../dataset/mnist', train=False),
        batch_size=args.batch_size, shuffle=False, num_workers=0)

    sdae = StackedDAE(input_dim=784, z_dim=10, binary=False,
        encodeLayer=[500,500,2000], decodeLayer=[2000,500,500], activation="relu", 
        dropout=0)
    logging.info(sdae)
    sdae.pretrain(train_loader, test_loader, lr=args.lr, batch_size=args.batch_size, 
        num_epochs=args.pretrainepochs, corrupt=0.2, loss_type="mse")
    sdae.fit(train_loader, test_loader, lr=args.lr, num_epochs=args.epochs, corrupt=0.2, loss_type="mse")
    if args.save!="":
        sdae.save_model(args.save)
