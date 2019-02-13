"""
python test_vae-2layer-binarize.py --lr 0.001 --epochs 300 --pretrain model/ae_mnist-2layer-binarized.pt
On MNIST achieves acc: 0.86540, nmi: 0.81587
"""
import sys
sys.path.append("..")
import torch
import torch.utils.data
from torchvision import datasets, transforms
import numpy as np
import argparse
from lib.vae import VAE
from lib.datasets import MNISTBinarized

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='VAE MNIST Example')
    parser.add_argument('--lr', type=float, default=0.001, metavar='N',
                        help='learning rate for training (default: 0.001)')
    parser.add_argument('--batch-size', type=int, default=256, metavar='N',
                        help='input batch size for training (default: 128)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--pretrain', type=str, default="", metavar='N',
                    help='number of epochs to train (default: 10)')
    args = parser.parse_args()
    
    train_loader = torch.utils.data.DataLoader(
        MNISTBinarized('../dataset/mnist', train=True, download=True),
        batch_size=args.batch_size, shuffle=True, num_workers=0)
    test_loader = torch.utils.data.DataLoader(
        MNISTBinarized('../dataset/mnist', train=False),
        batch_size=args.batch_size, shuffle=False, num_workers=0)

    vae = VAE(input_dim=784, z_dim=20, binary=True,
        encodeLayer=[400,400], decodeLayer=[400,400])
    print(vae)
    vae.load_model(args.pretrain)
    
    vae.fit(train_loader, test_loader, lr=args.lr, batch_size=args.batch_size,
        num_epochs=args.epochs, anneal=True)
    vae.save_model("model/vae-binarized.pt")
