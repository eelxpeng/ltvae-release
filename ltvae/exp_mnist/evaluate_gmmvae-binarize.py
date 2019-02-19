"""
python test_gmmvae.py --lr 0.0001 --lr-stepwise 0.01 --epochs 300 --pretrain model/sdae.pt
On MNIST achieves acc: 0.86540, nmi: 0.81587
"""
import sys
sys.path.append("..")
import torch
import torch.utils.data
from torch.autograd import Variable
from torchvision import datasets, transforms
import numpy as np
import argparse
from lib.gmmvae_fixed_var import GMMVAE
from lib.datasets import MNISTBinarized

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='VAE MNIST Example')
    parser.add_argument('--batch-size', type=int, default=256, metavar='N',
                        help='input batch size for training (default: 128)')
    parser.add_argument('--samples', type=int, default=100, metavar='N',
                    help='number of samples for importance sampling (default: 10)')
    parser.add_argument('--model', type=str, default="", metavar='N',
                        help='save model checkpoint')
    parser.add_argument('--alpha', type=float, default=1., metavar='N',
                    help='set value of alpha')
    args = parser.parse_args()
    
    train_loader = torch.utils.data.DataLoader(
        MNISTBinarized('../dataset/mnist', train=True, download=True),
        batch_size=args.batch_size, shuffle=True, num_workers=0)
    test_loader = torch.utils.data.DataLoader(
        MNISTBinarized('../dataset/mnist', train=False),
        batch_size=args.batch_size, shuffle=False, num_workers=0)

    gmmae = GMMVAE(input_dim=784, z_dim=20, n_components=10, binary=True, alpha=args.alpha,
        encodeLayer=[400,400], decodeLayer=[400,400], activation="relu", dropout=0)
    print(gmmae)
    
    gmmae.load_model(args.model)
    use_cuda = torch.cuda.is_available()
    if use_cuda:
        gmmae.cuda()
    # validate
    gmmae.train()
    loglikelihood = 0.0
    for batch_idx, (inputs, _) in enumerate(test_loader):
        inputs = inputs.view(inputs.size(0), -1).float()
        if use_cuda:
            inputs = inputs.cuda()
        inputs = Variable(inputs)
        loglikelihood += torch.sum(gmmae.log_marginal_likelihood_estimate(inputs, args.samples))
    ave = loglikelihood/len(test_loader.dataset)
    print("mean loglikelihood: ", ave)

