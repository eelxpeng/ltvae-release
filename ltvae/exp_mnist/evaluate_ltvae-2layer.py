import sys
sys.path.append("..")
import os
# os.environ["JAVA_HOME"] = "/usr/local/packages/j2se-8"
import jnius_config
jnius_config.add_options('-Xms1024m', '-Xmx2048m')
jnius_config.add_classpath('../pltm.jar','../../JAR/colt/lib/colt.jar',
    '../../JAR/colt/lib/concurrent.jar','../../JAR/commons-cli-1.2/commons-cli-1.2.jar',
    '../../JAR/commons-math3-3.6.1/commons-math3-3.6.1.jar')

import torch
from torch.autograd import Variable
import torch.utils.data
from torchvision import datasets, transforms
import numpy as np
from lib.ltvae_pyltm_fixed_var import LTVAE
from lib.datasets import MNISTBinarized
import argparse

np.set_printoptions(threshold=np.nan)

parser = argparse.ArgumentParser(description='LTVAE MNIST Example')
parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('--samples', type=int, default=100, metavar='N',
                    help='number of samples for importance sampling (default: 10)')
parser.add_argument('--model', type=str, default="", metavar='N',
                    help='save model checkpoint')
parser.add_argument('--ltmodel', type=str, default="", metavar='N',
                    help='latent tree model file (bif)')
args = parser.parse_args()
use_cuda = torch.cuda.is_available()
train_loader = torch.utils.data.DataLoader(
    MNISTBinarized('../dataset/mnist', train=True, download=True),
    batch_size=args.batch_size, shuffle=True, num_workers=0)
test_loader = torch.utils.data.DataLoader(
    MNISTBinarized('../dataset/mnist', train=False),
    batch_size=args.batch_size, shuffle=False, num_workers=0)

# z_dim = 10
# ltvae = LTVAE(input_dim=784, z_dim=z_dim, binary=True,
#         encodeLayer=[500,500,2000], decodeLayer=[2000,500,500])
z_dim = 20
ltvae = LTVAE(input_dim=784, z_dim=z_dim, binary=True,
        encodeLayer=[400,400], decodeLayer=[400,400])
print("Loading model from %s..." % args.model)
ltvae.load_model(args.model)
if use_cuda:
    ltvae.cuda()

varNames = [None]*z_dim
for i in range(z_dim):
    varNames[i] = "z"+str(i+1)
ltvae.updateLatentTree(args.ltmodel, varNames)

# validate
ltvae.train()
loglikelihood = 0.0
for batch_idx, (inputs, _) in enumerate(test_loader):
    inputs = inputs.view(inputs.size(0), -1).float()
    if use_cuda:
        inputs = inputs.cuda()
    inputs = Variable(inputs)
    loglikelihood += torch.sum(ltvae.log_marginal_likelihood_estimate(inputs, args.samples))
ave = loglikelihood/len(test_loader.dataset)
print("mean loglikelihood: ", ave)
