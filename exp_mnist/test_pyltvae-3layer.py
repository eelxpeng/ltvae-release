'''
python test_pyltvae_fixed_var.py --lr 0.0001 --stepwise_em_lr 0.01 --epochs 20 --everyepochs 10 --model model/sdae.pt --name pyltvae
acc: 0.86318, nmi: 0.81245
'''
import sys
sys.path.append("..")
sys.path.append("../../pyLTM")
import os
# os.environ["JAVA_HOME"] = "/usr/local/packages/j2se-8"
import jnius_config
jnius_config.add_options('-Xms1024m', '-Xmx2048m')
jnius_config.add_classpath('../pltm_no_state_ops.jar','../../JAR/colt/lib/colt.jar',
    '../../JAR/colt/lib/concurrent.jar','../../JAR/commons-cli-1.2/commons-cli-1.2.jar',
    '../../JAR/commons-math3-3.6.1/commons-math3-3.6.1.jar')

import torch
import torch.utils.data
from torchvision import datasets, transforms
import numpy as np
# from lib.ltvae_pyltm import LTVAE
from lib.ltvae_pyltm_fixed_var import LTVAE
from lib.plt_pyltm import learn_latentTree, learn_pouchlatentTree
from lib.datasets import MNIST
from lib.utils import saveToArff
import argparse
from lib.utils import init_logging
import logging
import time
import warnings
warnings.filterwarnings('error')

parser = argparse.ArgumentParser(description='LTVAE MNIST Example')
parser.add_argument('--lr', type=float, default=0.001, metavar='N',
                    help='learning rate for training (default: 0.001)')
parser.add_argument('--stepwise_em_lr', type=float, default=0.01, metavar='N',
                    help='learning rate for training (default: 0.001)')
parser.add_argument('--alpha', type=float, default=1., metavar='N',
                    help='set the covariance')
parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('--epochs', type=int, default=10, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--everyepochs', type=int, default=1, metavar='N',
                    help='number of epochs to learn structure (default: 10)')
parser.add_argument('--model', type=str, default="", metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--ltmodel', type=str, default="", metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--save', type=str, default="", metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--name', type=str, default="ltvae", metavar='N',
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

z_dim = 10
ltvae = LTVAE(input_dim=784, z_dim=z_dim, binary=False,
        encodeLayer=[500,500,2000], decodeLayer=[2000,500,500],
        alpha=args.alpha)
logging.info("Loading pretrain from %s..." % args.model)
ltvae.load_model(args.model)

varNames = [None]*z_dim
for i in range(z_dim):
    varNames[i] = "z"+str(i+1)

model_file = "tmp/mnist-pyltm-3layer-init.bif"
init_file = None

X, Y = ltvae.encodeBatch(train_loader, True)
X = X.cpu().numpy()
Y = Y.cpu().numpy()
n_components = 10
ltvae.initialize(X, Y, n_components, model_file, "mnist", varNames)

# update ltvae first 
logging.info("LTVAE training for for first %d epochs..." % args.everyepochs)
ltvae.updateLatentTree(model_file, varNames)
ltvae.fit(train_loader, test_loader, lr=args.lr, stepwise_em_lr=args.stepwise_em_lr,
    batch_size=args.batch_size, num_epochs=args.everyepochs, visualize=False)
model_file = "tmp/mnist-pyltm-3layer-epoch-0.bif"
ltvae.latentTree.latObj.saveAsBif(model_file)

for epoch in range(1, args.epochs+1):
    logging.info("Overall Epoch #%d" % (epoch))
    # project data to z
    logging.info("Encoding data...")
    ltvae.eval()
    encoded = ltvae.encodeBatch(train_loader).cpu().numpy()
    # sample 10% data
    num = encoded.shape[0]
    idx = np.random.choice(num, int(0.2*num), replace=False)
    data = encoded[idx]
    logging.info("Save encoding to arff...")
    saveToArff(data, "mnist", varNames, "tmp/mnist.arff")
    logging.info("Learn latent tree structure...")
    # if epoch==1:
    #     model_file = "tmp/mnist-plt-3layer-epoch-1.bif"
    # else:
    #     init_file = model_file
    #     model_file = ("tmp/mnist-plt-3layer-epoch-%d.bif" % epoch)
    #     start = time.time()
    #     learn_latentTree(model_file, "settings.xml", "tmp/mnist.arff", initpath=init_file)
    #     end = time.time()
    #     logging.info("Structure learning cost %.2f" % (end-start))

    start = time.time()
    init_file = model_file
    model_file = ("tmp/mnist-pyltm-3layer-epoch-%d.bif" % epoch)
    # learn_latentTree(model_file, "settings.xml", "tmp/mnist.arff", initpath=init_file)
    learn_pouchlatentTree(model_file, "../settings.xml", "tmp/mnist.arff", initpath=init_file)
    end = time.time()
    logging.info("Structure learning cost %.2f" % (end-start))

    ltvae.updateLatentTree(model_file, varNames)
    logging.info("LTVAE training...")
    ltvae.fit(train_loader, test_loader, lr=args.lr, stepwise_em_lr=args.stepwise_em_lr,
        batch_size=args.batch_size, num_epochs=args.everyepochs, visualize=False)
    ltvae.latentTree.latObj.saveAsBif(model_file)
    savepath = ("tmp/ltvae-3layer-epoch-%d.pt" % epoch)
    torch.save(ltvae.state_dict(), savepath)

if args.save != "":
	ltvae.save_model(args.save)