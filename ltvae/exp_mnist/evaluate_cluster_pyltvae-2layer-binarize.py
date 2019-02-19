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
from torchvision.utils import save_image
from torchvision import datasets, transforms
import numpy as np
from lib.ltvae_pyltm_fixed_var import LTVAE
from lib.datasets import MNISTBinarized
from lib.utils import saveToArff
import argparse
import sys
import pdb
from lib.utils import Dataset
import scipy.io

np.set_printoptions(threshold=np.nan)

parser = argparse.ArgumentParser(description='LTVAE MNIST Example')
parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('--model', type=str, default="", metavar='N',
                    help='save model checkpoint')
parser.add_argument('--ltmodel', type=str, default="", metavar='N',
                    help='latent tree model file (bif)')
parser.add_argument('--name', type=str, default="mnist", metavar='N',
                    help='name of this run')
args = parser.parse_args()
use_cuda = torch.cuda.is_available()

train_loader = torch.utils.data.DataLoader(
    MNISTBinarized('../dataset/mnist', train=True, download=True),
    batch_size=args.batch_size, shuffle=False, num_workers=0)
test_loader = torch.utils.data.DataLoader(
    MNISTBinarized('../dataset/mnist', train=False),
    batch_size=args.batch_size, shuffle=False, num_workers=0)

z_dim = 20
ltvae = LTVAE(input_dim=784, z_dim=z_dim, binary=True,
        encodeLayer=[400,400], decodeLayer=[400,400])
print("Loading model from %s..." % args.model)
ltvae.load_model(args.model)

varNames = [None]*z_dim
for i in range(z_dim):
    varNames[i] = "z"+str(i+1)
ltvae.updateLatentTree(args.ltmodel, varNames)

# validate
ltvae.eval()
if use_cuda:
    ltvae.cuda()
images = []
latents = None
ylabels = None
for batch_idx, (inputs, labels) in enumerate(test_loader):
    images.append(inputs.reshape(-1, 28, 28))
    inputs = inputs.float()
    if use_cuda:
        inputs = inputs.cuda()
    inputs = Variable(inputs)
    z, _, _, _ = ltvae(inputs)
    x = z.data.cpu().numpy()
    y = labels.numpy()
    latent = ltvae.latentTree.latObj.inference(x)
    if batch_idx==0:
        latents = latent
        ylabels = y
    else:
        for j in range(len(latent)):
            latents[j] = np.concatenate((latents[j], latent[j]), axis=0)
        ylabels = np.concatenate((ylabels, y))

images = torch.cat(images)
ylabels = np.squeeze(ylabels)

from sklearn.metrics.cluster import normalized_mutual_info_score
def cluster_accuracy(assigments, labels):
    """
    Assume labels are from 0 - K-1
    """
    clusters = np.unique(assigments)
    classes = np.unique(labels)
    ave_acc = 0.0
    num_hit = 0
    for c in clusters:
        subassignments = assigments[assigments==c]
        sublabels = labels[assigments==c]
        counts = np.zeros(len(classes))
        for l in range(len(classes)):
            counts[l] = np.sum(sublabels==classes[l])
        cluster_label = classes[np.argmax(counts)]
        acc = np.mean(sublabels==cluster_label)
        num_hit += np.sum(sublabels==cluster_label)
        ave_acc += acc
        print("cluster %d -> label %d: acc=%f" % (c, cluster_label, acc))
    # ave_acc = ave_acc/(len(clusters))
    ave_acc = 1.0*num_hit/len(assigments)
    print("Average acc=%f" % ave_acc)

for j in range(len(latents)):
    latent = latents[j]
    num, cluster = latent.shape
    print("Facet: %d, # clusters: %d" % (j, cluster))
    assigments = np.argmax(latent, axis=1)
    nmi = normalized_mutual_info_score(ylabels, assigments)
    print("NMI=", nmi)
    cluster_accuracy(assigments, ylabels)


testset = datasets.MNIST('../dataset/mnist', train=False, transform=transforms.ToTensor())

# interprete the latents
def show_examples(dataset, probs, n, index):
    num, cluster = probs.shape
    images = None
    for i in range(cluster):
        idx_sorted = np.argsort(probs[:, i])[::-1]
        this_images = []
        for idx in idx_sorted[:n]:
            img, label = dataset[idx]
            this_images.append(img.unsqueeze(0))
        this_images = torch.cat(this_images)
        if images is None:
            images = this_images
        else:
            images = torch.cat([images, this_images])
    # pdb.set_trace()
    directory = "results/pyltvae/cluster/"
    if not os.path.exists(directory):
        os.makedirs(directory)
    save_image(images, directory+args.name+"-"+str(index)+".png", nrow=n)

for j in range(len(latents)):
    print("Facet: %d" % j)
    latent = latents[j]
    num, cluster = latent.shape
    for i in range(cluster):
        idx_sorted = np.argsort(latent[:, i])[::-1]
        print(ylabels[idx_sorted[:100]])
    show_examples(testset, latent, 10, j)

'''
conditional cluster examples
'''
# interprete the latents
def show_conditional_examples(dataset, probs, n, index):
    num, cluster = probs.shape
    images = None
    for i in range(cluster):
        idx_sorted = np.argsort(probs[:, i])[::-1].copy()
        this_images = dataset[idx_sorted[:n]]
        if images is None:
            images = this_images
        else:
            images = torch.cat([images, this_images])
    directory = "results/pyltvae/facets/"
    if not os.path.exists(directory):
        os.makedirs(directory)
    save_image(images, directory+args.name+"-"+str(index)+".png", nrow=n)

facets = [0, 1]
latent = latents[facets[0]]
assigments = np.argmax(latent, axis=1)
cond_latent = latents[facets[1]]
num, cluster = latent.shape
_, cond_cluster = cond_latent.shape
for j in range(cluster):
    images = []
    for i in range(num):
        if assigments[i]==j:
            img, label = testset[i]
            images.append(img.unsqueeze(0))
    images = torch.cat(images)
    this_latent = latent[assigments==j]
    this_cond_latent = cond_latent[assigments==j]
    show_conditional_examples(images, this_cond_latent, 10, j)
