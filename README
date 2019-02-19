# Latent Tree Variational Autoencoder 

This repository is associated with the following paper:

Xiaopeng Li, Zhourong Chen, Leonard K.M. Poon and Nevin L. Zhang. Learning Latent Superstructures in Variational Autoencoders for Deep Multidimensional Clustering. ICLR 2019.

## Environment Requirements

* python3.6

* pytorch>=0.4.0

* numpy

* scipy

* sklearn

* pyjnius >= 1.1.1

## Project Folder Structure

This project is built upon two other projects: [PLTM-EAST](https://github.com/kmpoon/pltm-east) and [pyLTM](https://github.com/eelxpeng/pyLTM).

PLTM-EAST is a Java implementation of PLTM-EAST algorithm to do structure learning for Gaussian/Pouch latent tree models, originally proposed by Poon et al. The original code has been modified for this project, and pltm.jar is put under `ltvae/`. pltm.jar has some JAR dependencies, and all dependencies have been put under `JAR/`.

PyLTM is a Python implementation of latent tree models for convenient integration with other part of the code in this project. The original project of pyLTM will be likely to be improved. A version of pyLTM is under `pyLTM/` for compatibility of this project.

The main code for this project is under `ltvae/`. The folder structure of the whole project should be have `ltvae/`, `pyLTM/` and `JAR/` under the same root folder.

## Project Description

Latent Tree VAE consists of both parameter learning and structure learning. The proposed joint parameter learning algorithm is StepwiseEM. The proposed structure learning is conducted every several epochs of parameter learning.

A special case of LTVAE without structure learning is LTVAE-GMM, which has a fixed Gaussian mixture structure (with one single y variable). For this special case, we simplify the code in `ltvae/lib/gmmvae_fixed_var.py`. The joint learning algorithm is the same as LTVAE, except that the structure will not be learned.

The full version of LTVAE with structure learning is in `ltvae/lib/ltvae_pyltm_fixed_var.py`

## Running

The dataset will be downloaded under `dataset/`. The experiment for MNIST dataset need to be conducted under `exp_mnist/`.

To run experiments of testset loglikelihood with stochastically binarized MNIST dataset, simply run

```console
cd ltvae/exp_mnist/

bash run-experiment-loglikelihood.sh
```
It includes all experiments related to VAE, IWAE, LTVAE-GMM and LTVAE.

To run experiments of clustering with 3 layers of encoder (784-500-500-2000-10)and decoder structure (10-2000-500-500-784), simply run
```console
bash run-gmmvae-3layer.sh
```
for LTVAE-GMM without structure learning, or run
```console
bash run-pyltvae-3layer.sh
```
for LTVAE with structure learning.

To evaluate qualitative results for LTVAE models with two facets, run
```console
python evaluate_cluster_pyltvae-2layer-binarize.py --model [LTVAE .pt file] --ltmodel [LTM .bif file]
```
for stochastic binarized model, or run
```console
python evaluate_cluster_pyltvae-2layer.py --model [LTVAE .pt file] --ltmodel [LTM .bif file]
```
for standard MNIST dataset model.

For example, an LTVAE model is provided under `saved_model/`. To evaluate the qualitative results, simply run
```console
python evaluate_cluster_pyltvae-2layer.py --model saved_model/ltvae-2layer-finetune.pt --ltmodel saved_model/mnist-plt-2layer-finetune.bif
```


