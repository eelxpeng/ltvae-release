# run experiments for testset loglikelihood
RUNID=1
# pretrain using autoencoder
echo "####pretrain using autoencoder"
python test_ae-2layer-binarize.py --save model/ae_mnist-2layer-binarized-run-$RUNID.pt

# train and evaluate VAE
echo "####train and evaluate VAE"
python test_vae-2layer-binarize.py --lr 0.001 --epochs 300 --pretrain model/ae_mnist-2layer-binarized-run-$RUNID.pt --save model/vae-2layer-binarized-run-$RUNID.pt
python evaluate_vae-2layer-binarize.py --model model/vae-2layer-binarized-run-$RUNID.pt --samples 5000

# train and evaluate IWAE
echo "####train and evaluate IWAE"
python test_iwae-2layer.py --lr 0.001 --epochs 300 --pretrain model/ae_mnist-2layer-binarized-run-$RUNID.pt --save model/iwae-binarized-run-$RUNID.pt
python evaluate_iwae-2layer.py --model model/iwae-binarized-run-$RUNID.pt --samples 5000

# train and evaluate GMMVAE
echo "####train and evaluate GMMVAE"
python test_gmmvae_fixed_var_binarize.py --lr 0.001 --lr-stepwise 0.01 --epochs 500 --pretrain model/iwae-binarized-run-$RUNID.pt --save model/gmmvae-binarized-run-$RUNID
python evaluate_gmmvae-binarize.py --model model/gmmvae-binarized-run-$RUNID --samples 5000

# train and evaluate LTVAE
echo "####train LTVAE"
python test_pyltvae-2layer.py --lr 0.0001 --stepwise_em_lr 0.01 --epochs 20 --everyepochs 10 --model model/iwae-binarized-run-$RUNID.pt --name pyltvae-2layer-binarized-run-$RUNID --save model/pyltvae-2layer-binarized-run-$RUNID
echo "####finetune LTVAE"
python finetune_pyltvae-2layer.py --lr 0.0001 --stepwise_em_lr 0.01 --epochs 500 --model model/pyltvae-2layer-binarized-run-$RUNID.pt --ltmodel model/pyltvae-2layer-binarized-run-$RUNID.bif --save model/pyltvae-2layer-binarized-finetune-run-$RUNID
echo "####evaluate LTVAE"
python evaluate-ltvae-2layer.py --model model/pyltvae-2layer-binarized-finetune-run-$RUNID.pt --ltmodel model/pyltvae-2layer-binarized-finetune-run-$RUNID.bif --samples 5000
