python test_sdae-3layer.py --lr 0.0001 --pretrainepochs 300 --epochs 300 --save model/sdae-run-2.pt
python test_gmmvae-3layer.py --lr 0.0001 --lr-stepwise 0.01 --alpha 0.1 --epochs 300 --pretrain model/sdae-run-2.pt --save model/gmmvae-run-2.pt
