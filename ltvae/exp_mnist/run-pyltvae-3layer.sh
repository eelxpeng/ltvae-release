python test_sdae-3layer.py --lr 0.0001 --pretrainepochs 300 --epochs 300 --save model/sdae-run-1.pt
python test_pyltvae-3layer.py --lr 0.0001 --stepwise_em_lr 0.01 --alpha 1. --epochs 20 --everyepochs 10 --model model/sdae-run-1.pt --name pyltvae-3layer --save model/pyltvae-3layer-run-1.pt
