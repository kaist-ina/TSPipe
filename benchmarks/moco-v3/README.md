# MoCo v3
- Original code from https://github.com/facebookresearch/moco-v3

## Configure
- Setup `tspipe.yaml`.

## How to run
```bash
python main.py  -a=vit_small -b=128 --optimizer=lars --lr=1.5e-4 \
                --weight-decay=.1 --epochs=300 --warmup-epochs=40 \
                --stop-grad-conv1 --moco-m-cos --moco-t=0.2 \
                --tspipe-enable --ip=localhost --rank=0 --num-nodes=1 --tspipe-config=tspipe.yaml /datasets/imagenet
```