# Knowledge Distillation (Soft Target)
- Original code from https://github.com/AberHu/Knowledge-Distillation-Zoo

## Configure
- Setup `tspipe.yaml` : TSPipe related configuration.


## How to run
1. Pre-train base model (No support for TSPipe)
    ```bash
    python train_base.py \
        --img_root=/datasets/imagenet --save_root=./results/base/ \
        --data_name=imagenet100 \
        --net_name=vit_large \
        --num_class=100 \
        --note=base-i100-vit-large
    ```
    ```
    python train_base.py \
        --img_root=/datasets/imagenet --save_root=./results/base/ \
        --data_name=imagenet100 \
        --net_name=resnet152 \
        --num_class=100 --batch_size=16 --epochs=0 \
        --note=base-i100-resnet152
    ```

2. Perform knowledge distillation
    ```bash
    python train_kd.py \
        --img_root=/datasets/imagenet --save_root=./results/st/ \
        --t_model=./results/base/base-i100-vit-large/model_best.pth.tar \
        --s_init=./results/base/base-i100-resnet152/initial_r152.pth.tar \
        --kd_mode=st --lambda_kd=0.1 --t_name=vit_large --s_name=resnet152 \
        --T=4.0 --data_name=imagenet100 --num_class=100 --batch_size=16 \
        --tspipe-enable --tspipe-config=tspipe.yaml --num-node=1 --rank=0 --ip=localhost \
        --note=kd-run
    ```
