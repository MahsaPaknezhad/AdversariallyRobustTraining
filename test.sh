#!/bin/bash
source ~/.bashrc
conda activate reg_env

python main.py \
        --seed=48563 \
        --epsilon=0.0 \
        --setting='test' \
        --dataset='CIFAR10' \
        --output_dir='testing' \
        --data_dir='~/data' \
        --num_class=10 \
        --num_train=-1 \
        --num_val=-1 \
        --imsize=32 \
        --model='ResNet9' \
        --activation='celu' \
        --time=-1 \
        --grad_reg_lambda=10000 \
        --num_unlabeled_per_labeled=1 \
        --unlabeled_noise_std=0.35654 \
        --num_neighbor_per_anchor=1 \
        --neighbor_noise_std=0.035654 \
        --device='cuda' \
        --lr=1e-4 \
        --linear_epoch=5 \
        --num_epochs=200 \
        --batch_size=1 \
        --log_freq=10 \
        --adversarial=0 \
        --adv_ratio=0.0 \
        --resume=1 \
        --resume_epoch=-1\
        --jacobian=0 \
        --inject_noise=1
