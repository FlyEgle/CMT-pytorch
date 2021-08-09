#!/bin/bash
cd /data/jiangmingchao/data/code/ImageClassification;
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -W ignore /data/jiangmingchao/data/code/ImageClassification/test.py \
--dist-url 'tcp://127.0.0.1:9966' \
--dist-backend 'nccl' \
--multiprocessing-distributed=1 \
--world-size=1 \
--rank=0 \
--test_file /data/jiangmingchao/data/dataset/imagenet/val_oss_imagenet_128w.txt \
--batch-size 128 \
--num-workers 48 \
--num-classes 1000 \
--input_size 184 \
--crop_size 160 \
--ape 0 \
--rpe 1 \
--pe_nd 0 \
--qkv_bias 1 \
--swin 0 \
--model_name cmtti \
--depth 12 \
--patch_size 32 \
--heads 12 \
--dim_head 64 \
--dim 768 \
--mlp_dim 3072 \
--dropout 0.1 \
--emb_dropout 0.1 \
--checkpoints-path /data/jiangmingchao/data/AICutDataset/transformers/CMT/cmt_tiny_160x160_300epoch_mixup_cutmix_adamw_all_wd_0.1_6e-3_dp/checkpoints/r50_accuracy_0.76806640625.pth \
--save_folder /data/jiangmingchao/data/AICutDataset/imagenet/r50_acc_result/
