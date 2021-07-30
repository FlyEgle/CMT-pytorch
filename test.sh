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
--swin 0 \
--checkpoints-path /data/jiangmingchao/data/AICutDataset/transformers/R50_2k_sgd_1.6_cosine_120_5/checkpoints/r50_accuracy_0.6207682291666666.pth \
--save_folder /data/jiangmingchao/data/AICutDataset/imagenet/r50_acc_result/


# /data/jiangmingchao/data/dataset/imagenet/val_oss_imagenet_128w.txt

