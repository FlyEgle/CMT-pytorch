#!/bin/bash
OMP_NUM_THREADS=1
MKL_NUM_THREADS=1
export OMP_NUM_THREADS
export MKL_NUM_THREADS
cd /data/jiangmingchao/data/code/ImageClassification;
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -W ignore -m torch.distributed.launch --nproc_per_node 8 train_lanuch.py \
--batch_size 512 \
--num_workers 48 \
--lr 1.6 \
--optimizer_name sgd \
--cosine 1 \
--model_name vit \
--max_epochs 120 \
--warmup_epochs 5 \
--num-classes 1000 \
--input_size 256 \
--crop_size 224 \
--weight_decay 1e-3 \
--depth 12 \
--patch_size 32 \
--heads 12 \
--dim_head 64 \
--dim 768 \
--mlp_dim 3072 \
--dropout 0.1 \
--emb_dropout 0.1 \
--grad_clip 0 \
--FP16 0 \
--mode O2 \
--amp 1 \
--apex 0 \
--train_file /data/jiangmingchao/data/dataset/imagenet/train_oss_imagenet_128w_sample_rate_0.2.txt \
--val_file /data/jiangmingchao/data/dataset/imagenet/val_oss_imagenet_128w.txt \
--log-dir /data/jiangmingchao/data/AICutDataset/transformers/vit_2k_sgd_1.6_cosine_120_5_wd_1e-3/log_dir \
--checkpoints-path /data/jiangmingchao/data/AICutDataset/transformers/vit_2k_sgd_1.6_cosine_120_5_wd_1e-3/checkpoints


# /data/jiangmingchao/data/dataset/imagenet/train_oss_imagenet_128w.txt


