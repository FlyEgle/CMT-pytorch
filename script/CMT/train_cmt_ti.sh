# #!/bin/bash
# OMP_NUM_THREADS=1
# MKL_NUM_THREADS=1
# export OMP_NUM_THREADS
# export MKL_NUM_THREADS
# cd /data/jiangmingchao/data/code/ImageClassification;
# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -W ignore -m torch.distributed.launch --nproc_per_node 8 train_lanuch.py \
# --batch_size 512 \
# --num_workers 48 \
# --lr 1.6 \
# --optimizer_name "sgd" \
# --tf_optimizer 1 \
# --cosine 1 \
# --model_name cmtti \
# --max_epochs 120 \
# --warmup_epochs 5 \
# --num-classes 1000 \
# --input_size 184 \
# --crop_size 160 \
# --weight_decay 1e-4 \
# --grad_clip 1 \
# --max_grad_norm 5.0 \
# --FP16 0 \
# --qkv_bias 1 \
# --ape 0 \
# --rpe 1 \
# --pe_nd 0 \
# --mode O2 \
# --amp 1 \
# --apex 0 \
# --train_file /data/jiangmingchao/data/dataset/imagenet/train_oss_imagenet_128w_sample_rate_0.2-0.4.txt \
# --val_file /data/jiangmingchao/data/dataset/imagenet/val_oss_imagenet_128w.txt \
# --log-dir /data/jiangmingchao/data/AICutDataset/transformers/CMT/cmt_tiny_160x160_300epoch_mixup_cutmix_0.2-0.4/log_dir \
# --checkpoints-path /data/jiangmingchao/data/AICutDataset/transformers/CMT/cmt_tiny_160x160_300epoch_mixup_cutmix_0.2-0.4/checkpoints


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
--optimizer_name "sgd" \
--tf_optimizer 1 \
--cosine 1 \
--model_name cmtti \
--max_epochs 300 \
--warmup_epochs 5 \
--num-classes 1000 \
--input_size 184 \
--crop_size 160 \
--weight_decay 1e-4 \
--grad_clip 1 \
--max_grad_norm 5.0 \
--FP16 0 \
--qkv_bias 1 \
--ape 0 \
--rpe 1 \
--pe_nd 0 \
--mode O2 \
--amp 1 \
--apex 0 \
--train_file /data/jiangmingchao/data/dataset/imagenet/train_oss_imagenet_128w_sample_rate_0.2-0.4.txt \
--val_file /data/jiangmingchao/data/dataset/imagenet/val_oss_imagenet_128w.txt \
--log-dir /data/jiangmingchao/data/AICutDataset/transformers/CMT/cmt_tiny_160x160_300epoch_mixup_cutmix_0.2-0.4_2/log_dir \
--checkpoints-path /data/jiangmingchao/data/AICutDataset/transformers/CMT/cmt_tiny_160x160_300epoch_mixup_cutmix_0.2-0.4_2/checkpoints




# /data/jiangmingchao/data/dataset/imagenet/train_oss_imagenet_128w.txt
# /data/jiangmingchao/data/dataset/imagenet/train_oss_imagenet_128w_sample_rate_0.2.txt
# "/data/jiangmingchao/data/dataset/imagenet/train_oss_imagenet_128w_sample_rate_0.2-0.4.txt
# /data/jiangmingchao/data/dataset/imagenet/train_oss_imagenet_128w_sample_rate_0.5-0.8.txt
# /data/jiangmingchao/data/dataset/imagenet/train_oss_imagenet_128w_sample_rate_0.8_end.txt
