# ImageClassification

### 1. Introduction
This repo is only used for image classification task, such as imagenet. Include ddp training and inference, calculate the real acc and so on.

### 2. Enveriments
- python 3.7+
- pytorch 1.7.1 
- pillow
- apex 
- opencv-python

You can see this [repo](https://github.com/NVIDIA/apex) to find how to install the apex 

### 3. Training & Inference
- dataset prepare
    ```
    /data/home/imagenet/xxx.jpeg, 0
    /data/home/imagenet/xxx.jpeg, 1
    ...
    /data/home/imagenet/xxx.jpeg, 999
    ```
- training 
    1. Only used FP16 with bn FP32
        ```bash
        #!/bin/bash
        OMP_NUM_THREADS=1
        MKL_NUM_THREADS=1
        export OMP_NUM_THREADS
        export MKL_NUM_THREADS
        cd ImageClassification;
        CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -W ignore -m torch.distributed.launch --nproc_per_node 8 train_lanuch.py \
        --batch_size 512 \
        --num_workers 48 \
        --lr 1.6 \
        --max_epochs 90 \
        --warmup_epochs 5 \
        --num-classes 1000 \
        --input_size 256 \
        --crop_size 224 \
        --FP16 1 \
        --mode O2 \
        --apex 0 \
        --amp 0 \
        --train_file $train_file \
        --val_file $val_file \
        --log-dir $log_dir \
        --checkpoints-path $ckpt_dir
        ```
    2. Use Apex training
        ```bash
        #!/bin/bash
        OMP_NUM_THREADS=1
        MKL_NUM_THREADS=1
        export OMP_NUM_THREADS
        export MKL_NUM_THREADS
        cd ImageClassification;
        CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -W ignore -m torch.distributed.launch --nproc_per_node 8 train_lanuch.py \
        --batch_size 512 \
        --num_workers 48 \
        --lr 1.6 \
        --max_epochs 90 \
        --warmup_epochs 5 \
        --num-classes 1000 \
        --input_size 256 \
        --crop_size 224 \
        --FP16 0 \
        --mode O1 \
        --apex 1 \
        --amp 0 \
        --train_file $train_file \
        --val_file $val_file \
        --log-dir $log_dir \
        --checkpoints-path $ckpt_dir
        ```
    3. Use pytorch amp training
        ```bash
        #!/bin/bash
        OMP_NUM_THREADS=1
        MKL_NUM_THREADS=1
        export OMP_NUM_THREADS
        export MKL_NUM_THREADS
        cd ImageClassification;
        CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -W ignore -m torch.distributed.launch --nproc_per_node 8 train_lanuch.py \
        --batch_size 512 \
        --num_workers 48 \
        --lr 1.6 \
        --max_epochs 90 \
        --warmup_epochs 5 \
        --num-classes 1000 \
        --input_size 256 \
        --crop_size 224 \
        --FP16 1 \
        --mode O2 \
        --apex 0 \
        --amp 1 \
        --train_file $train_file \
        --val_file $val_file \
        --log-dir $log_dir \
        --checkpoints-path $ckpt_dir
        ```
- inference
    ```bash
    #!/bin/bash
    cd ImageClassification;
    CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -W ignore test.py \
    --dist-url 'tcp://127.0.0.1:9966' \
    --dist-backend 'nccl' \
    --multiprocessing-distributed=1 \
    --world-size=1 \
    --rank=0 \
    --test_file $test_file \
    --batch-size 128 \
    --num-workers 48 \
    --num-classes 1000 \
    --swin 0 \
    --checkpoints-path $ckpt_path \
    --save_folder $logits_folder
    ```
- calculate acc
```python 
python utils/calculate_acc.py --logits_file $logits_folder
```

