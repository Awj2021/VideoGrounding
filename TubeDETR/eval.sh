#!/bin/bash
set -e 
# Evaluation:
CUDA_VISIBLE_DEVICES=1 CUBLAS_WORKSPACE_CONFIG=:16:8 python main.py --ema --test --load=pretrained_resnet101_checkpoint.pth --combine_datasets=chaos --combine_datasets_val=chaos --dataset_config config/chaos.json --output-dir=output