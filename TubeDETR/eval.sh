#!/bin/bash
set -e 
# Evaluation:
CUDA_VISIBLE_DEVICES=0 CUBLAS_WORKSPACE_CONFIG=:16:8 python main.py --resume=./output/Sun_Oct_23_16:01:25_2022/checkpoint.pth \
--eval --load=pretrained_resnet101_checkpoint.pth --combine_datasets=chaos --combine_datasets_val=chaos \
--dataset_config config/chaos.json --output_dir=output
