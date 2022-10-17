#!/bin/bash
# CUDA_VISIBLE_DEVICES=1 CUBLAS_WORKSPACE_CONFIG=:16:8 python main.py --ema \
# --load=pretrained_resnet101_checkpoint.pth --combine_datasets=vidstg --combine_datasets_val=vidstg \
# --dataset_config config/vidstg.json --output-dir=output

# Run Chaos dataset.
# Preparing the config file. 
# Motify the config file. Especially the dir of Videos and Annotation.
CUDA_VISIBLE_DEVICES=1 CUBLAS_WORKSPACE_CONFIG=:16:8 python main.py --ema --load=pretrained_resnet101_checkpoint.pth --combine_datasets=chaos --combine_datasets_val=chaos --dataset_config config/chaos.json --output-dir=output
# Evaluation:
# CUDA_VISIBLE_DEVICES=1 CUBLAS_WORKSPACE_CONFIG=:16:8 python main.py --ema --test --load=pretrained_resnet101_checkpoint.pth --combine_datasets=chaos --combine_datasets_val=chaos --dataset_config config/chaos.json --output-dir=output
