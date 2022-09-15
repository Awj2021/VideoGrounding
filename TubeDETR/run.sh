#!bin/bash
CUDA_VISIBLE_DEVICES=1 CUBLAS_WORKSPACE_CONFIG=:16:8 python main.py --ema \
--load=pretrained_resnet101_checkpoint.pth --combine_datasets=vidstg --combine_datasets_val=vidstg \
--dataset_config config/vidstg.json --output-dir=output
