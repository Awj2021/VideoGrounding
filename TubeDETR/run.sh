#!bin/bash
CUDA_VISIBLE_DEVICES=1 python -m torch.distributed.launch --nproc_per_node=1 --use_env main.py --ema \
--load=pretrained_resnet101_checkpoint.pth --combine_datasets=vidstg --combine_datasets_val=vidstg \
--dataset_config config/vidstg.json --output-dir=output
