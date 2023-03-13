#!/bin/bash
set -e
# CUDA_VISIBLE_DEVICES=1 CUBLAS_WORKSPACE_CONFIG=:16:8 python main.py --ema \
# --load=pretrained_resnet101_checkpoint.pth --combine_datasets=vidstg --combine_datasets_val=vidstg \
# --dataset_config config/vidstg.json --output-dir=output

# Run Chaos dataset.
# TODO: Due to the pre-preparing the .npy file, so before regenerating the .npy file please removing them.
# cd /data/chaos/tube_extract_frames && rm -rf *.npy && cd -

# Preparing the config file. 
# Motify the config file. Especially the dir of Videos and Annotation.
# CUDA_VISIBLE_DEVICES=1 CUBLAS_WORKSPACE_CONFIG=:16:8 python main.py --ema --load=pretrained_resnet101_checkpoint.pth \
# --combine_datasets=chaos --combine_datasets_val=chaos --dataset_config config/chaos.json --output_dir=output


# TODO: Only use one sample for evaluation.
# FIXME: Please check the branch of git before training.
CUDA_VISIBLE_DEVICES=1 CUBLAS_WORKSPACE_CONFIG=:16:8 python -m torch.distributed.launch --nproc_per_node=1 --use_env --master_port 47769 main.py --ema \
--load=/data/chaos/models/VG/pretrained_resnet101_checkpoint.pth \
--combine_datasets=chaos --combine_datasets_val=chaos --dataset_config config/chaos.json --output_dir=output
