#!/bin/bash
set -e
# # Run for training
# nproc_per_node: if training on a single machine, which will be the number of GPUs.
CUDA_VISIBLE_DEVICES=1 python3 -m torch.distributed.launch \
 --nproc_per_node=1 \
 scripts/train_net.py \
 --config-file "experiments/Chaos/e2e_STCAT_R101_Chaos.yaml" \
 --use-seed \
 OUTPUT_DIR ./checkpoints/output \
 TENSORBOARD_DIR ./checkpoints/output/tensorboard \
 INPUT.RESOLUTION 448   # TODO: Whether this parameter affects the training.

# Run for Testing
# python3 -m torch.distributed.launch \
# --nproc_per_node=8 \
# scripts/test_net.py \
# --config-file "experiments/VidSTG/e2e_STCAT_R101_VidSTG.yaml" \
# --use-seed \
# MODEL.WEIGHT /home/tiger/data/vidstg/checkpoints/stcat_res448/model_022500.pth \
# OUTPUT_DIR /home/tiger/data/vidstg/checkpoints/output \
# INPUT.RESOLUTION 320