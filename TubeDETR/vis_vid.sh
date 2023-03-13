#!/bin/bash
set -e

python demo_vidstg.py \
--load=pretrained_resnet101_checkpoint.pth \
--resume=vidstgk2res224.pth \
--combine_datasets=vidstg \
--combine_datasets_val=vidstg \
--dataset_config config/vidstg.json \
--caption_example "an adult man plays a guitar on a stage." \
--video_example /home/chaos/data/Chaos/video_grounding_old/data/VidOR/video/1001/3783730077.mp4 \
--start_example=-1 \
--end_example=-1 \
--output_dir ./vis_vidstg