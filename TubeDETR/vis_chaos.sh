#!/bin/bash
set -e

# Using one video as example.
# start_example: the start of clips. 
# end_example: the end of clips.
# caption: 
# - video: BBZREQCL, 
# ---- start: 2186
# ---- end: 2303
# ---- tube_start: 2217
# ---- tube_end: 2279
# ---- caption: A young girl with a pink mask is answering the reporter's question.
# ---- video_path: /data/chaos/videos_320x180/BBZREQCL.mp4

python demo_chaos.py --load=./output/Sun_Oct_23_16:01:25_2022/checkpoint0019.pth \
 --combine_datasets=chaos \
 --combine_datasets_val=chaos \
 --dataset_config config/chaos.json \
 --output_dir ./vis_chaos_new/