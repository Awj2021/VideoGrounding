import torch
import os
import numpy as np
import ffmpeg
import argparse
from PIL import Image
import matplotlib.pyplot as plt

from main import get_args_parser
from models.transformer import build_transformer
from models.backbone import build_backbone
from models.tubedetr import TubeDETR
from models.postprocessors import PostProcessSTVG, PostProcess
from datasets.video_transforms import prepare, make_video_transforms
from util.misc import NestedTensor
import json
import ipdb
  
from tqdm import tqdm

def load_json(data_dir):
    """
    load the json file.
    args:
        data_dir: the dir of annotation.
    """
    with open(data_dir, 'r') as f:
        data = json.load(f)
    f.close()
    return data

parser = argparse.ArgumentParser(
    "TubeDETR training and evaluation script", parents=[get_args_parser()]
)

args = parser.parse_args()
device = args.device

# load model
backbone = build_backbone(args)
transformer = build_transformer(args)
anno_data = load_json('/data/chaos/VG_big_format/val_split_expanded_bounder.json')

model = TubeDETR(
    backbone,
    transformer,
    num_queries=args.num_queries,
    aux_loss=args.aux_loss,
    video_max_len=args.video_max_len_train,
    stride=args.stride,
    guided_attn=args.guided_attn,
    fast=args.fast,
    fast_mode=args.fast_mode,
    sted=args.sted,
)
model.to(device)
print("##model loaded")

postprocessors = {"chaos": PostProcessSTVG(), "bbox": PostProcess()}

# load checkpoint
assert args.load
checkpoint = torch.load(args.load, map_location="cpu")

print("*"*50)
print("## Loading the model....")

# if "model_ema" in checkpoint:
#     ipdb.set_trace()
#     if (
#         args.num_queries < 100 and "query_embed.weight" in checkpoint["model_ema"]
#     ):  # initialize from the first object queries
#         checkpoint["model_ema"]["query_embed.weight"] = checkpoint["model_ema"][
#             "query_embed.weight"
#         ][: args.num_queries]
#     if "transformer.time_embed.te" in checkpoint["model_ema"]:
#         del checkpoint["model_ema"]["transformer.time_embed.te"]
#     model.load_state_dict(checkpoint["model_ema"], strict=False)
# else:
if (
    args.num_queries < 100 and "query_embed.weight" in checkpoint["model"]
):  # initialize from the first object queries
    checkpoint["model"]["query_embed.weight"] = checkpoint["model"][
        "query_embed.weight"
    ][: args.num_queries]
if "transformer.time_embed.te" in checkpoint["model"]:
    del checkpoint["model"]["transformer.time_embed.te"]
model.load_state_dict(checkpoint["model"], strict=False)
print("checkpoint loaded")

# load video (with eventual start & end) & caption demo examples

# TODO: add the loop for all videos appeared int the annotation.
for video_info in tqdm(anno_data['videos']):
    video_id = video_info['video_id']
    tid = video_info['tid']
    start_frame, end_frame = video_info['start_frame'], video_info['end_frame']
    tube_start_frame, tube_end_frame = video_info['tube_start_frame'], video_info['tube_end_frame']
    captions = [video_info['caption']]  # caption.e.g., A car is driving on a road.
    vid_path = os.path.join('/data/chaos/videos_320x180', video_info['video_path'])
    probe = ffmpeg.probe(vid_path)     # get the info of a video.
    video_stream = next(
        (stream for stream in probe["streams"] if stream["codec_type"] == "video"), None
    )
    """num, denum = video_stream["avg_frame_rate"].split("/")
    video_fps = int(num) / int(denum)"""
    clip_start = (
       int(start_frame) / 25
    )
    clip_end = (
        int(end_frame) / 25
    )
    ss = clip_start
    t = clip_end - clip_start
    # ipdb.set_trace()
    extracted_fps = (
        min((args.fps * t), args.video_max_len) / t
    )  # actual fps used for extraction given that the model processes video_max_len frames maximum
    cmd = ffmpeg.input(vid_path, ss=ss, t=t).filter("fps", fps=extracted_fps)   # 开始，时长，以fps=5,抽取得到数据长度。
    out, _ = cmd.output("pipe:", format="rawvideo", pix_fmt="rgb24").run(
        capture_stdout=True, quiet=True
    )
    w = int(video_stream["width"])
    h = int(video_stream["height"])
    images_list = np.frombuffer(out, np.uint8).reshape([-1, h, w, 3])  # 将视频进行抽帧，并将所有帧保存为.npy文件。
    print("*"*50)
    print("### Processing the data as numpy.")
    assert len(images_list) <= args.video_max_len
    image_ids = [[k for k in range(len(images_list))]]

    # video transforms
    empty_anns = []  # empty targets as placeholders for the transforms
    placeholder_target = prepare(w, h, empty_anns)
    placeholder_targets_list = [placeholder_target] * len(images_list)
    transforms = make_video_transforms("test", cautious=True, resolution=args.resolution)
    images, targets = transforms(images_list, placeholder_targets_list)
    samples = NestedTensor.from_tensor_list([images], False)
    if args.stride:
        samples_fast = samples.to(device)
        samples = NestedTensor.from_tensor_list([images[:, :: args.stride]], False).to(
            device
        )
    else:
        samples_fast = None
    durations = [len(targets)]  # durations计数单位为帧。
    # ipdb.set_trace()
    print("*"*60)
    print("=== Start to test this video...")
    with torch.no_grad():  # forward through the model
        # encoder
        memory_cache = model(
            samples,
            durations,
            captions,
            encode_and_save=True,
            samples_fast=samples_fast,
        )
        # decoder
        outputs = model(
            samples,
            durations,
            captions,
            encode_and_save=False,
            memory_cache=memory_cache,
        )

        pred_steds = postprocessors["chaos"](outputs, image_ids, video_ids=[0])[
            0
        ]  # (start, end) in terms of image_ids
        orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0).to(device)
        results = postprocessors["bbox"](outputs, orig_target_sizes)
        vidstg_res = {}  # maps image_id to the coordinates of the detected box
        for im_id, result in zip(image_ids[0], results):
            vidstg_res[im_id] = {"boxes": [result["boxes"].detach().cpu().tolist()]}

        # TODO: add the ground-truth bboxes.
        ground_res = {}
        frame_ids = [int(start_frame) + int(id * extracted_fps) for id in image_ids[0]]
        # ipdb.set_trace()
        for im_id, frame_id in zip(image_ids[0], frame_ids):
            if int(tube_start_frame) <= frame_id < int(tube_end_frame):
                # ipdb.set_trace()
                ground_res[im_id] = {"boxes": [anno_data['trajectories'][video_id][str(tid)][str(frame_id)]["bbox"]]}
        # ipdb.set_trace()
        # create output dirs
        if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir)
        if not os.path.exists(os.path.join(args.output_dir, video_id)):
            os.makedirs(os.path.join(args.output_dir, video_id))
        # extract actual images from the video to process them adding boxes
        os.system(
            f'ffmpeg -i {vid_path} -ss {ss} -t {t} -qscale:v 2 -r {extracted_fps} {os.path.join(args.output_dir, video_id, "%05d.jpg")}'
        )

        print("*"*60)
        print("Starting to visualize these frames...")
        for img_id in image_ids[0]:
            # load extracted image
            img_path = os.path.join(
                args.output_dir,
                video_id,
                str(int(img_id) + 1).zfill(5) + ".jpg",     # 这样的填充方式。很可以！
            )
            img = Image.open(img_path).convert("RGB")
            imgw, imgh = img.size
            fig, ax = plt.subplots()
            ax.axis("off")
            ax.imshow(img, aspect="auto")

            if (
                pred_steds[0] <= img_id < pred_steds[1]
            ):  # add predicted box if the image_id is in the predicted start and end
                x1, y1, x2, y2 = vidstg_res[img_id]["boxes"][0]
                w = x2 - x1
                h = y2 - y1
                rect = plt.Rectangle(
                    (x1, y1), w, h, linewidth=2, edgecolor="#FAFF00", fill=False
                )
                ax.add_patch(rect)
            # ipdb.set_trace()
            if img_id in ground_res.keys():
                # 1. x, y, w, h.
                x1, y1, w, h = ground_res[img_id]["boxes"][0]
                rect1 = plt.Rectangle(
                    (x1, y1), w, h, linewidth=2, edgecolor="#054E9F", fill=False
                )

                # 2. x, y, x, y
                # x1, y1, x2, y2 = ground_res[img_id]["boxes"][0]
                # w = x2 - x1
                # h = y2 - y1
                # rect1 = plt.Rectangle(
                #     (x1, y1), w, h, linewidth=2, edgecolor="#054E9F", fill=False
                # )
                ax.add_patch(rect1)
            threshold = 50
            if len(captions[0]) <= threshold:
                ax.text(10,15, captions[0], fontsize=8, fontstyle='italic',color='red', backgroundcolor='black')
            elif threshold < len(captions[0]) <= threshold * 2:
                ax.text(10,15, captions[0][:threshold], fontsize=10, fontstyle='italic',color='red', backgroundcolor='black')
                ax.text(10,27, captions[0][threshold:threshold*2], fontsize=10, fontstyle='italic',color='red', backgroundcolor='black')
            elif len(captions[0]) > threshold * 2:
                ax.text(10,15, captions[0][:threshold], fontsize=10, fontstyle='italic',color='red', backgroundcolor='black')
                ax.text(10,27, captions[0][threshold:threshold*2], fontsize=10, fontstyle='italic',color='red', backgroundcolor='black')
                ax.text(10,39, captions[0][threshold*2 :-1], fontsize=10, fontstyle='italic',color='red', backgroundcolor='black')
            else:
                raise RuntimeError('Please check the lenth of caption: {} !!!'.format(len(captions[0])))
            fig.set_dpi(70)
            fig.set_size_inches(imgw / 70, imgh / 70)
            fig.tight_layout(pad=0)

            # save image with eventual box
            fig.savefig(
                img_path,
                format="jpg",
            )
            plt.close(fig)

        # save video with tube
        os.system(
            f"ffmpeg -r {extracted_fps} -pattern_type glob -i '{os.path.join(args.output_dir, video_id)}/*.jpg' -vf 'pad=ceil(iw/2)*2:ceil(ih/2)*2' -r {extracted_fps} -crf 25 -c:v libx264 -pix_fmt yuv420p -movflags +faststart {os.path.join(args.output_dir, video_id + '.mp4')}"
        )
