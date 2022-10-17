import os
import json
from torch.utils.data import Dataset
from pathlib import Path
from .video_transforms import make_video_transforms, prepare
import time
import ffmpeg
import numpy as np
import random
import ipdb

class VideoModulatedSTGrounding(Dataset):
    def __init__(
        self,
        vid_folder,
        ann_file,
        logger,
        transforms,
        is_train=False,
        video_max_len=200,
        video_max_len_train=100,
        fps=5,
        tmp_crop=False,
        tmp_loc=True,
        stride=0,
    ):
        """
        :param vid_folder: path to the folder containing a folder "video"
        :param ann_file: path to the json annotation file
        :param logger: the log records.
        :param transforms: video data transforms to be applied on the videos and boxes
        :param is_train: whether training or not
        :param video_max_len: maximum number of frames to be extracted from a video
        :param video_max_len_train: maximum number of frames to be extracted from a video at training time
        :param fps: number of frames per second
        :param tmp_crop: whether to use temporal cropping preserving the annotated moment
        :param tmp_loc: whether to use temporal localization annotations
        :param stride: temporal stride k
        # :param invalid_ids: the invalid videos ids for filterring in __getitem__
        """
        self.vid_folder = vid_folder
        self.logger = logger
        print("loading annotations into memory...")
        tic = time.time()
        # TODO: if the annotation is not satisfied, please check with other code.
        self.annotations = json.load(open(ann_file, "r"))
        # print("Done (t={:0.2f}s)".format(time.time() - tic))
        self.logger.info("Done (t={:0.2f}s)".format(time.time() - tic))
        self._transforms = transforms
        self.is_train = is_train
        self.video_max_len = video_max_len
        self.video_max_len_train = video_max_len_train
        self.fps = fps
        self.tmp_crop = tmp_crop
        self.tmp_loc = tmp_loc
        self.vid2imgids = (
            {}
        )  # map video_id to [list of frames to be forwarded, list of frames in the annotated moment]
        self.stride = stride
        # self.videos = []
        for i_vid, video in enumerate(self.annotations["videos"]):
            ### Checking the size of videos.
            vid_path = os.path.join(self.vid_folder, str(video["video_path"]))
            probe = ffmpeg.probe(vid_path)
            if probe['streams'][0]['width'] != 1280 or probe['streams'][0]['height'] !=720:
                self.logger.info('=== Check the size of Video: {}'.format(vid_path))
                # print(video['video_path'])
                continue

            # self.videos.append(video)
            video_fps = video["fps"]  # used for extraction
            sampling_rate = fps / video_fps
            assert sampling_rate <= 1  # downsampling at fps
            start_frame = (
                video["start_frame"] if self.tmp_loc else video["tube_start_frame"]
            )
            end_frame = video["end_frame"] if self.tmp_loc else video["tube_end_frame"]
            frame_ids = [start_frame]
            for frame_id in range(start_frame, end_frame):
                # sample.
                if int(frame_ids[-1] * sampling_rate) < int(frame_id * sampling_rate):
                    frame_ids.append(frame_id)

            if len(frame_ids) > video_max_len:  # subsample at video_max_len
                frame_ids = [
                    frame_ids[(j * len(frame_ids)) // video_max_len]
                    for j in range(video_max_len)
                ]
            # 采样之后，只选择在tube_start_frame 和 tube_end_frame之间的帧。
            inter_frames = set(
                [
                    frame_id
                    for frame_id in frame_ids
                    if video["tube_start_frame"] <= frame_id < video["tube_end_frame"]
                ]
            )  # frames in the annotated moment
            self.vid2imgids[video["video_id"]] = [frame_ids, inter_frames]
        self.logger.info('=== Loading all the videos')

    def __len__(self) -> int:
        return len(self.annotations["videos"])

    def __getitem__(self, idx):
        """
        :param idx: int
        :return:
        images: a CTHW video tensor
        targets: list of frame-level target, one per frame, dictionary with keys image_id, boxes, orig_sizes
        tmp_target: video-level target, dictionary with keys video_id, qtype, inter_idx, frames_id, caption
        """
        video = self.annotations["videos"][idx]
        # video = self.videos[idx]
        caption = video["caption"]
        video_id = video["video_id"]
        video_original_id = video["original_video_id"]
        clip_start = video["start_frame"]  # included
        clip_end = video["end_frame"]  # excluded
        frame_ids, inter_frames = self.vid2imgids[video_id]
        trajectory = self.annotations["trajectories"][video_id][
            str(int(video["tid"]))
        ]
        # print(video_id)
        length = int(list(trajectory)[-1]) - int(list(trajectory)[0]) + 1 
        ### TODO: check the len(trajectory.keys()) as the keys is not continuous. 
        # assert len(list(trajectory)) == length, (
        #     print('Missing some frames in segment: {}/{}'.format(video_id, video_original_id))
        # )   
        # ffmpeg decoding
        vid_path = os.path.join(self.vid_folder, video["video_path"])
        video_fps = video["fps"]
        ss = clip_start / video_fps             # 帧开始的时间，
        t = (clip_end - clip_start) / video_fps # 相于开始帧，结束帧的相对时间
        try:
            cmd = ffmpeg.input(vid_path, ss=ss, t=t).filter("fps", fps=len(frame_ids) / t) # fps: len(frame_ids) / t: 单位时间内采样的帧数
            out, _ = cmd.output("pipe:", format="rawvideo", pix_fmt="rgb24").run(
                capture_stdout=True, quiet=True
            )
        except ffmpeg.Error as e:
            print('stdout:', e.stdout.decode('utf8'))
            print('stderr', e.stderr.decode('utf8'))
        w = video["width"] * 4 # 1280
        h = video["height"] * 4 # 720

        images_list = np.frombuffer(out, np.uint8).reshape([-1, h, w, 3]) # 问题确认：报错是因为out转换之后，不能整除进行转换。
        # ipdb.set_trace()
        assert len(images_list) == len(frame_ids)

        # prepare frame-level targets
        targets_list = []
        inter_idx = []  # list of indexes of the frames in the annotated moment
        for i_img, img_id in enumerate(frame_ids): # 采样后start, end_frame对应的帧
            if img_id in inter_frames: # 采样后tube对应的帧
                try:
                    anns = trajectory[
                        str(img_id)
                    ]  # dictionary with bbox [left, top, width, height] key
                except: 
                    self.logger.info("=== Sampling has Wrong: {} Missing frame: {}".format(video_id, str(img_id)))
                    raise ValueError('Please Checking!!!')
                anns = [anns]
                inter_idx.append(i_img)
            else:
                anns = []
            # ipdb.set_trace()
            target = prepare(w, h, anns)
            target["image_id"] = f"{video_id}_{img_id}"
            targets_list.append(target)

        # video spatial transform
        if self._transforms is not None:
            images, targets = self._transforms(images_list, targets_list)
        else:
            images, targets = images_list, targets_list

        if (
            inter_idx
        ):  # number of boxes should be the number of frames in annotated moment
            assert (
                len([x for x in targets if len(x["boxes"])])
                == inter_idx[-1] - inter_idx[0] + 1
            ), (len([x for x in targets if len(x["boxes"])]), inter_idx)

        # temporal crop
        if self.tmp_crop:
            p = random.random()
            if p > 0.5:  # random crop
                # list possible start indexes
                if inter_idx:
                    starts_list = [i for i in range(len(frame_ids)) if i < inter_idx[0]]
                else:
                    starts_list = [i for i in range(len(frame_ids))]

                # sample a new start index
                if starts_list:
                    new_start_idx = random.choice(starts_list)
                else:
                    new_start_idx = 0

                # list possible end indexes
                if inter_idx:
                    ends_list = [i for i in range(len(frame_ids)) if i > inter_idx[-1]]
                else:
                    ends_list = [i for i in range(len(frame_ids)) if i > new_start_idx]

                # sample a new end index
                if ends_list:
                    new_end_idx = random.choice(ends_list)
                else:
                    new_end_idx = len(frame_ids) - 1

                # update everything
                prev_start_frame = frame_ids[0]
                prev_end_frame = frame_ids[-1]
                frame_ids = [
                    x
                    for i, x in enumerate(frame_ids)
                    if new_start_idx <= i <= new_end_idx
                ]
                images = images[:, new_start_idx : new_end_idx + 1]  # CTHW
                targets = [
                    x
                    for i, x in enumerate(targets)
                    if new_start_idx <= i <= new_end_idx
                ]
                clip_start += frame_ids[0] - prev_start_frame
                clip_end += frame_ids[-1] - prev_end_frame
                if inter_idx:
                    inter_idx = [x - new_start_idx for x in inter_idx]

        if (
            self.is_train and len(frame_ids) > self.video_max_len_train
        ):  # densely sample video_max_len_train frames
            if inter_idx:
                starts_list = [
                    i
                    for i in range(len(frame_ids))
                    if inter_idx[0] - self.video_max_len_train < i <= inter_idx[-1]
                ]
            else:
                starts_list = [i for i in range(len(frame_ids))]

            # sample a new start index
            if starts_list:
                new_start_idx = random.choice(starts_list)
            else:
                new_start_idx = 0

            # select the end index
            new_end_idx = min(
                new_start_idx + self.video_max_len_train - 1, len(frame_ids) - 1
            )

            # update everything
            prev_start_frame = frame_ids[0]
            prev_end_frame = frame_ids[-1]
            frame_ids = [
                x for i, x in enumerate(frame_ids) if new_start_idx <= i <= new_end_idx
            ]
            images = images[:, new_start_idx : new_end_idx + 1]  # CTHW
            targets = [
                x for i, x in enumerate(targets) if new_start_idx <= i <= new_end_idx
            ]
            clip_start += frame_ids[0] - prev_start_frame
            clip_end += frame_ids[-1] - prev_end_frame
            if inter_idx:
                inter_idx = [
                    x - new_start_idx
                    for x in inter_idx
                    if new_start_idx <= x <= new_end_idx
                ]

        # video level annotations
        tmp_target = {
            "video_id": video_id,
            "qtype": video["qtype"],
            "inter_idx": [inter_idx[0], inter_idx[-1]]
            if inter_idx
            else [
                -100,
                -100,
            ],  # start and end (included) indexes for the annotated moment
            "frames_id": frame_ids,
            "caption": caption,
        }
        if self.stride:
            return images[:, :: self.stride], targets, tmp_target, images
        return images, targets, tmp_target


def build(image_set, args, logger):
    vid_dir = Path(args.chaos_vid_path)
    if args.test:
        ann_file = Path(args.chaos_ann_path) / f"test.json"
    elif image_set == "val":
        ann_file = Path(args.chaos_ann_path) / f"val.json"
    else:
        ann_file = (
            Path(args.chaos_ann_path) / f"train.json"
            if args.video_max_len_train == 200 or (not args.sted)
            else Path(args.chaos_ann_path) / f"train_{args.video_max_len_train}.json"
        )
    logger.info("anno_file: {}".format(ann_file))
    dataset = VideoModulatedSTGrounding(
        vid_dir,
        ann_file,
        logger,
        transforms=make_video_transforms(
            image_set, cautious=True, resolution=args.resolution
        ),
        is_train=image_set == "train",
        video_max_len=args.video_max_len,
        video_max_len_train=args.video_max_len_train,
        fps=args.fps,
        tmp_crop=args.tmp_crop and image_set == "train",
        tmp_loc=args.sted,
        stride=args.stride,
    )
    return dataset
