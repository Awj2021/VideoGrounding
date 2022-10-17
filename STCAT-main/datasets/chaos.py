import os
import re
import shutil
import json
from copy import deepcopy
import torch
import random

from tqdm import tqdm
import torch.utils.data as data
import numpy as np
from PIL import Image
import ffmpeg

from torchvision.transforms import ToTensor, ToPILImage
from utils.bounding_box import BoxList
from .data_utils import SENTENCE_SPLIT_REGEX, make_chaos_input_clip
from .words import replace_dict
import ipdb

def merge_anno(rootdir):
    """
    Args:
        rootdir: the dataset folder
    """
    origin_dir = os.path.join(rootdir,'annotation') 
    output_dir = os.path.join(rootdir,'bbox_annos')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for root, _, files in os.walk(origin_dir):
        if len(files) == 0:
            continue
        for file in files:
            src = os.path.join(root,file)
            dst = os.path.join(output_dir,file)
            shutil.copy(src,dst)

    return os.listdir(output_dir)


def clean_anno(data):
    """
    Args:
        data : all the groundtruth data item
    Usage:
        clean the language description, modify the wrong words
    """
    word_pt = re.compile(r'[A-Za-z]',re.S)
    check = lambda word : bool(len(re.findall(word_pt,word)))
    max_len = 0
    for idx in range(len(data)):
        data_item = data[idx]
        sentence = data_item['description']
        words = SENTENCE_SPLIT_REGEX.split(sentence.strip())
        words = [w.lower() for w in words if (len(w) > 0 and w!=' ')]   ## do not include space as a token
        words = list(filter(check,words))
        for widx, word in enumerate(words):
            if word in replace_dict:
                words[widx] = replace_dict[word]
        data[idx]['description'] = ' '.join(words)
        max_len = max(max_len,len(words))
    # print(max_len)
    return data


class ChaosDataset(data.Dataset):

    def __init__(self, cfg, split, transforms=None) -> None:
        super(ChaosDataset,self).__init__()
        self.cfg = cfg.clone()
        self.split = split
        self.transforms = transforms

        self.data_dir = cfg.DATA_DIR
        self.frame_dir = os.path.join(self.data_dir,'frame') # 没有这个数据集
        # self.sent_anno_dir = os.path.join(self.data_dir,'')
        # TODO: Every json file.  /home/chaos/data/Chaos/video_grounding_old/data/VidSTG/training
        if split == 'train':
            self.bbox_anno_dir = os.path.join(self.data_dir,f'{split}ing')
        elif split == 'test':
            self.bbox_anno_dir = os.path.join(self.data_dir, 'validation')
        elif split == 'val': # val part in the training folder.
            self.bbox_anno_dir = os.path.join(self.data_dir, 'validation')
        else:
            raise ValueError(f'please check the name of split: {split}')
        print(f'======= Loading the dataset of {split} ==========')

        self.sent_file = os.path.join(self.data_dir, f'{split}_annotations.json')
        self.epsilon = 1e-10
        
        all_gt_data = self.load_data()  # In fact, this function aims to generate the train.json/test.json/val.json file, 
                                         # these files all exist in the annotation folder.
        self.all_gt_data = clean_anno(all_gt_data)  # clean the sentence annotation
        self.vocab = None
        
        if cfg.DATA_TRUNK is not None:
            self.all_gt_data = self.all_gt_data[:cfg.DATA_TRUNK]

        if cfg.MODEL.USE_LSTM:
            vocab_pth = os.path.join(cfg.GLOVE_DIR,'vocab.pth')
            self.vocab = torch.load(vocab_pth)

    def folder_dir(self, file_format='json'):
        """
        return the dir of every video. Like 0000/123456.mp4 or 0000/123456.json
        """
        file_rel_dir = {}
        files = os.listdir(self.bbox_anno_dir) # e.g. ./data/VidSTG/training
        for num in files:
            jsons = os.listdir(os.path.join(self.bbox_anno_dir, num))
            for single_file in jsons:
                file_no = single_file.split('.')[0]
                file_rel_dir[file_no] = os.path.join(num, f'{file_no}.json') if file_format=='json' else os.path.join(num, f'{file_no}.mp4')
        return file_rel_dir 
        
    def check_vocab(self):
        bad_words = set()
        for idx in tqdm(range(len(self.all_gt_data))):
            data_item = self.all_gt_data[idx]
            sentence = data_item['description']
            words = sentence.strip().split()
            for w in words:
                word_idx = self.vocab.stoi.get(w.lower(), 400000)
                if word_idx == 400000:
                    bad_words.add(w)
        print(bad_words)
    
    def get_video_info(self,index):
        video_info = {}
        data_item = self.all_gt_data[index]
        video_info['height'] = data_item['height']
        video_info['width'] = data_item['width']
        return video_info

    def load_frames(self, data_item, videos_dir, load_video=True):
        # TODO: still need to add the folder of every video.
        # videos_dir return the video_name's folder dir.
        video_name = data_item['vid']
        frame_ids = data_item['frame_ids']
        patience = 20
        
        if load_video:
            # TODO: remember to modity the dir of videos.
            video_dir = '/home/chaos/data/Chaos/dataset/video'
            video_path = os.path.join(video_dir, videos_dir[video_name]) 
            h, w = data_item['height'], data_item['width']
            succ_flag = False
            for _ in range(patience):
                # try:
                try:
                    out, _ = (
                        ffmpeg
                        .input(video_path)
                        .output('pipe:', format='rawvideo', pix_fmt='rgb24')
                        .run(capture_stdout=True, quiet=True)
                    )
                    frames = np.frombuffer(out, np.uint8).reshape([-1, h, w, 3])
                    succ_flag = True
                    if succ_flag:
                        break
                except ffmpeg.Error as e:
                    print('stdiyt', e.stdout.decode('utf8'))
                    print('stderr', e.stderr.decode('utf8'))
                # except Exception:
                #     print(video_name)
                    
            if not succ_flag:
                print("=======here=====", video_name)
                raise RuntimeError("Load Video Error")
            
            frames = frames[frame_ids]
            frames = [ToTensor()(frame) for frame in frames]
            frames = torch.stack(frames)
        else:
            frames = []
            frame_paths = [os.path.join(self.frame_dir, video_name, \
                    'img_{:05d}.jpg'.format(fid)) for fid in frame_ids]
            for img_path in frame_paths:
                img = Image.open(img_path).convert("RGB")
                frames.append(ToTensor()(img))

            frames = torch.stack(frames)   # T * C * H * W
        
        return frames

    def __getitem__(self, index: int):
        """
        Usage:
            In training, sample a random clip from video
            In testing, chunk the video to a set of clips
        """
        video_data = deepcopy(self.all_gt_data[index]) 

        data_item = make_chaos_input_clip(self.cfg, self.split, video_data)
        
        videos_dir = self.folder_dir(file_format='mp4')
        frames = self.load_frames(data_item, videos_dir)   # T * C * H * W
        
        # load the sampled gt bounding box
        frame_ids = data_item['frame_ids']
        temp_gt = data_item['gt_temp_bound']
        action_idx = np.where(data_item['actioness'])[0]
        start_idx, end_idx = action_idx[0], action_idx[-1]
        bbox_idx = [frame_ids[idx] - temp_gt[0] for idx in range(start_idx,end_idx + 1)]
        bboxs = torch.from_numpy(data_item['bboxs'][bbox_idx]).reshape(-1, 4)
        assert bboxs.shape[0] == len(action_idx)

        w, h = data_item['width'], data_item['height']
        bboxs = BoxList(bboxs, (w, h), 'xyxy')
        
        sentence = data_item['description']
        sentence = sentence.lower()
        input_dict = {'frames': frames, 'boxs': bboxs, 'text': sentence, \
                'actioness' : data_item['actioness']}

        if self.transforms is not None:
            input_dict = self.transforms(input_dict)
        
        targets = {
            'item_id' : data_item['item_id'],
            'frame_ids' : data_item['frame_ids'],
            'actioness' : torch.from_numpy(data_item['actioness']) ,
            'start_heatmap' : torch.from_numpy(data_item['start_heatmap']),
            'end_heatmap' : torch.from_numpy(data_item['end_heatmap']),
            'boxs' : input_dict['boxs'],
            'qtype' : data_item['qtype'],
            'img_size' : input_dict['frames'].shape[2:],
            'ori_size' : (h, w)
        }
        
        return input_dict['frames'], sentence, targets

    def __len__(self) -> int:
        return len(self.all_gt_data)

    def load_data(self):
        """
        Prepare the Input Data Cache and the evaluation data groundtruth
        """
        
        cache_dir = os.path.join(self.data_dir,'data_cache')
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)
            
        dataset_cache = os.path.join(cache_dir, f'vidstd-{self.split}-input.cache')
        # Used For Evaluateion
        gt_anno_cache = os.path.join(cache_dir, f'vidstd-{self.split}-anno.cache')
        
        # if exists, then return.
        if os.path.exists(dataset_cache):
            data = torch.load(dataset_cache)
            return data

        gt_data, gt_anno = [], []
        vstg_anno = self.make_data_pairs(self.sent_file) # sent_file. e.g. train_annotations.json

        for anno_id in tqdm(vstg_anno):  
            gt_file = vstg_anno[anno_id]
            #The spoiled data
            if len(gt_file['target_bboxs']) != gt_file['temp_gt']['end_fid']\
                    - gt_file['temp_gt']['begin_fid'] + 1:
                continue
            
            if gt_file['ori_temp_gt']['end_fid'] < gt_file['used_segment']['begin_fid'] or \
                gt_file['ori_temp_gt']['begin_fid'] > gt_file['used_segment']['end_fid']:
                continue
            
            video_name = gt_file['vid']
            # video_fps = gt_file['fps']
            # sampling_rate = fps / video_fps
            start_fid = gt_file['used_segment']['begin_fid']
            end_fid = gt_file['used_segment']['end_fid']

            temp_gt_begin = gt_file['ori_temp_gt']['begin_fid']
            temp_gt_end = min(gt_file['ori_temp_gt']['end_fid'],end_fid)

            assert len(gt_file['target_bboxs']) == temp_gt_end - temp_gt_begin + 1
                    
            frame_ids = []
            for frame_id in range(start_fid, end_fid + 1):
                frame_ids.append(frame_id)
            
            actioness = np.array([int(fid <= temp_gt_end and fid >= temp_gt_begin) \
                                    for fid in frame_ids]) 
            
            # prepare the temporal heatmap
            action_idx = np.where(actioness)[0]
            start_idx, end_idx = action_idx[0], action_idx[-1]
            
            start_heatmap = np.ones(actioness.shape) * self.epsilon
            pesudo_prob = (1 - (start_heatmap.shape[0] - 3) * self.epsilon - 0.5) / 2
            
            start_heatmap[start_idx] = 0.5
            if start_idx > 0:
                start_heatmap[start_idx-1] = pesudo_prob
            if start_idx < actioness.shape[0] - 1:
                start_heatmap[start_idx+1] = pesudo_prob

            end_heatmap = np.ones(actioness.shape) * self.epsilon
            end_heatmap[end_idx] = 0.5
            if end_idx > 0:
                end_heatmap[end_idx-1] = pesudo_prob
            if end_idx < actioness.shape[0] - 1:
                end_heatmap[end_idx+1] = pesudo_prob

            bbox_array = []
            for idx in range(len(gt_file['target_bboxs'])):
                bbox = gt_file['target_bboxs'][idx]
                x1, y1, x2, y2 = bbox['xmin'], bbox['ymin'], bbox['xmax'], bbox['ymax']
                bbox_array.append(np.array([x1,y1,x2,y2]))
            
            bbox_array = np.array(bbox_array)
            assert bbox_array.shape[0] == temp_gt_end - temp_gt_begin + 1
            
            gt_bbox_dict = {fid : bbox_array[fid - temp_gt_begin].tolist() \
                    for fid in range(temp_gt_begin, temp_gt_end + 1)}
            
            gt_item = {
                'item_id' : gt_file['id'],
                'vid' : video_name,
                'bboxs' : gt_bbox_dict,
                'description' : gt_file['sentence']['description'],
                'qtype' : gt_file['qtype'],
                'gt_temp_bound' : [temp_gt_begin, temp_gt_end],
                'segment_bound' : [start_fid, end_fid],
            }

            item = {
                'item_id' : gt_file['id'],
                'vid' : video_name,
                'frame_ids' : frame_ids,
                'width' : gt_file['width'],
                'height' : gt_file['height'],
                'start_heatmap': start_heatmap,
                'end_heatmap': end_heatmap,
                'actioness': actioness,
                'bboxs' : bbox_array,
                'gt_temp_bound' : [temp_gt_begin, temp_gt_end],
                'segment_bound' : [start_fid, end_fid],
                'qtype' : gt_file['qtype'],
                'description' : gt_file['sentence']['description'],
                'object' : gt_file['target_category']
            }
            
            gt_data.append(item)
            gt_anno.append(gt_item)
        
        random.shuffle(gt_data)
        torch.save(gt_data, dataset_cache)
        torch.save(gt_anno, gt_anno_cache)
        return gt_data

    def make_data_pairs(self,anno_file):
        """
        Args:
            anno_file: the origin vid-stg annos
        Usage:
            merge temporal gt and spatial gt
        """
        pair_cnt = 0
        spoiled = set()
        print(f"Prepare {self.split} Data")

        vstg_anno_dir = os.path.join(self.data_dir, 'vstg_annos')
        vstg_anno_path = os.path.join(vstg_anno_dir, self.split + '.json')
        
        if os.path.exists(vstg_anno_path):
            print(f"Load Anno Json from {vstg_anno_path}")
            with open(vstg_anno_path, 'r') as fr:
                vstg_anno = json.load(fr)
            return vstg_anno
            
        if not os.path.exists(vstg_anno_dir):
            os.makedirs(vstg_anno_dir)
        
        with open(anno_file,'r') as fr:
            sent_annos = json.load(fr)

        def get_bbox(bboxs,tid):
            for bbox in bboxs:
                if bbox['tid'] == tid:
                    return bbox
            return None

        vstg_anno = {}
        
        # TODO: If using the chaos dataset, to modify.
        file_rel_dir = self.folder_dir(file_format='json')
        for anno in tqdm(sent_annos):
            data_pairs = {}
            data_pairs['vid'] = anno['vid']
            data_pairs['fps'] = anno['fps']
            data_pairs['used_segment'] = anno['used_segment']
            data_pairs['width'] = anno['width']
            data_pairs['height'] = anno['height']
            data_pairs['ori_temp_gt'] = deepcopy(anno['temporal_gt'])
            data_pairs['frame_count'] = anno['used_segment']['end_fid'] - \
                        anno['used_segment']['begin_fid'] + 1

            data_pairs['temp_gt'] = deepcopy(anno['temporal_gt'])
            data_pairs['temp_gt']['begin_fid'] = anno['temporal_gt']['begin_fid'] - \
                    anno['used_segment']['begin_fid']
            data_pairs['temp_gt']['end_fid'] = anno['temporal_gt']['end_fid'] - \
                    anno['used_segment']['begin_fid']
            data_pairs['temp_gt']['end_fid'] = min(data_pairs['frame_count']-1,\
                    data_pairs['temp_gt']['end_fid'])
            # TODO: Find the original json file.
            # ipdb.set_trace()
            bbox_anno_path = os.path.join(self.bbox_anno_dir, file_rel_dir[anno['vid']])
            # bbox_anno_path = os.path.join(self.bbox_anno_dir, anno['vid']+'.json')
            with open(bbox_anno_path,'r') as fr:
                bbox_annos = json.load(fr)

            for sent_type in ['captions','questions']:
                for descrip_sent in anno[sent_type]:
                    data_pair = deepcopy(data_pairs)
                    data_pair['id'] = pair_cnt
                    data_pair['qtype'] = 'declar' if sent_type == 'captions' else 'inter'
                    data_pair['sentence'] = descrip_sent
                    target_id = data_pair['sentence']['target_id']
                    data_pair['target_category'] = get_bbox(anno['subject/objects'],target_id)['category']

                    trajectories = bbox_annos['trajectories']
                    data_pair['target_bboxs'] = [] 
                    start_idx = anno['temporal_gt']['begin_fid']
                    end_idx = min(anno['temporal_gt']['end_fid']+1,anno['frame_count'])

                    for idx in range(start_idx,end_idx):
                        frame_bboxs = trajectories[idx]
                        bbox = get_bbox(frame_bboxs,target_id)
                        if bbox is None:
                            if idx != end_idx - 1:
                                data_pair['target_bboxs'].append({})
                                spoiled.add(pair_cnt)
                            else:
                                # add the last bbox annotation
                                data_pair['target_bboxs'].append(data_pair['target_bboxs'][-1].copy())
                        else:
                            data_pair['target_bboxs'].append(bbox['bbox'])
                    
                    vstg_anno[pair_cnt] = data_pair
                    pair_cnt += 1
        
        print(f'Spoiled pair : {len(spoiled)}')
        print(f'{self.split} pair number : {pair_cnt}')
        # ipdb.set_trace()
        with open(vstg_anno_path, 'w') as fw:
            json.dump(vstg_anno, fw)
        
        return vstg_anno, file_rel_dir 
