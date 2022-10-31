# TubeDETR: Spatio-Temporal Video Grounding with Transformers

[Original Project](https://github.com/antoyang/TubeDETR)  

## Setup
Running Environment:   python=3.8  
```
conda activate tube1
```  
if want to install the environment:

```
conda install pytorch==1.8.0 torchvision==0.9.0 cudatoolkit=11.1 -c pytorch -c conda-forge
```

## Data Preprocessing
**annotation dir**:  /data/chaos/VG_big_format  
mainly the train.json, test.json and val.json.

## Training

**Chaos** To train on Chaos, run:
```
bash run.sh
```
if you want to do any modification, please replace some command in run.sh. The main **Runing Command**: 
```
CUDA_VISIBLE_DEVICES=0 CUBLAS_WORKSPACE_CONFIG=:16:8 python main.py --load=/data/chaos/models/VG/pretrained_resnet101_checkpoint.pth \
--combine_datasets=chaos --combine_datasets_val=chaos --dataset_config config/chaos.json --output_dir=output

```

## Evaluation
For evaluation only, simply run the same commands as for training with `--resume=CHECKPOINT --eval`. 
For this to be done on the test set, add `--test` (in this case predictions and attention weights are also saved).

**Chaos** Directly run eval.sh  
```
bash eval.sh
```
The **Eval Command**:
```
CUDA_VISIBLE_DEVICES=0 CUBLAS_WORKSPACE_CONFIG=:16:8 python main.py --resume=/output/Sun_Oct_23_16:01:25_2022/checkpoint.pth \
--eval --load=pretrained_resnet101_checkpoint.pth --combine_datasets=chaos --combine_datasets_val=chaos \
--dataset_config config/chaos.json --output_dir=output
```

## Result Dir
**Output_dir**  
/home/chaos/data/Chaos/video_grounding_old/code/TubeDETR/output   