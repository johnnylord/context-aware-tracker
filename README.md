# Context-Aware Multiple Object Tracking with Dynamic Matching Strategy

Context-aware tracker (CAT) is a multiple object tracker (MOT) extending tracking scenario from 2D space to 3D space, and using body orientation to maintain stable and discriminative feature presentation of each object. Without further ado, CAT can track objects really well even when **objects are occluded by others or have similar appearance**.  

[![example.png](https://i.imgur.com/SkPezTH.png)](https://www.youtube.com/watch?v=Vz7h-lxVEfc)

## Download NTU-MOTD Dataset
- Download RGB videos and depth maps with semantic labels (bounding box, body keypoints, object mask)
```bash
$ cd resource/
$ wget -O dataset.tar.gz https://www.dropbox.com/s/5q1ep3bo5wn5uvr/dataset.tar.gz?dl=1
$ tar -xzvf dataset.tar.gz
```
- Download ground truth mot2d tracking result
```bash
$ wget -O groundtruth.tar.gz https://www.dropbox.com/s/aqfd1wics3krtsz/groundtruth.tar.gz?dl=1
$ tar -xzvf groundtruth.tar.gz
```

## How to run my code
In this section, I will show you how to perform tracking with CAT tracker on specific video dataset.

### 1. Install dependent packages
```bash
# Python3.7.1
$ pip install -r requirements.txt
```
### 2. Run CAT tracker
Run the CAT MOT tracker (you can change tracker option to run different tracker)
```bash
$ python main.py \
    --video resource/dataset/5p_sa_pm_up.msv \
    --output run/5p_sa_pm_up.msv \
    --tracker CAT \
    [--birdeye] [--verbose]
```
### 3. Evaluate with MOT2D metric
```bash
# Process all video at once
$ python run.py --input resource/dataset --output result/cat --tracker CAT
$ python script/mot2deval.py --gt resource/groundtruth/ --pred result/cat
```

## Experiment Result
- Overall
```
            IDF1   IDP   IDR  Rcll   Prcn GT MT PT ML   FP   FN IDs   FM  MOTA  MOTP IDt IDa IDm
UMA         42.2% 41.8% 42.6% 96.3% 94.5% 64 64  0  0 1640 1095 750  548 88.2% 0.179 403 128  11
JDE         58.1% 59.5% 56.6% 93.0% 98.0% 64 64  0  0 554  2056 442  680 89.7% 0.191 120 207   0
DeepMOT     43.0% 43.5% 40.5% 91.9% 99.2% 64 64  0  0 227  2388 296  595 90.1% 0.147  91 201   0
SORT        37.9% 38.9% 36.9% 94.0% 99.1% 64 63  1  0 251  1782 415  459 91.7% 0.007  25 371   0
DeepSORT    75.7% 75.7% 75.6% 96.5% 97.0% 64 63  1  0 897  1023 128  260 93.1% 0.024  78  32   2
CAT         84.6% 84.4% 84.5% 97.3% 97.4% 64 64  0  0 766   794  54  216 94.5% 0.034  26  22   0
```
- Ablation Study (Tracking with different information)
```
                IDF1   IDP   IDR  Rcll   Prcn GT MT PT ML  FP  FN IDs   FM  MOTA  MOTP IDt IDa IDm
CAT(w/o D+O)    78.7% 78.6% 78.5% 96.9% 97.2% 64 64  0  0 830 913 104  254 93.7% 0.034  61  30   2
CAT(w/o D)      79.4% 79.3% 79.3% 97.0% 97.2% 64 64  0  0 832 899 104  250 93.8% 0.034  63  28   2
CAT(w/o O)      84.2% 84.1% 84.1% 97.3% 97.4% 64 64  0  0 762 801  56  219 94.5% 0.033  25  24   0
CAT             84.6% 84.4% 84.5% 97.3% 97.4% 64 64  0  0 766 794  54  216 94.5% 0.034  26  22   0
```
- Ablation Study (Affinity Measurement Method BIAS on appearance)
```
            IDF1   IDP   IDR  Rcll   Prcn GT MT PT ML  FP  FN IDs   FM  MOTA  MOTP IDt IDa IDm
CAT(BIAS)   85.9% 85.8% 85.9% 97.3% 97.4% 64 64  0  0 777 804  56  218 94.5% 0.034  25  22   0
CAT         84.6% 84.4% 84.5% 97.3% 97.4% 64 64  0  0 766 794  54  216 94.5% 0.034  26  22   0
```
- **CAT Tracker**
```
                 IDF1   IDP   IDR  Rcll  Prcn GT MT PT ML  FP  FN IDs   FM  MOTA  MOTP IDt IDa IDm
3p_da_pm_pp.msv 98.3% 97.8% 98.9% 98.9% 97.8%  3  3  0  0  36  17   0    3 96.7% 0.025   0   0   0
3p_da_pm_up.msv 92.9% 92.6% 93.2% 98.9% 98.2%  3  3  0  0  29  18   1    4 97.1% 0.026   0   1   0
3p_da_um_pp.msv 98.8% 98.8% 98.9% 98.9% 98.9%  3  3  0  0  19  18   2    6 97.7% 0.027   1   0   0
3p_da_um_up.msv 98.8% 98.8% 98.8% 98.8% 98.8%  3  3  0  0  17  16   0    4 97.6% 0.029   0   0   0
3p_sa_pm_pp.msv 89.3% 89.6% 89.0% 98.1% 98.8%  3  3  0  0  19  30   1    7 96.9% 0.023   0   1   0
3p_sa_pm_up.msv 98.4% 98.5% 98.2% 98.2% 98.5%  3  3  0  0  23  28   0    6 96.7% 0.033   0   0   0
3p_sa_um_pp.msv 82.5% 82.6% 82.3% 98.2% 98.5%  3  3  0  0  24  28   3    5 96.5% 0.022   3   0   0
3p_sa_um_up.msv 99.1% 98.9% 99.4% 99.4% 98.9%  3  3  0  0  17   9   0    1 98.3% 0.019   0   0   0
5p_da_pm_pp.msv 97.0% 96.9% 97.1% 97.1% 96.9%  5  5  0  0  67  63   0   10 93.9% 0.037   0   0   0
5p_da_pm_up.msv 97.1% 96.7% 97.5% 97.5% 96.7%  5  5  0  0  70  51   0   15 94.2% 0.041   0   0   0
5p_da_um_pp.msv 77.2% 77.5% 76.9% 97.2% 98.1%  5  5  0  0  41  59   8    8 94.9% 0.031   2   4   0
5p_da_um_up.msv 72.7% 72.2% 73.3% 97.6% 96.2%  5  5  0  0  80  50   5   12 93.5% 0.031   1   4   0
5p_sa_pm_pp.msv 92.5% 91.8% 92.2% 95.8% 96.4%  5  5  0  0  78  91   2   33 92.2% 0.038   0   2   0
5p_sa_pm_up.msv 69.4% 68.8% 68.5% 93.2% 95.6%  5  5  0  0  92 145   9   64 88.5% 0.050   4   4   0
5p_sa_um_pp.msv 58.1% 58.4% 57.9% 96.1% 96.9%  5  5  0  0  68  85  11   16 92.5% 0.042   8   2   0
5p_sa_um_up.msv 51.8% 51.7% 51.8% 95.8% 95.8%  5  5  0  0  86  86  12   22 91.0% 0.047   7   4   0
OVERALL         84.6% 84.4% 84.5% 97.3% 97.4% 64 64  0  0 766 794  54  216 94.5% 0.034  26  22   0
```

## How to collect dataset
### Hardwares
I use **intel-realsense L515** lidar camera to record videos. It will give you RGB raw video and depth map in grayscale. Connect L515 lidar camera to your computer and use the helper script `script/record.py` to record the video.
```bash
# An `example.msv` directory will be generated with RGB video (`video.mp4`) and depth video (`depth.mp4`) in it
$ python script/record.py --model l515 --export --output example.msv
```
### Softwares
To get other information, such as bounding boxes, keypoints, reid features of each person, and etc. Use the helper script `script/play.py` to postprocess the `msv` video. (All AI models are deployed elsewhere and communicate with `script/play.py` with grpc protocal,( **SOMETIMES I MAY CLOSE THE SERVICES**)
```bash
$ python script/play.py --path exampl.msv --process --export
```
> all processed metadata information will be stored in `bodyposes.pkl` under target `msv` video
### MSV Video Player
I develop a multiple stream video (MSV) player, which can easily add different streaming sources in the video container (msv container). I consider the processed metadata as a streaming source. Here is an example of how to use MSV player.
```python
import cv2
from multimedia.container.msv import MultiStreamVideo

video = MultiStreamVideo('example.msv')

# Display video
while True:
    frames = video.read()
    if frames['video'] is None:
        break
    video_frame = frames['video']
    depth_frame = frames['depth']
    # bodyposes = frames['bodyposes'] # If you have processed the video

    cv2.imshow("RGB", video_frame)
    cv2.imshow("Depth", depth_frame)
    # ...
```

## How to annotate ground truth tracking result
All `gt.txt` in the provided dataset are labeled in the following process.
1. Run the simplest tracker, `SORT`, to get an initial mot2d result file
2. Remap the track IDs in the result file with `script/remap.py`
