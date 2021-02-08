# Context-Aware Multiple Object Tracking with Dynamic Matching Strategy

Context-aware tracker (CAT) is a multiple object tracker (MOT) extending tracking scenario from 2D space to 3D space, and using body orientation to maintain stable and discriminative feature presentation of each object. Without further ado, CAT can track objects really well even when **objects are occluded by others or have similar appearance**.  

[![example.png](https://i.imgur.com/nWQPFh9.png)](https://www.youtube.com/watch?v=QA1_ft5JmS4)

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
            IDF1   IDP   IDR  Rcll   Prcn GT MT PT ML  FP   FN IDs   FM  MOTA  MOTP IDt IDa IDm
SORT        37.9% 38.9% 36.9% 94.0% 99.1% 64 63  1  0 251 1782 415  459 91.7% 0.007  25 371   0
DeepSORT    75.7% 75.7% 75.6% 96.5% 97.0% 64 63  1  0 897 1023 128  260 93.1% 0.024  78  32   2
CAT         82.0% 81.8% 82.0% 97.3% 97.3% 64 64  0  0 791  803  60  213 94.4% 0.033  31  24   0
```
- **SORT Tracker**
```
                 IDF1   IDP   IDR  Rcll   Prcn GT MT PT ML  FP   FN IDs   FM  MOTA  MOTP IDt IDa IDm
3p_da_pm_pp.msv 37.8% 38.7% 37.1% 95.9% 100.0%  3  3  0  0   0   65  17   17 94.9% 0.001   0  17   0
3p_da_pm_up.msv 37.6% 38.3% 36.9% 96.1%  99.9%  3  3  0  0   1   63  15   20 95.1% 0.005   0  15   0
3p_da_um_pp.msv 59.4% 60.1% 58.8% 97.7%  99.8%  3  3  0  0   3   39   9   13 97.0% 0.005   0   9   0
3p_da_um_up.msv 55.3% 56.0% 54.6% 96.9%  99.4%  3  3  0  0   8   43  11   12 95.5% 0.007   0  11   0
3p_sa_pm_pp.msv 43.6% 44.4% 42.9% 96.1%  99.5%  3  3  0  0   8   63  14   19 94.7% 0.009   0  14   0
3p_sa_pm_up.msv 26.5% 27.1% 25.9% 95.5%  99.8%  3  3  0  0   3   70  17   20 94.3% 0.008   0  17   0
3p_sa_um_pp.msv 32.1% 32.6% 31.6% 96.7%  99.8%  3  3  0  0   3   52  15   12 95.5% 0.003   1  14   0
3p_sa_um_up.msv 38.7% 39.0% 38.4% 97.5%  98.9%  3  3  0  0  16   38  12   11 95.6% 0.003   2  10   0
5p_da_pm_pp.msv 36.5% 37.8% 35.3% 92.6%  99.1%  5  5  0  0  18  159  32   37 90.3% 0.007   0  32   0
5p_da_pm_up.msv 29.8% 30.8% 28.8% 92.3%  98.5%  5  5  0  0  29  161  41   45 88.9% 0.012   2  37   0
5p_da_um_pp.msv 35.8% 36.8% 34.9% 94.5%  99.7%  5  5  0  0   7  117  27   28 92.9% 0.007   3  24   0
5p_da_um_up.msv 38.6% 39.5% 37.7% 93.5%  98.0%  5  5  0  0  39  134  30   30 90.2% 0.005   1  29   0
5p_sa_pm_pp.msv 35.0% 36.0% 33.7% 91.7%  99.1%  5  5  0  0  19  181  36   50 89.2% 0.006   1  35   0
5p_sa_pm_up.msv 39.1% 40.8% 36.7% 87.5%  99.6%  5  4  1  0   7  268  62   70 84.3% 0.014   4  44   0
5p_sa_um_pp.msv 31.2% 32.3% 30.2% 92.3%  98.7%  5  5  0  0  27  168  38   36 89.3% 0.009   4  33   0
5p_sa_um_up.msv 36.9% 37.8% 36.0% 92.1%  96.8%  5  5  0  0  63  161  39   39 87.2% 0.013   7  30   0
OVERALL         37.9% 38.9% 36.9% 94.0%  99.1% 64 63  1  0 251 1782 415  459 91.7% 0.007  25 371   0
```
- **DeepSORT Tracker**
```
                 IDF1   IDP   IDR  Rcll  Prcn GT MT PT ML  FP   FN IDs   FM  MOTA  MOTP IDt IDa IDm
3p_da_pm_pp.msv 98.2% 97.6% 98.7% 98.7% 97.6%  3  3  0  0  39   20   0    4 96.3% 0.014   0   0   0
3p_da_pm_up.msv 84.8% 84.5% 85.2% 98.6% 97.7%  3  3  0  0  37   23   2    6 96.2% 0.014   0   2   0
3p_da_um_pp.msv 98.8% 98.9% 98.7% 98.8% 98.9%  3  3  0  0  18   21   2    6 97.6% 0.016   1   0   0
3p_da_um_up.msv 97.4% 97.5% 97.3% 98.3% 98.5%  3  3  0  0  21   24   3    6 96.5% 0.020   2   0   0
3p_sa_pm_pp.msv 76.3% 76.7% 75.9% 97.4% 98.4%  3  3  0  0  25   41   2    9 95.7% 0.015   1   2   1
3p_sa_pm_up.msv 53.8% 53.8% 53.7% 97.6% 97.8%  3  3  0  0  35   38   5    7 95.0% 0.024   3   2   0
3p_sa_um_pp.msv 59.6% 60.0% 59.3% 97.6% 98.8%  3  3  0  0  19   38   7    8 95.9% 0.013   5   1   0
3p_sa_um_up.msv 85.8% 85.7% 86.0% 98.8% 98.4%  3  3  0  0  24   18   4    4 96.9% 0.011   3   0   0
5p_da_pm_pp.msv 97.0% 97.0% 97.0% 97.0% 97.0%  5  5  0  0  65   65   0    9 93.9% 0.025   0   0   0
5p_da_pm_up.msv 85.9% 85.5% 86.2% 97.1% 96.3%  5  5  0  0  78   61   1   18 93.3% 0.029   0   1   0
5p_da_um_pp.msv 87.2% 87.6% 86.8% 97.0% 98.0%  5  5  0  0  43   63   6   12 94.7% 0.021   1   3   0
5p_da_um_up.msv 73.5% 73.0% 74.0% 96.7% 95.4%  5  5  0  0  97   68  13   18 91.4% 0.024   9   3   0
5p_sa_pm_pp.msv 54.0% 53.7% 53.7% 94.5% 95.5%  5  5  0  0  97  120  17   38 89.3% 0.033  11   5   0
5p_sa_pm_up.msv 58.5% 58.5% 57.2% 90.3% 94.4%  5  4  1  0 115  208  30   73 83.6% 0.048  20   6   0
5p_sa_um_pp.msv 54.3% 54.7% 54.0% 95.3% 96.6%  5  5  0  0  74  102  15   19 91.3% 0.031  11   3   1
5p_sa_um_up.msv 59.7% 59.7% 59.7% 94.5% 94.6%  5  5  0  0 110  113  21   23 88.1% 0.039  11   4   0
OVERALL         75.7% 75.7% 75.6% 96.5% 97.0% 64 63  1  0 897 1023 128  260 93.1% 0.024  78  32   2
```
- **CAT Tracker**
```
                 IDF1   IDP   IDR  Rcll  Prcn GT MT PT ML  FP  FN IDs   FM  MOTA  MOTP IDt IDa IDm
3p_da_pm_pp.msv 98.1% 97.5% 98.7% 98.7% 97.5%  3  3  0  0  41  20   0    3 96.2% 0.025   0   0   0
3p_da_pm_up.msv 92.8% 92.5% 93.2% 98.9% 98.2%  3  3  0  0  30  18   1    4 97.0% 0.026   0   1   0
3p_da_um_pp.msv 98.8% 98.8% 98.9% 98.9% 98.9%  3  3  0  0  19  18   2    6 97.7% 0.027   1   0   0
3p_da_um_up.msv 98.5% 98.4% 98.6% 98.6% 98.4%  3  3  0  0  22  19   0    5 97.0% 0.029   0   0   0
3p_sa_pm_pp.msv 89.3% 89.6% 89.0% 98.2% 98.8%  3  3  0  0  19  29   1    7 96.9% 0.023   0   1   0
3p_sa_pm_up.msv 84.3% 84.4% 84.2% 97.8% 98.0%  3  3  0  0  31  35   2    8 95.7% 0.033   1   1   0
3p_sa_um_pp.msv 82.4% 82.5% 82.3% 98.4% 98.5%  3  3  0  0  23  25   3    4 96.7% 0.022   3   0   0
3p_sa_um_up.msv 84.7% 84.5% 85.0% 99.3% 98.7%  3  3  0  0  19  11   2    2 97.9% 0.021   2   0   0
5p_da_pm_pp.msv 89.0% 88.8% 89.2% 97.3% 96.8%  5  5  0  0  69  58   1    5 94.0% 0.036   0   1   0
5p_da_pm_up.msv 97.2% 96.6% 97.8% 97.8% 96.6%  5  5  0  0  71  46   0   14 94.4% 0.042   0   0   0
5p_da_um_pp.msv 68.4% 68.9% 67.8% 96.9% 98.5%  5  5  0  0  32  66   7    9 95.1% 0.030   3   4   0
5p_da_um_up.msv 74.6% 74.0% 75.3% 97.5% 95.8%  5  5  0  0  89  52   7   13 92.9% 0.033   2   4   0
5p_sa_pm_pp.msv 92.5% 91.8% 92.2% 95.9% 96.4%  5  5  0  0  79  90   2   32 92.2% 0.037   0   2   0
5p_sa_pm_up.msv 69.6% 69.0% 68.8% 93.3% 95.7%  5  5  0  0  90 143   9   63 88.7% 0.048   4   4   0
5p_sa_um_pp.msv 58.1% 58.4% 57.9% 96.0% 96.8%  5  5  0  0  70  87  11   16 92.3% 0.040   8   2   0
5p_sa_um_up.msv 51.7% 51.7% 51.8% 95.8% 95.8%  5  5  0  0  87  86  12   22 91.0% 0.047   7   4   0
OVERALL         82.0% 81.8% 82.0% 97.3% 97.3% 64 64  0  0 791 803  60  213 94.4% 0.033  31  24   0
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
