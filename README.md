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
- **SORT Tracker**
```
                 IDF1   IDP   IDR  Rcll  Prcn GT MT PT ML  FP  FN IDs  FM  MOTA  MOTP IDt IDa IDm
3p_da_pm_pp.msv 37.4% 37.1% 37.6% 99.7% 98.3%  3  3  0  0  28   5  17   0 96.9% 0.001   0  17   0
3p_da_pm_up.msv 30.4% 30.3% 30.6% 99.8% 98.8%  3  3  0  0  20   4  19   1 97.4% 0.002   0  19   0
3p_da_um_pp.msv 46.4% 46.2% 46.7% 99.6% 98.5%  3  3  0  0  25   7  11   3 97.5% 0.005   0  11   0
3p_da_um_up.msv 50.1% 49.7% 50.6% 99.9% 98.1%  3  3  0  0  27   2  14   1 96.9% 0.005   0  14   0
3p_sa_pm_pp.msv 37.7% 37.6% 37.7% 99.1% 98.8%  3  3  0  0  20  15  19   4 96.6% 0.007   0  19   0
3p_sa_pm_up.msv 26.5% 26.4% 26.5% 99.6% 99.4%  3  3  0  0  10   7  19   1 97.7% 0.006   0  19   0
3p_sa_um_pp.msv 31.4% 31.2% 31.5% 99.6% 98.6%  3  3  0  0  22   7  18   0 97.0% 0.004   0  18   0
3p_sa_um_up.msv 38.1% 37.8% 38.4% 99.9% 98.4%  3  3  0  0  25   2  12   2 97.4% 0.003   1  11   0
5p_da_pm_pp.msv 36.0% 36.1% 36.0% 98.1% 98.4%  5  5  0  0  35  41  37   6 94.7% 0.006   0  37   0
5p_da_pm_up.msv 27.9% 27.8% 28.0% 98.6% 97.7%  5  5  0  0  49  29  43   9 94.2% 0.011   1  42   0
5p_da_um_pp.msv 32.3% 32.1% 32.5% 99.0% 98.0%  5  5  0  0  43  21  33   3 95.4% 0.004   3  30   0
5p_da_um_up.msv 38.5% 37.9% 39.2% 99.1% 96.0%  5  5  0  0  85  18  31   5 93.5% 0.004   0  31   0
5p_sa_pm_pp.msv 33.0% 32.7% 33.0% 98.3% 98.5%  5  5  0  0  32  37  53   7 94.4% 0.004   0  41   0
5p_sa_pm_up.msv 28.5% 28.0% 28.4% 97.5% 98.4%  5  5  0  0  33  53  85  19 92.0% 0.013   0  57   0
5p_sa_um_pp.msv 32.6% 32.5% 32.6% 98.0% 97.9%  5  5  0  0  45  43  40   4 94.1% 0.008   2  36   0
5p_sa_um_up.msv 33.2% 32.6% 33.8% 97.8% 94.4%  5  5  0  0 119  45  45   6 89.8% 0.011   4  38   0
OVERALL         34.6% 34.4% 34.8% 98.9% 97.9% 64 64  0  0 618 336 496  71 95.1% 0.006  11 440   0
```
- **DeepSORT Tracker**
```
                 IDF1   IDP   IDR  Rcll  Prcn GT MT PT ML  FP  FN IDs   FM  MOTA  MOTP IDt IDa IDm
3p_da_pm_pp.msv 93.9% 93.0% 94.8% 99.5% 97.6%  3  3  0  0  39   8   1    3 97.0% 0.014   0   1   0
3p_da_pm_up.msv 85.0% 84.2% 85.7% 99.5% 97.8%  3  3  0  0  37   8   2    4 97.1% 0.014   0   2   0
3p_da_um_pp.msv 86.6% 86.3% 86.9% 99.4% 98.8%  3  3  0  0  21  10   3    6 98.0% 0.016   1   1   0
3p_da_um_up.msv 94.7% 94.4% 95.1% 98.9% 98.2%  3  3  0  0  25  15   3    6 96.9% 0.020   2   1   0
3p_sa_pm_pp.msv 58.4% 58.1% 58.6% 98.7% 98.0%  3  3  0  0  32  20   6    6 96.4% 0.016   1   6   1
3p_sa_pm_up.msv 58.6% 58.2% 58.9% 99.1% 97.9%  3  3  0  0  34  14   5    5 96.6% 0.023   2   3   0
3p_sa_um_pp.msv 65.4% 65.5% 65.4% 98.5% 98.6%  3  3  0  0  22  24   8    8 96.6% 0.014   4   3   0
3p_sa_um_up.msv 78.9% 78.5% 79.2% 99.4% 98.5%  3  3  0  0  23   9   5    4 97.5% 0.010   3   1   0
5p_da_pm_pp.msv 97.4% 97.0% 97.8% 97.8% 97.0%  5  5  0  0  65  48   0    9 94.7% 0.025   0   0   0
5p_da_pm_up.msv 78.5% 77.8% 79.2% 98.2% 96.4%  5  5  0  0  77  38   4   15 94.3% 0.028   0   4   0
5p_da_um_pp.msv 72.7% 72.5% 72.9% 98.3% 97.7%  5  5  0  0  49  37   9   11 95.5% 0.022   1   6   0
5p_da_um_up.msv 59.3% 58.2% 60.3% 98.6% 95.2%  5  5  0  0 104  30  17   14 92.7% 0.024  10   6   0
5p_sa_pm_pp.msv 53.1% 52.3% 53.3% 96.3% 95.5%  5  5  0  0 100  82  18   38 90.9% 0.033  11   6   0
5p_sa_pm_up.msv 60.8% 60.0% 60.2% 93.2% 95.0%  5  5  0  0 105 145  26   72 87.2% 0.045  13   9   0
5p_sa_um_pp.msv 51.9% 51.9% 51.8% 96.6% 96.7%  5  5  0  0  72  75  17   17 92.5% 0.031  12   5   1
5p_sa_um_up.msv 56.6% 56.2% 57.0% 96.1% 94.8%  5  5  0  0 108  79  19   22 90.0% 0.038   8   7   0
OVERALL         71.0% 70.5% 71.4% 97.8% 96.9% 64 64  0  0 913 642 143  240 94.3% 0.024  68  61   2
```
- **CAT Tracker**
```
                 IDF1   IDP   IDR  Rcll  Prcn GT MT PT ML  FP  FN IDs   FM  MOTA  MOTP IDt IDa IDm
3p_da_pm_pp.msv 98.3% 97.4% 99.2% 99.2% 97.4%  3  3  0  0  43  12   0    4 96.5% 0.025   0   0   0
3p_da_pm_up.msv 93.0% 92.3% 93.7% 99.6% 98.1%  3  3  0  0  31   6   1    3 97.7% 0.027   0   1   0
3p_da_um_pp.msv 86.0% 85.8% 86.2% 98.9% 98.5%  3  3  0  0  26  18   4    6 97.2% 0.025   2   0   0
3p_da_um_up.msv 95.9% 95.4% 96.3% 99.2% 98.3%  3  3  0  0  24  11   1    5 97.4% 0.029   0   1   0
3p_sa_pm_pp.msv 84.7% 84.7% 84.7% 98.9% 98.8%  3  3  0  0  19  18   2    4 97.6% 0.023   0   2   0
3p_sa_pm_up.msv 87.1% 86.7% 87.5% 99.0% 98.0%  3  3  0  0  31  15   1    5 97.0% 0.033   0   1   0
3p_sa_um_pp.msv 77.8% 77.8% 77.8% 98.7% 98.7%  3  3  0  0  20  20   5    5 97.1% 0.022   2   2   0
3p_sa_um_up.msv 88.4% 88.0% 88.9% 99.7% 98.7%  3  3  0  0  19   4   1    2 98.4% 0.020   0   1   0
5p_da_pm_pp.msv 78.4% 77.8% 79.0% 98.0% 96.5%  5  5  0  0  77  43   3    6 94.3% 0.035   1   2   0
5p_da_pm_up.msv 63.8% 63.3% 64.3% 97.9% 96.4%  5  5  0  0  77  44   9   16 93.7% 0.043   2   7   0
5p_da_um_pp.msv 71.0% 70.6% 71.5% 98.0% 96.8%  5  5  0  0  70  43  11   13 94.2% 0.033   4   7   0
5p_da_um_up.msv 62.1% 61.1% 63.2% 98.3% 95.1%  5  5  0  0 106  35  10   14 92.7% 0.036   5   5   0
5p_sa_pm_pp.msv 92.7% 91.6% 93.0% 96.9% 96.4%  5  5  0  0  79  68   2   32 93.2% 0.037   0   2   0
5p_sa_pm_up.msv 55.8% 54.8% 55.6% 94.9% 95.5%  5  5  0  0  97 110  19   66 89.5% 0.053  10   8   0
5p_sa_um_pp.msv 56.2% 56.1% 56.4% 96.4% 95.9%  5  5  0  0  91  79  23   19 91.2% 0.047  16   4   0
5p_sa_um_up.msv 58.5% 57.8% 59.2% 96.8% 94.6%  5  5  0  0 114  66  17   20 90.4% 0.047   5   9   0
OVERALL         76.5% 75.9% 76.9% 98.0% 96.9% 64 64  0  0 924 592 109  220 94.5% 0.035  47  52   0
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
