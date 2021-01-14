# Animating Cloud Images with Flow Style Transfer
[Project Page](https://kult0922.github.io/Animating-Cloud-Images-with-Flow-Style-Transfer "Project Page")
<table border="0">
<tr>
<td><img src="https://github.com/Kult0922/Animating-Arbirary-SkyScene/blob/master/figs/driving_video.gif"></td>
<td><img src="https://github.com/Kult0922/Animating-Arbirary-SkyScene/blob/master/figs/generate_videos.gif"></td>
</tr>
</table>

## Installation
We support ```Python3``` enviroment. To install the dependencies run:

```
pip install -r requirements.txt
```

## training

```
python3 train.py
```

## Motion Transfer Demo

```
python3 demo.py --checkpoint path/to/checkpoint --startEpoch 20 --sourceImage path/to/source_image --drivingVideo path/to/driving_video_dir
```

## Motion Transfer

```
python3 test.py --mode transfer --batchSize 8 --checkpoint path/to/checkpoint --startEpoch 20 --sourceImage path/to/source_image --drivingVideo path/to/driving_video_dir
```

## Video Reconstruction

```
python3 test.py --mode reconstruction --batchSize 8 --checkpoint path/to/checkpoint --startEpoch 20 --sourceImage path/to/source_image --drivingVideo path/to/driving_video_dir
```

## Publication
Kazuma Kurisaki and Kazuhiko Kawamoto, Animating Cloud Images With Flow Style Transfer, IEEE Access, Vol.9, pp.3269-3277, 2021.

## Citation
```
@ARTICLE{9311211,
author={K. {Kurisaki} and K. {Kawamoto}},
journal={IEEE Access},
title={Animating Cloud Images With Flow Style Transfer},
year={2021},  volume={9},  number={},  pages={3269-3277},
doi={10.1109/ACCESS.2020.3048160}}
```
