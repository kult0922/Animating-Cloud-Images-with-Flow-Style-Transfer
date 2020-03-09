# Animating Cloud Images with Flow Style Transfer
<table border="0">
<tr>
<td><img src="https://github.com/Kult0922/Animating-Arbirary-SkyScene/blob/master/figs/driving_video.gif"></td>
<td><img src="https://github.com/Kult0922/Animating-Arbirary-SkyScene/blob/master/figs/generate_videos.gif"></td>
</tr>
</table>

## Installation

We support ```Python3```. To install the dependenceis run:

```
pip install -r requirements.txt
```
## train
```
python3 train.py
```

## Demo
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
