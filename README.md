# DFRF-jittor
A jittor implementation for ECCV2022 paper [DFRF](https://arxiv.org/abs/2207.11770) "Learning Dynamic Facial Radiance Fields for Few-Shot Talking Head Synthesis". The code is based on authors' Pytorch implementation [here](https://github.com/sstzal/DFRF).

## Requirements

First, install the requirements following `install_script`.

We conduct the experiments with a 24G RTX3090.

- Download 3DMM model from [Basel Face Model 2009](https://faces.dmi.unibas.ch/bfm/main.php?nav=1-1-0&id=details):

  ```
  cp 01_MorphableModel.mat data_util/face_tracking/3DMM/
  cd data_util/face_tracking
  python convert_BFM.py
  ```

## Dataset

Put the video `${id}.mp4` to `dataset/vids/`, then run the following command for data preprocess.

```
sh process_data.sh ${id}
```

Specially, to process data for training base model:

- Download the videos for training base model provided by the author [here](https://github.com/sstzal/DFRF/releases/tag/Base_Videos).
- Put the videos to `dataset/vids/`.
- Run the following command for data preprocess.

```
sh process_base_train_data.sh 0
sh process_base_train_data.sh 1
sh process_base_train_data.sh 2
```

## Training

```
sh run.sh ${id}
```

Some pre-trained models are [here](https://github.com/sstzal/DFRF/releases/tag/Pretrained_Models).

Specially, to train the base model:

```
sh train_base.sh
```

## Test

Change the configurations in the `rendering.sh`, including the `iters, names, datasets, near and far`.

```
sh rendering.sh
```

## Acknowledgement

This code is built upon the publicly available code [DFRF](https://arxiv.org/abs/2207.11770). Thanks the authors of DFRF for making their excellent work and codes publicly available.