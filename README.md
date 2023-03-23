# S3TS: Self-Supervised Streaming Time Series Classification

This repository contains the code for the paper **S3TS: Self-supervised Streaming Time Series Classification** by **Raúl Coterillo** and **Aritz Perez**, where we propose a novel approach for the classification of streaming time series (STS) using self-supervised learning and an image-based feature representation. The images, known as dissimilarity frames (DFs), are constructed using the dynamic time Warping (DTW) similarity measure between the STS and several of pattern time series. We use time series regression as a pretext task for the self-supervised learning (unsupervised pretraining) of the 2D convolutional decoder, based on the image-time series duality of our model.
We used UCR Time Series Classification Archive (UCR/UEA) datasets to construct the STSs to pretrain, finetune and test our model.

**DISCLAIMER:** The code in this repository is intended for testing and development of this methodology for later publication, and under no circumstance should it be used in a production setting.

## Environment / Setup

```bash
git clone https://github.com/rcote98/s3ts.git   # clone the repo
cd s3ts                                         # move in the folder
python3 -m venv s3ts_env                        # create virtualenv
source s3ts_env/bin/activate                    # activate it
pip install -r requirements.txt                 # install dependencies
python -m pip install -e .                      # install dev package
```

## Visualize Training Progress

```bash
tensorboard --logdir=[whatever-dir-name]/logs
```
