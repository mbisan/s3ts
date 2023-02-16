# Self-Supervised Streaming Time Series Classification

Implementation of self-supervised learning to the **streaming time series (STS) classification** using an **image-based feature representation**. 

The images, known as **dissimilarity frames** (DFs), are constructed using the **dynamic time Warping** (DTW) similarity measure between the STS and several of pattern time series.

The **pretext tasks** for the self-supervised learning (unsupervised pretraining) of the 2D convolutional decoder, **quantile identification** and **frame ordering**, are based on the image-time series duality of the approach.

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