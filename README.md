# Cross_Modal_GCN
Semi-supervised Graph Convolutional Hashing Network For Large-Scale Cross-Modal Hashing (Shen, Zhai et al., *ICIP 2020*)

### Overview
Here we provide the implementation of our model in TensorFlow, along with all experimental datasets. The repository is organised as follows:
- `data/` contains the necessary dataset files. Here, we have three datasets including *MIRFLICKR-25K*, *NUS-WIDE-10K* and *Wiki*, which can be downloaded from pan.baidu.com: link: https://pan.baidu.com/s/1DlIxCvT_3vRKphMydnljVQ code: b6be.
- `train_flickr.py` execute a full training run on MIRFLICKR-25K dataset.
- `train_nuswide.py` execute a full training run on NUS-WIDE-10K dataset.
- `train_wiki.py` execute a full training run on Wiki dataset.

### Dependencies
The script has been tested running under Python 2.7, with the following packages installed (along with their dependencies):
- `numpy==1.15.4`
- `tensorflow-gpu==1.12.0`

In addition, CUDA 9.0 and cuDNN 7 have been used.
