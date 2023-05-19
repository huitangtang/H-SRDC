# H-SRDC
Code release for `Towards Uncovering the Intrinsic Data Structures for Unsupervised Domain Adaptation using Structurally Regularized Deep Clustering`, which is published in IEEE TRANSACTIONS ON PATTERN ANALYSIS AND MACHINE INTELLIGENCE in 2022. 

[Project Page](https://huitangtang.github.io/H-SRDC/) $\cdot$ [PDF Download](https://arxiv.org/pdf/2012.04280)

The paper is available [here](https://ieeexplore.ieee.org/document/9449976) or at the [arXiv archive](https://arxiv.org/abs/2012.04280).

## Requirements
- python 3.6.4
- pytorch 1.4.0
- torchvision 0.5.0

## Data preparation
The structure of the used datasets is shown in the folder `./data/datasets/`. 

For each adaptation task in an inductive setting, we use all the data on the source domain as the training ones, and make a random, half-half splitting of training and test data for samples of each class on the target domain; the data settings are fixed once prepared. 

The lists of image names for the training and test sets of each target domain are provided in corresponding files, e.g., `./data/datasets/Office31/amazon_half/amazon_half.txt`.

The original datasets can be downloaded [here](https://github.com/jindongwang/transferlearning/blob/master/data/dataset.md).

## Model training
1. Replace paths and domains in run.sh with those in one's own system. 
2. Install necessary python packages.
3. Run command `sh run.sh`.

The results are saved in the folder `./checkpoints/`.

## Article citation
```
@article{tang2021towards,
  author={Tang, Hui and Zhu, Xiatian and Chen, Ke and Jia, Kui and Chen, C. L. Philip},
  journal={IEEE Transactions on Pattern Analysis and Machine Intelligence}, 
  title={Towards Uncovering the Intrinsic Data Structures for Unsupervised Domain Adaptation Using Structurally Regularized Deep Clustering}, 
  year={2022},
  volume={44},
  number={10},
  pages={6517-6533},
  doi={10.1109/TPAMI.2021.3087830}
}
```
