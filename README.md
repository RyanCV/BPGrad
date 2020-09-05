# BPGrad: Towards Global Optimality in Deep Learning via Branch and Pruning
This repository contains the code for our CVPR'18 paper [BPGrad: Towards Global Optimality in Deep Learning via Branch and Pruning](https://openaccess.thecvf.com/content_cvpr_2018/papers/Zhang_BPGrad_Towards_Global_CVPR_2018_paper.pdf), by Ziming Zhang, Yuanwei Wu and Guanghui Wang, Mitsubishi Electric Research Laboratories and The University of Kansas.

## Introduction
Understanding the global optimality in deep learning (DL) has been attracting more and more attention recently. Conventional DL solvers, however, have not been developed intentionally to seek for such global optimality. In this pa- per we propose a novel approximation algorithm, BPGrad, towards optimizing deep models globally via branch and pruning. Our BPGrad algorithm is based on the assumption of Lipschitz continuity in DL, and as a result it can adap- tively determine the step size for current gradient given the history of previous updates, wherein theoretically no smaller steps can achieve the global optimality. We prove that, by re- peating such branch-and-pruning procedure, we can locate the global optimality within finite iterations. Empirically an efficient solver based on BPGrad for DL is proposed as well, and it outperforms conventional DL solvers such as Ada- grad, Adadelta, RMSProp, and Adam in the tasks of object recognition, detection, and segmentation.

## Requirements
- Matlab 2017
- [MatConvNet](https://github.com/vlfeat/matconvnet)

## Training and Test
- To install MatConvNet following the instruction
- Run training and test for cifar10, different solvers could be selected. The default solver is BPGrad.
```
>> cnn_train_v2_cifar_BPGrad.m
```


## Citation
If you find BPGrad helps your research, please cite our paper:
```
@InProceedings{Zhang_2018_CVPR,
author = {Zhang, Ziming and Wu, Yuanwei and Wang, Guanghui},
title = {BPGrad: Towards Global Optimality in Deep Learning via Branch and Pruning},
booktitle = {Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
month = {June},
year = {2018}
}
```
