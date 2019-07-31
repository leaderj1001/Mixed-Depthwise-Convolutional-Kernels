# Implementing Mixed-Depthwise-Convolutional-Kernels using Pytorch (22 Jul 2019)
- Author:
  - Mingxing Tan (Google Brain)
  - Quoc V. Le (Google Brain)
- [Paper Link](https://arxiv.org/abs/1907.09595?context=cs.LG)

# Method
![캡처](https://user-images.githubusercontent.com/22078438/62100515-fbdabe00-b2cc-11e9-950d-e02da609f60b.PNG)

# Experiment

| Datasets | Model | Acc1 | Acc5 | Parameters (My Model, Paper Model)
| :---: | :---: | :---: | :---: | :---: |
CIFAR-10 | MixNet-s (WORK IN PROCESS) | 90.59% | 99.72% | 2.6M, -
CIFAR-10 | MixNet-m (WORK IN PROCESS) | 90.48% | 99.63% | 3.5M, -
CIFAR-10 | MixNet-l (WORK IN PROCESS) | 91.39% | 99.81% | 5.8M, -
IMAGENET | MixNet-s (WORK IN PROCESS) | | | 4.1M, 4.1M
IMAGENET | MixNet-m (WORK IN PROCESS) | | | 5.0M, 5.0M
IMAGENET | MixNet-l (WORK IN PROCESS) | | | 7.3M, 7.3M

# Reference
- [MixNet Official Github](https://github.com/tensorflow/tpu/blob/master/models/official/mnasnet/mixnet/README.md)
