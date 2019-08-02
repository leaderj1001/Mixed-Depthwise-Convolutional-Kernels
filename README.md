# Implementing Mixed-Depthwise-Convolutional-Kernels using Pytorch (22 Jul 2019)
- Author:
  - Mingxing Tan (Google Brain)
  - Quoc V. Le (Google Brain)
- [Paper Link](https://arxiv.org/abs/1907.09595?context=cs.LG)

# Method
![캡처](https://user-images.githubusercontent.com/22078438/62100515-fbdabe00-b2cc-11e9-950d-e02da609f60b.PNG)
- By using a multi scale kernel size, performance improvements and efficiency were obtained.
- Each kernel size has a different receptive field, so we can get different feature maps for each kernel size.

# Experiment

| Datasets | Model | Acc1 | Acc5 | Parameters (My Model, Paper Model)
| :---: | :---: | :---: | :---: | :---: |
CIFAR-10 | MixNet-s (WORK IN PROCESS) | 92.82% | 99.79% | 2.6M, -
CIFAR-10 | MixNet-m (WORK IN PROCESS) | 92.52% | 99.78% | 3.5M, -
CIFAR-10 | MixNet-l (WORK IN PROCESS) | 92.72% | 99.79% | 5.8M, -
IMAGENET | MixNet-s (WORK IN PROCESS) | | | 4.1M, 4.1M
IMAGENET | MixNet-m (WORK IN PROCESS) | | | 5.0M, 5.0M
IMAGENET | MixNet-l (WORK IN PROCESS) | | | 7.3M, 7.3M

## Usage
```python
python main.py
```
- `--data` (str): the ImageNet dataset path
- `--dataset` (str): dataset name, (example: CIFAR10, CIFAR100, MNIST, IMAGENET)
- `--batch-size` (int)
- `--num-workers` (int)
- `--epochs` (int)

- `--lr` (float): learning rate
- `--momentum` (float): momentum
- `--weight-decay` (float): weight dacay
- `--print-interval` (int): training log print cycle
- `--cuda` (bool)
- `--pretrained-model` (bool): hether to use the pretrained model

## Todo
- Distributed SGD
- ImageNet experiment

# Reference
- [MixNet Official Github](https://github.com/tensorflow/tpu/blob/master/models/official/mnasnet/mixnet/README.md)
