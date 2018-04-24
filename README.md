<p align="center"><img width="40%" src="logo/pytorch_logo.png" /></p>

--------------------------------------------------------------------------------

This repository implements popular CNN architectures on CIFAR 10 dataset using pytorch.

## Dependency

[Pytorch (0.3.1)](http://pytorch.org/)
[Python 2.7.12](https://www.python.org/)

## Results

CNN Architecture | Accuracy
------------ | -------------
VGG16 | 93.15%
VGG19 | 93.10%
ResNet18 | 93.33%
ResNet34 | 93.49%
ResNet101 | 95.11%
ResNet152 | 95.50%
DenseNet121 | 95.37%
DenseNet161 | 95.49%
DenseNet201 | 95.50%

## Getting started 

**need GPUS**


```bash
python main.py --help
python main.py --models=densenet121 --gpu=0 --visible=0,1,2
```


## Author
Haibin Yu [@HeroKillerEver](https://github.com/HeroKillerEver)