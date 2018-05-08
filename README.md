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
model will be saved in checkpoints/

## Pretrained Models

CNN Architecture | Model
------------ | -------------
VGG16 | [vgg16](https://drive.google.com/file/d/1J0Yv-8McvIoQvpSvVi5DoGOJf-s5zljG/view?usp=sharing) 
VGG19 | [vgg19](https://drive.google.com/file/d/1j1-qXsSXykrRpmIGmRt28PHUZH3Tk6py/view?usp=sharing)
ResNet18 | [resnet18](https://drive.google.com/file/d/1fe-6V5ATGEM8LEHvS-_YMmGq31WC_aug/view?usp=sharing)
ResNet34 | [resnet34](https://drive.google.com/file/d/1I2nTjFsExK4c85W_sIzFdR1ZUFniGUXM/view?usp=sharing)
ResNet101 | [resnet101](https://drive.google.com/file/d/1E4oSunFtPhAC8EhnYKSPOswwEftjfj8l/view?usp=sharing)
ResNet152 | [resnet152](https://drive.google.com/file/d/1zETx6Tli60q41geOAtavSjK_kklfYr9Z/view?usp=sharing)
DenseNet121 | [densenet121](https://drive.google.com/file/d/1UkMpfNYNEFLvRIGV_HSFJf7UmfG8a3Mv/view?usp=sharing)
DenseNet161 | [densenet161](https://drive.google.com/file/d/1nZuzzEvEuUrTH4L4rkTY8FmjuGqJsODR/view?usp=sharing)
DenseNet201 | [densenet201](https://drive.google.com/file/d/1X9w8wDdwsRYjV7N_51-ueMGPdk4S0GHc/view?usp=sharing)

## Author
Haibin Yu [@HeroKillerEver](https://github.com/HeroKillerEver)
