# Rotation-Equivariant Self-Supervised Method in Image Denoising（CVPR 2025）
Official implementation.

> **Abstract**: Self-supervised image denoising methods have garnered significant research attention in recent years, for this kind of method reduces the requirement of large training datasets.
> Compared to supervised methods, self-supervised methods rely more on the prior embedded in deep networks themselves. As a result, most of the self-supervised methods are designed with Convolution Neural Networks (CNNs) architectures, which well capture one of the most important image prior, translation equivariant prior. Inspired by the great success achieved by the introduction of translational equivariance, in this paper, we explore the way to further incorporate another important image prior. 
> Specifically, we first apply high-accuracy rotation equivariant convolution to self-supervised image denoising. Through rigorous theoretical analysis, we have proved that simply replacing all the convolution layers with rotation equivariant convolution layers would modify the network into its rotation equivariant version.
> To the best of our knowledge, this is the first time that rotation equivariant image prior is introduced to self-supervised image denoising at the network architecture level with a comprehensive theoretical analysis of equivariance errors, which
> offers a new perspective to the field of self-supervised image denoising.
> Moreover, to further improve the performance, we design a new mask mechanism to fusion the output of rotation equivariant network and vanilla CNN-based network, and construct an adaptive rotation equivariant framework. 
> Through extensive experiments on three typical methods, we have demonstrated the effectiveness of the proposed method.


## Introduction

<div align="center">
  <img src="/image/equivariance.png" alt="Equivariance comparison">
  <p><em>Illustration of the output feature map of a typical image obtained by standard CNN and our used rotation equivariant convolution neural network. Both networks are initialized randomly.</em></p>
</div>

## Network Architecture

<div align="center">
  <img src="/image/adarenet.png" alt="AdaReNet architecture">
</div>


## Experiments
### Installation
```bash
conda env create -f environment.yml
conda activate n2n
```

### Dataset
Same setup as previous articles, we train the network using 256×256-pixel crops drawn from the 50k images in the IMAGENET validation set.


### Training
```bash
CUDA_VISIBLE_DEVICES=0 python src/train.py --train-dir [training data] --train-size 50000 --valid-dir [valid data] --valid-size 24 --ckpt-save-path [save path] --nb-epochs 40 --batch-size 4 --loss l2 --noise-type [gaussian] --noise-param [50] --seed [seed] --crop-size 256 --plot-stats --cuda  --report-interval [1250]
```
