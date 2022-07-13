# Uncertainty Learning in Kernel Estimation for Multi-Stage Blind Image Super-Resolution
This repository contains the Pytorch codes for paper **Uncertainty Learning in Kernel Estimation for
Multi-Stage Blind Image Super-Resolution** (***ECCV (2022)***) by [Zhenxuan Fang](https://github.com/Fangzhenxuan), [Weisheng Dong](https://see.xidian.edu.cn/faculty/wsdong/), et al.  



## Contents
1. [Overview](#Overview)
2. [Architecture](#Architecture)
3. [Usage](#Usage)
4. [Acknowledgements](#Acknowledgements)
5. [References](#References)
6. [Citation](#Citation)
7. [Contact](#Contact)

## Overview
We formulate the blind SR problem as a joint maximum a posteriori probability (MAP) problem for estimating the unknown kernel and high-resolution image simultaneously. To improve the robustness of the kernel estimation network, we introduce uncertainty learning in the latent space instead of using deterministic feature maps. Then we propose a novel multi-stage SR network by unfolding the MAP estimator with the learned LSM prior and the estimated kernel. Both the scale prior coefficient and the local means of the LSM model are estimated through deep convolutional neural networks. All parameters of the MAP estimation algorithm and the DCNN parameters are jointly
optimized through end-to-end training. Extensive experimental results on both synthetic and real datasets demonstrate that the proposed method outperforms existing state-of-the-art methods. Future research directions include the extension of this work to spatially varying blur kernels and the generalization study to more real-world test images.

<p align="center">
<img src="/illustrations/visual.png" width="1200">
</p>
Fig. 1 Visual comparison to other methods.

## Architecture
<p align="center">
<img src="/illustrations/network.png" width="1200">
</p>
Fig. 2 The overall framework of the proposed KULNet for blind SR.

## Usage
### Download the DGSMP repository
1. Requirements are Python 3 and PyTorch 1.8.0.
2. Download this repository via git
```
git clone https://github.com/Fangzhenxuan/UncertaintySR
```
or download the [zip file](https://github.com/Fangzhenxuan/UncertaintySR/archive/main.zip) manually.


### Testing
1. Testing on synthetic testsets   
Run **codes/test.py**. For different settings (noise level, testsets), Modify the corresponding parameters in **codes/options/test.yml** 
2. Testing on real data   
Run **codes/test_real.py**. 

### Training 
Run **codes/train.py**.


## Acknowledgements
The codes are based on [MANet](https://github.com/JingyunLiang/MANet) [1]. Please also follow their licenses. Thanks for their great works.

## References
[1] Liang, Jingyun, et al. "Mutual affine network for spatially variant kernel estimation in blind image super-resolution." Proceedings of the IEEE/CVF International Conference on Computer Vision. 2021.



