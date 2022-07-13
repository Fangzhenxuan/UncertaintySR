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
We formulate the blind SR problem as a joint maximum a posteriori probability (MAP) problem for estimating the unknown kernel and highresolution image simultaneously. To improve the robustness of the kernel estimation network, we introduce uncertainty learning in the latent space instead
of using deterministic feature maps. Then we propose a novel multi-stage SR
network by unfolding the MAP estimator with the learned LSM prior and the
estimated kernel. Both the scale prior coefficient and the local means of the LSM
model are estimated through deep convolutional neural networks. All parameters of the MAP estimation algorithm and the DCNN parameters are jointly
optimized through end-to-end training. Extensive experimental results on both
synthetic and real datasets demonstrate that the proposed method outperforms
existing state-of-the-art methods. Future research directions include the extension of this work to spatially varying blur kernels and the generalization study
to more real-world test images.

<p align="center">
<img src="/illustrations/visual.png" width="1200">
</p>
Fig. 1 A single shot measurement captured by [1] and 28 reconstructed spectral channels using our proposed method.

## Architecture
<p align="center">
<img src="/Images/Fig2.png" width="1200">
</p>
Fig. 2 Architecture of the proposed network for hyperspectral image reconstruction. The architectures of (a) the overall network, (b)
the measurement matrix, (c) the transposed version of the measurement matrix, (d) the weight generator, and (e) the filter generator.

## Usage
### Download the DGSMP repository
0. Requirements are Python 3 and PyTorch 1.2.0.
1. Download this repository via git
```
git clone https://github.com/TaoHuang95/DGSMP
```
or download the [zip file](https://github.com/TaoHuang95/DGSMP/archive/main.zip) manually.

### Download the training data
1. CAVE:28 channels (https://pan.baidu.com/s/1ue26weBAbn61a7hyT9CDkg) PW:ixoe
2. KAIST:28 channels (https://pan.baidu.com/s/1LfPqGe0R_tuQjCXC_fALZA) PW:5mmn


### Testing 
1. Testing on simulation data   
Run **Simulation/Test.py** to reconstruct 10 synthetic datasets ([Ziyi Meng](https://github.com/mengziyi64/TSA-Net)). The results will be saved in 'Simulation/Results/' in the MatFile format.  
2. Testing on real data   
Run **Real/Test.py** to reconstruct 5 real datasets ([Ziyi Meng](https://github.com/mengziyi64/TSA-Net)). The results will be saved in 'Real/Results/' in the MatFile format.  

### Training 
1. Training simulation model
    1) Put hyperspectral datasets (Ground truth) into corrsponding path, i.e., 'Simulation/Data/Training_data/'.
    2) Run **Simulation/Train.py**.
2. Training real data model  
    1) Put hyperspectral datasets (Ground truth) into corrsponding path, i.e., 'Real/Data/Training_data/'.  
    2) Run **Real/Train.py**.

## Acknowledgements
We thank the author of TSA-Net[1] ([Ziyi Meng](https://github.com/mengziyi64/TSA-Net)) for providing simulation data and real data.

## References
[1] Ziyi Meng, Jiawei Ma, and Xin Yuan. End-to-end low cost compressive spectral imaging with spatial-spectral self-attention. In Proceedings of the European Conference on
Computer Vision (ECCV), August 2020.

## Citation

If you find our work useful for your research, please consider citing the following papers :)

```
@InProceedings{Huang_2021_CVPR,
    author    = {Huang, Tao and Dong, Weisheng and Yuan, Xin and Wu, Jinjian and Shi, Guangming},
    title     = {Deep Gaussian Scale Mixture Prior for Spectral Compressive Imaging},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2021},
    pages     = {16216-16225}
}
```

## Contact
Tao Huang, Xidian University, Email: thuang_666@stu.xidian.edu.cn, thuang951223@163.com  
Weisheng Dong, Xidian University, Email: wsdong@mail.xidian.edu.cn  
Xin Yuan, Bell Labs, Email: xin_x.yuan@nokia-bell-labs.com