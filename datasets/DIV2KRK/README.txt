###################
# DIV2KRK Dataset #
###################

For detailed discription of the data-creation proccess see paper: 
"KernelGAN" Blind Super-Resolution Kernel Estimation using an Internal-GAN

## Zip file contains:
    gt: 100 ground truth validation images from DIV2K  
    gt_k_x2: 100 11x11 random kernels
    lr_x2: gt images downscaled x2 using gt_k_x2 kernels
    gt_k_x4: 100 31x31 random kernels for scale factor 4
    lr_x2: gt images downscaled x4 using gt_k_x4 kernels