import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import functools
from collections import OrderedDict
import models.modules.module_util as mutil
from models.modules.my_module import kernel_extra, kernel_extra_mean_var
from models.modules.DLSM import DLSM


class BlindNet(nn.Module):
    def __init__(self, scale=4, kernel_size=21):
        super(BlindNet, self).__init__()

        self.kernel_extra = kernel_extra_mean_var(only_mean=False)
        self.SR_net = DLSM(scale)
        self.scale = scale

    def forward(self, x, gt_K):
        # kernel estimation
        kernel_fea, kernel, mean, log_var = self.kernel_extra(x)

        # nonblind sr
        out = self.SR_net(x, kernel)

        # expand kernel to cal loss
        kernel_ex = kernel.unsqueeze(1)
        kernel_ex = kernel_ex.expand(kernel_ex.size(0), kernel_fea.size(2) * kernel_fea.size(3) * self.scale * self.scale, kernel_ex.size(2),kernel_ex.size(3))
        return out, kernel_ex, mean, log_var



class BlindNet_nouncer(nn.Module):
    def __init__(self, scale=4, kernel_size=21):
        super(BlindNet_nouncer, self).__init__()

        self.kernel_extra = kernel_extra()
        self.SR_net = DLSM(scale)
        self.scale = scale

    def forward(self, x, gt_K):
        # kernel estimation
        kernel_fea, kernel = self.kernel_extra(x)

        # nonblind sr
        out = self.SR_net(x, kernel)

        # expand kernel to cal loss
        kernel_ex = kernel.unsqueeze(1)
        kernel_ex = kernel_ex.expand(kernel_ex.size(0),
                                     kernel_fea.size(2) * kernel_fea.size(3) * self.scale * self.scale,
                                     kernel_ex.size(2), kernel_ex.size(3))

        return out, kernel_ex, None, None


if __name__ == '__main__':
    model = BlindNet().cuda()
    print("Total number of paramerters in networks is {}  ".format(sum(x.numel() for x in model.parameters())))


    # x = torch.randn((2, 3, 123, 115)).cuda()
    # kernel = torch.randn((2, 21, 21)).cuda()
    # with torch.no_grad():
    #     out, kernel = model(x, kernel)
    # print(out.shape)
