import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import functools
from collections import OrderedDict
import models.modules.module_util as mutil
from models.modules.my_module import kernel_extra_mean_var
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



