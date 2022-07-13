import torch
import logging
import models.modules.BlindNet_arch as BlindNet_arch

logger = logging.getLogger('base')


####################
# define network
####################
#### Generator
def define_G(opt):
    opt_net = opt['network_G']
    which_model = opt_net['which_model_G']

    if which_model == 'BlindNet':
        netG = BlindNet_arch.BlindNet(scale=opt['scale'], kernel_size=opt['kernel_size'])
    else:
        raise NotImplementedError('Generator model [{:s}] not recognized'.format(which_model))
    return netG






