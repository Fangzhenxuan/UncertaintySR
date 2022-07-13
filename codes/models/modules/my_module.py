
import numpy as np
import torch.nn as nn
import torch

# input channel: 3


class ResBlock(nn.Module):
    def __init__(self, ch):
        super(ResBlock, self).__init__()
        self.body = nn.Sequential(
                        nn.ReLU(),
                        nn.Conv2d(ch, ch, kernel_size=3, stride=1, padding=1),
                        nn.ReLU(),
                        nn.Conv2d(ch, ch, kernel_size=3, stride=1, padding=1))

    def forward(self, input):
        res = self.body(input)
        output = res + input
        return output


class kernel_extra_Encoding_Block(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(kernel_extra_Encoding_Block, self).__init__()
        self.Conv_head = nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1)
        self.ResBlock1 = ResBlock(out_ch)
        self.ResBlock2 = ResBlock(out_ch)
        self.downsample = nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=2, padding=1)
        self.act = nn.ReLU()

    def forward(self, input):
        output = self.Conv_head(input)
        output = self.ResBlock1(output)
        output = self.ResBlock2(output)
        skip = self.act(output)
        output = self.downsample(skip)

        return output, skip


class kernel_extra_conv_mid(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(kernel_extra_conv_mid, self).__init__()
        self.body = nn.Sequential(
                        nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1),
                        nn.ReLU(),
                        nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1),
                        nn.ReLU(),
                        nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1),
                        nn.ReLU(),
                        nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1),
                        nn.ReLU(),
                        nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1),
                        nn.ReLU()
                        )

    def forward(self, input):
        output = self.body(input)
        return output


class kernel_extra_Decoding_Block(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(kernel_extra_Decoding_Block, self).__init__()
        self.Conv_t = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=3, stride=2, padding=1)
        self.Conv_head = nn.Conv2d(out_ch*2, out_ch, kernel_size=3, stride=1, padding=1)
        self.ResBlock1 = ResBlock(out_ch)
        self.ResBlock2 = ResBlock(out_ch)
        self.act = nn.ReLU()

    def forward(self, input, skip):
        output = self.Conv_t(input, output_size=[skip.shape[0], skip.shape[1], skip.shape[2], skip.shape[3]])
        output = torch.cat([output, skip], dim=1)
        output = self.Conv_head(output)
        output = self.ResBlock1(output)
        output = self.ResBlock2(output)
        output = self.act(output)

        return output





class kernel_extra_conv_tail_mean_var(nn.Module):
    def __init__(self, in_ch, out_ch, only_mean=False):
        super(kernel_extra_conv_tail_mean_var, self).__init__()
        self.mean = nn.Sequential(
                        nn.Conv2d(in_ch, in_ch, kernel_size=3, stride=1, padding=1),
                        nn.ReLU()
                        )
        self.log_var = nn.Sequential(
                        nn.Conv2d(in_ch, in_ch, kernel_size=3, stride=1, padding=1),
                        )
        self.Conv_end = nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1)
        self.kernel_size = 21
        self.only_mean = only_mean

    def forward(self, input):
        mean = self.mean(input)
        log_var = self.log_var(input)

        if self.only_mean:
            output = mean
        else:
            epsilon = torch.randn(log_var.shape[0], log_var.shape[1], log_var.shape[2], log_var.shape[3]).cuda()
            var = torch.exp(log_var)
            output = mean + torch.mul(epsilon, var)
        # mean = self.mean(input)
        # var = self.var(input)
        #
        # if self.only_mean:
        #     output = mean
        # else:
        #     epsilon = torch.randn(var.shape[0], var.shape[1], var.shape[2], var.shape[3]).cuda()
        #     output = mean + torch.mul(epsilon, var)

        output = self.Conv_end(output)
        output = nn.Softmax2d()(output)
        kernel = output.mean(dim=[2, 3], keepdim=True)
        kernel = kernel.view(-1, self.kernel_size, self.kernel_size)
        return output, kernel, mean, log_var



class kernel_extra_mean_var(nn.Module):
    def __init__(self, only_mean):
        super(kernel_extra_mean_var, self).__init__()
        self.kernel_size = 21
        self.Encoding_Block1 = kernel_extra_Encoding_Block(3, 64)
        self.Encoding_Block2 = kernel_extra_Encoding_Block(64, 64)
        self.Conv_mid = kernel_extra_conv_mid(64, 64)
        self.Decoding_Block1 = kernel_extra_Decoding_Block(64, 64)
        self.Decoding_Block2 = kernel_extra_Decoding_Block(64, 64)
        self.Conv_tail = kernel_extra_conv_tail_mean_var(64, self.kernel_size*self.kernel_size, only_mean=only_mean)

    def forward(self, input):
        output, skip1 = self.Encoding_Block1(input)
        output, skip2 = self.Encoding_Block2(output)
        output = self.Conv_mid(output)
        output = self.Decoding_Block1(output, skip2)
        output = self.Decoding_Block2(output, skip1)
        output, kernel, mean, var = self.Conv_tail(output)

        return output, kernel, mean, var


if __name__ == '__main__':
    input1 = torch.rand(2, 3, 128, 128).cuda()
    net = RCAN(3).cuda()
    output, kernel = net(input1)
    # print(net)
    print(output.size())
    # print("Total number of paramerters in networks is {}  ".format(sum(x.numel() for x in net.parameters())))








