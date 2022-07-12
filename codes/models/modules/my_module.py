
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


class kernel_extra_conv_tail(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(kernel_extra_conv_tail, self).__init__()
        self.body = nn.Sequential(
                        nn.Conv2d(in_ch, in_ch, kernel_size=3, stride=1, padding=1),
                        nn.ReLU(),
                        nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1)
                        )
        self.kernel_size = 21

    def forward(self, input):
        output = self.body(input)
        output = nn.Softmax2d()(output)
        kernel = output.mean(dim=[2, 3], keepdim=True)
        kernel = kernel.view(-1, self.kernel_size, self.kernel_size)
        return output, kernel


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


class kernel_extra(nn.Module):
    def __init__(self):
        super(kernel_extra, self).__init__()
        self.kernel_size = 21
        self.Encoding_Block1 = kernel_extra_Encoding_Block(3, 64)
        self.Encoding_Block2 = kernel_extra_Encoding_Block(64, 64)
        self.Conv_mid = kernel_extra_conv_mid(64, 64)
        self.Decoding_Block1 = kernel_extra_Decoding_Block(64, 64)
        self.Decoding_Block2 = kernel_extra_Decoding_Block(64, 64)
        self.Conv_tail = kernel_extra_conv_tail(64, self.kernel_size*self.kernel_size)

    def forward(self, input):
        output, skip1 = self.Encoding_Block1(input)
        output, skip2 = self.Encoding_Block2(output)
        output = self.Conv_mid(output)
        output = self.Decoding_Block1(output, skip2)
        output = self.Decoding_Block2(output, skip1)
        output, kernel = self.Conv_tail(output)

        return output, kernel


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

'''
class net_conv_mid(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(net_conv_mid, self).__init__()
        self.body = nn.Sequential(
                        nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1),
                        nn.ReLU(),
                        nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1),
                        nn.ReLU(),
                        nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1),
                        nn.ReLU()
                        )

    def forward(self, input):
        output = self.body(input)
        return output
'''

'''
class KOALA_module(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_ch):
        super(KOALA_module, self).__init__()
        self.Conv_head = nn.Sequential(
                            nn.ReLU(),
                            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1),
                            nn.ReLU(),
                            nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1)
                            )
        self.Conv_m = nn.Sequential(
                            nn.Conv2d(kernel_ch, out_ch, kernel_size=3, stride=1, padding=1),
                            nn.ReLU(),
                            nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1)
                            )
        self.Conv_k = nn.Sequential(
                            nn.Conv2d(kernel_ch, kernel_ch, kernel_size=1, stride=1),
                            nn.ReLU(),
                            nn.Conv2d(kernel_ch, 49, kernel_size=1, stride=1)
                            )

    def forward(self, input, kernel_fea):
        output = self.Conv_head(input)
        #m = self.Conv_m(kernel_fea)
        #output = torch.mul(output, m)       # output: [B, 49, H, W]
        k = self.Conv_k(kernel_fea)
        k = nn.Softmax2d()(k)
        k = k.mean(dim=[2, 3], keepdim=True)
        k = k.view(-1, 7, 7)      # k:[B, 7, 7]
        output = conv_k(output, k, 7)
        output = output + input

        return output
'''

'''
class KRCAN(nn.Module):
    def __init__(self):
        super(KRCAN, self).__init__()
        self.Conv_head = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        self.KOALA = KOALA_module(in_ch=64, out_ch=64, kernel_ch=64)
        self.RCAN = RCAN(64)

    def forward(self, input, kernel_fea):
        output = self.Conv_head(input)
        output = self.KOALA(output, kernel_fea)
        output = self.RCAN(output)

        return output
'''

'''
def conv_k(input_img, kernel, kernel_size):
    # input_img: tensor[B, C, H, W]
    # kernel: tensor[B, size, size]

    # padding
    input_img_pad = nn.ReflectionPad2d(int((kernel_size - 1) / 2))(input_img)
    # output_img: tensor[B, C, H, W]
    output_img = torch.zeros(input_img.size()[0], input_img.size()[1], input_img.size()[2], input_img.size()[3]).cuda()

    # show_image(kernel)
    for b in range(input_img.size(0)):
        conv_kernel = kernel[b:b + 1, :, :]     # tensor[1, kernel_size, kernel_size]
        conv_kernel = conv_kernel.expand(input_img.size(1), 1, conv_kernel.size(1), conv_kernel.size(2)) # tensor[C, 1, kernel_size, kernel_size]
        conv_input_img = input_img_pad[b:b + 1, :, :, :]  # tensor[1, C, kernel_size, kernel_size]

        output_conv_img = nn.functional.conv2d(conv_input_img, conv_kernel, bias=None, groups=input_img.size(1))

        output_img[b:b + 1, :, :, :] = output_conv_img

    return output_img

'''
if __name__ == '__main__':
    input1 = torch.rand(2, 3, 128, 128).cuda()
    net = RCAN(3).cuda()
    output, kernel = net(input1)
    # print(net)
    print(output.size())
    # print("Total number of paramerters in networks is {}  ".format(sum(x.numel() for x in net.parameters())))








