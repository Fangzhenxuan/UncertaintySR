import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
import torch.nn.functional as F
from utils.util import show_image, save_image


class Resblock(nn.Module):
    def __init__(self, HBW):
        super(Resblock, self).__init__()
        self.block1 = nn.Sequential(nn.Conv2d(HBW, HBW, kernel_size=3, stride=1, padding=1),
                                    nn.ReLU(),
                                    nn.Conv2d(HBW, HBW, kernel_size=3, stride=1, padding=1))
        self.block2 = nn.Sequential(nn.Conv2d(HBW, HBW, kernel_size=3, stride=1, padding=1),
                                    nn.ReLU(),
                                    nn.Conv2d(HBW, HBW, kernel_size=3, stride=1, padding=1))

    def forward(self, x):
        tem = x
        r1 = self.block1(x)
        out = r1 + tem
        r2 = self.block2(out)
        out = r2 + out
        return out


class Encoding(nn.Module):
    def __init__(self):
        super(Encoding, self).__init__()
        self.E1 = nn.Sequential(nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),
                                nn.ReLU(),
                                nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),
                                nn.ReLU()
                                )
        self.E2 = nn.Sequential(nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
                                nn.ReLU(),
                                nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
                                nn.ReLU()
                                )
        self.E3 = nn.Sequential(nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
                                nn.ReLU(),
                                nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
                                nn.ReLU()
                                )
        self.E4 = nn.Sequential(nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
                                nn.ReLU(),
                                nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
                                nn.ReLU()
                                )
        self.E5 = nn.Sequential(nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
                                nn.ReLU(),
                                nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
                                nn.ReLU()
                                )

    def forward(self, x):
        ## encoding blocks
        E1 = self.E1(x)
        E2 = self.E2(F.avg_pool2d(E1, kernel_size=2, stride=2))
        E3 = self.E3(F.avg_pool2d(E2, kernel_size=2, stride=2))
        E4 = self.E4(F.avg_pool2d(E3, kernel_size=2, stride=2))
        E5 = self.E5(F.avg_pool2d(E4, kernel_size=2, stride=2))
        return E1, E2, E3, E4, E5


class Decoding(nn.Module):
    def __init__(self, Ch):
        super(Decoding, self).__init__()
        self.upMode = 'bilinear'
        self.Ch = Ch
        self.D1 = nn.Sequential(nn.Conv2d(in_channels=128+128, out_channels=128, kernel_size=3, stride=1, padding=1),
                                nn.ReLU(),
                                nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
                                nn.ReLU()
                                )
        self.D2 = nn.Sequential(nn.Conv2d(in_channels=128+64, out_channels=64, kernel_size=3, stride=1, padding=1),
                                nn.ReLU(),
                                nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
                                nn.ReLU()
                                )
        self.D3 = nn.Sequential(nn.Conv2d(in_channels=64+64, out_channels=64, kernel_size=3, stride=1, padding=1),
                                nn.ReLU(),
                                nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
                                nn.ReLU()
                                )
        self.D4 = nn.Sequential(nn.Conv2d(in_channels=64+32, out_channels=32, kernel_size=3, stride=1, padding=1),
                                nn.ReLU(),
                                nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),
                                nn.ReLU()
                                )

        self.w_generator = nn.Sequential(nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),
                                         nn.ReLU(),
                                         nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),
                                         nn.ReLU(),
                                         nn.Conv2d(in_channels=32, out_channels=self.Ch, kernel_size=1, stride=1, padding=0)
                                         )
        self.u_generator = nn.Sequential(nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),
                                         nn.ReLU(),
                                         nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),
                                         nn.ReLU(),
                                         nn.Conv2d(in_channels=32, out_channels=self.Ch, kernel_size=1, stride=1, padding=0)
                                         )

    def forward(self, E1, E2, E3, E4, E5):
        # decoding blocks
        D1 = self.D1(torch.cat([E4, F.interpolate(E5, scale_factor=2, mode=self.upMode, align_corners=False)], dim=1))
        D2 = self.D2(torch.cat([E3, F.interpolate(D1, scale_factor=2, mode=self.upMode, align_corners=False)], dim=1))
        D3 = self.D3(torch.cat([E2, F.interpolate(D2, scale_factor=2, mode=self.upMode, align_corners=False)], dim=1))
        D4 = self.D4(torch.cat([E1, F.interpolate(D3, scale_factor=2, mode=self.upMode, align_corners=False)], dim=1))

        # estimating the regularization parameters w
        w = self.w_generator(D4)
        u = self.u_generator(D4)

        return w, u


class DLSM(nn.Module):
    def __init__(self, scale):
        super(DLSM, self).__init__()
        self.Ch = 3
        self.T = 4
        self.up_factor = scale

        # Encoding blocks
        self.Encoding = Encoding()

        # Decoding blocks
        self.Decoding = Decoding(Ch=self.Ch)

        # Dense connection
        self.conv = nn.Conv2d(self.Ch, 32, kernel_size=3, stride=1, padding=1)
        self.Den_con1 = nn.Conv2d(32, 32, kernel_size=1, stride=1, padding=0)
        self.Den_con2 = nn.Conv2d(32 * 2, 32, kernel_size=1, stride=1, padding=0)
        self.Den_con3 = nn.Conv2d(32 * 3, 32, kernel_size=1, stride=1, padding=0)
        self.Den_con4 = nn.Conv2d(32 * 4, 32, kernel_size=1, stride=1, padding=0)
        # self.Den_con5 = nn.Conv2d(32 * 5, 32, kernel_size=1, stride=1, padding=0)
        # self.Den_con6 = nn.Conv2d(32 * 6, 32, kernel_size=1, stride=1, padding=0)

        self.delta_0 = Parameter(torch.ones(1), requires_grad=True)
        self.delta_1 = Parameter(torch.ones(1), requires_grad=True)
        self.delta_2 = Parameter(torch.ones(1), requires_grad=True)
        self.delta_3 = Parameter(torch.ones(1), requires_grad=True)
        # self.delta_4 = Parameter(torch.ones(1), requires_grad=True)
        # self.delta_5 = Parameter(torch.ones(1), requires_grad=True)

        self._initialize_weights()
        torch.nn.init.normal_(self.delta_0, mean=0.1, std=0.01)
        torch.nn.init.normal_(self.delta_1, mean=0.1, std=0.01)
        torch.nn.init.normal_(self.delta_2, mean=0.1, std=0.01)
        torch.nn.init.normal_(self.delta_3, mean=0.1, std=0.01)
        # torch.nn.init.normal_(self.delta_4, mean=0.1, std=0.01)
        # torch.nn.init.normal_(self.delta_5, mean=0.1, std=0.01)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight.data)
                nn.init.constant_(m.bias.data, 0.0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data)
                nn.init.constant_(m.bias.data, 0.0)

    def recon(self, res, w, u, i):
        if i == 0 :
            delta = self.delta_0
        elif i == 1:
            delta = self.delta_1
        elif i == 2:
            delta = self.delta_2
        elif i == 3:
            delta = self.delta_3
        # elif i == 4:
        #     delta = self.delta_4
        # elif i == 5:
        #     delta = self.delta_5

        w = w *delta

        case1 = torch.where(res < (u - w), res + w, res)
        mask1 = torch.where((res - case1) != 0, 1, 0)

        case3 = torch.where(res > (u + w), res - w, res)
        mask3 = torch.where((res - case3) != 0, 1, 0)

        mask2 = torch.ones_like(mask1) - mask1 - mask3

        Xt = case1 * mask1 + u * mask2 + case3 * mask3

        return Xt

    def conv_AT(self, input_img, kernel, kernel_size, scale):
        # input_img: tensor[B, C, H, W]
        # kernel: tensor[B, size, size]

        # output_img: tensor[B, C, H*s, W*s]
        input_img = F.interpolate(input_img, scale_factor=scale, mode='bicubic', align_corners=False)
        output_img = torch.zeros_like(input_img).cuda()

        # show_image(kernel)
        for b in range(input_img.size(0)):
            conv_kernel = kernel[b:b + 1, :, :]  # tensor[1, kernel_size, kernel_size]
            conv_kernel = conv_kernel.expand(input_img.size(1), 1, conv_kernel.size(1),
                                             conv_kernel.size(2))  # tensor[C, 1, kernel_size, kernel_size]
            conv_input_img = input_img[b:b + 1, :, :, :]  # tensor[1, C, kernel_size, kernel_size]

            output_conv_img = F.conv_transpose2d(conv_input_img, conv_kernel, stride=1, groups=input_img.size(1),
                                                 padding=int((kernel_size - 1) / 2))

            output_img[b:b + 1, :, :, :] = output_conv_img

        return output_img

    def conv_A(self, input_img, kernel, kernel_size, scale):
        # input_img: tensor[B, C, H, W]
        # kernel: tensor[B, size, size]

        # output_img: tensor[B, C, H, W]
        output_img = torch.zeros_like(input_img).cuda()

        # show_image(kernel)
        for b in range(input_img.size(0)):
            conv_kernel = kernel[b:b + 1, :, :]  # tensor[1, kernel_size, kernel_size]
            conv_kernel = conv_kernel.expand(input_img.size(1), 1, conv_kernel.size(1),
                                             conv_kernel.size(2))  # tensor[C, 1, kernel_size, kernel_size]
            conv_input_img = input_img[b:b + 1, :, :, :]  # tensor[1, C, kernel_size, kernel_size]

            output_conv_img = F.conv2d(conv_input_img, conv_kernel, stride=1, groups=input_img.size(1),
                                       padding=int((kernel_size - 1) / 2))

            output_img[b:b + 1, :, :, :] = output_conv_img

        output_img = F.interpolate(output_img, scale_factor=1 / scale, mode='bicubic', align_corners=False)
        return output_img

    def forward(self, y, kernel):
        # y: tenor[B, 3, H, W]
        # kernel: tensor[B, 21, 21]
        Xt = self.conv_AT(y, kernel, kernel_size=kernel.size(2), scale=self.up_factor)

        feature_list = []

        for i in range(self.T):
            AXt = self.conv_A(Xt, kernel, kernel_size=kernel.size(2), scale=self.up_factor)  # y = Ax
            Res = self.conv_AT(y - AXt, kernel, kernel_size=kernel.size(2), scale=self.up_factor)  # A^T * (y - Ax)
            Res = Res + Xt

            fea = self.conv(Xt)

            if i == 0:
                feature_list.append(fea)
                fufea = self.Den_con1(fea)
            elif i == 1:
                feature_list.append(fea)
                fufea = self.Den_con2(torch.cat(feature_list, 1))
            elif i == 2:
                feature_list.append(fea)
                fufea = self.Den_con3(torch.cat(feature_list, 1))
            elif i == 3:
                feature_list.append(fea)
                fufea = self.Den_con4(torch.cat(feature_list, 1))

            B, C, h, w = fufea.size()
            bottom = (16 - h % 16) % 16
            right = (16 - w % 16) % 16
            padding = torch.nn.ReflectionPad2d((0, right, 0, bottom))
            fufea_pad = padding(fufea)

            E1, E2, E3, E4, E5 = self.Encoding(fufea_pad)
            W, U = self.Decoding(E1, E2, E3, E4, E5)

            W = W[:, :, 0:h, 0:w]
            U = U[:, :, 0:h, 0:w]

            # Reconstructing
            Xt = self.recon(Res, W, U, i)

        return Xt


if __name__ == '__main__':

    input1 = torch.rand(4, 3, 157, 199).cuda()
    kernel = torch.rand(4, 21, 21).cuda()

    net = DLSM(scale=4).cuda()
    with torch.no_grad():
        output = net(input1, kernel)

    print(output.size())



