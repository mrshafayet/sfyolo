import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from einops import rearrange
    from einops.layers.torch import Rearrange
except:
    pass


try:
    from typing import Optional, Sequence
    # from mmengine.model import BaseModule
    # from mmcv.cnn import ConvModule, build_norm_layer
    from ultralytics.nn.modules.conv import autopad
    from torch.cuda.amp import autocast
    # from timm.models.layers import DropPath
except:
    pass

try:
    import torch_dct as dct
    # from mmcv.ops.modulated_deform_conv import ModulatedDeformConv2d, modulated_deform_conv2d
except:
    pass


from ultralytics.nn.modules.block import C2f, C3, SPPF, C3k2, C3k, Bottleneck
from ultralytics.nn.other_modules.conv import SCConv, ScConv, AKConv, DropPath, DySnakeConv, GhostModuleV3, PConv
from ultralytics.nn.other_modules.attention import Attention_xy, OmniAttention

from ultralytics.nn.modules.conv import (Conv, Conv2, DWConv, DWConvTranspose2d, ConvTranspose, RepConv, LightConv,
                                         GhostConv)




# _____________________________________RCRep2A____________________________#
class SFBlock(nn.Module):
    """Standard block by xy."""
    def __init__(self, c1, c2, shortcut=True, e=0.5):
        super().__init__()
        c_ = int(c2 * e)  # mid channels
        self.cv1 = RepConv(c1, c_, 3)
        self.att = Attention_xy(c_)
        self.cv2 = RepConv(c_, c2, 3)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        return x + self.cv2(self.att(self.cv1(x))) if self.add else self.cv2(self.att(self.cv1(x)))


class SF_model(nn.Module):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__()
        self.c = int(c2 * e)  # hidden channels

        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        # self.cv_m = Conv(self.c, self.c, 3)
        self.cv_m = GhostConv(self.c, self.c, 3)
        self.cv2 = Conv((3 + n) * self.c, c2, 1)
        self.m = nn.ModuleList(SFBlock(self.c, self.c, shortcut, e=e) for _ in range(n))

    def forward(self, x):
        x1, x2 = self.cv1(x).chunk(2, 1)
        y = list([self.cv_m(x1), x1, x2])
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


# _____________________________________SPPF_WD____________________________#
class SPPF_WD(SPPF):
    def __init__(self, c1, c2, k=5):
        """
        Initializes the SPPF layer with given input/output channels and kernel size.

        This module is equivalent to SPP(k=(5, 9, 13)).
        """
        super().__init__(c1, c2, k)
        c_ = c1 // 2  # hidden channels
        self.cv1 = LightConv(c1, c_, 1, 1)
        # self.cv1 = Conv(c1, c_, 1, 1)
        # self.cv2 = ConvTranspose(c_ * 4, c2, 1, 1)
        self.cv2 = Conv(c_ * 4, c2, 1, 1)
        self.m = nn.ModuleList([nn.MaxPool2d(kernel_size=k-2*i, stride=1, padding=(k-2*i) // 2) for i in range(3)])

    def forward(self, x):
        """Forward pass through Ghost Convolution block."""
        y = [self.cv1(x)]
        y.extend(self.m[i](y[-1]) for i in range(3))
        return self.cv2(torch.cat(y, 1))


# _____________________________________HLADP____________________________#
def conv_bn(in_channels, out_channels, kernel_size, stride, padding, groups=1, bias=False):
    '''Basic cell for rep-style block, including conv and bn'''
    result = nn.Sequential()
    result.add_module('conv', nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                                        kernel_size=kernel_size, stride=stride, padding=padding, groups=groups,
                                        bias=bias))
    result.add_module('bn', nn.BatchNorm2d(num_features=out_channels))
    return result


class RepVGGBlock(nn.Module):
    '''RepVGGBlock is a basic rep-style block, including training and deploy status
    This code is based on https://github.com/DingXiaoH/RepVGG/blob/main/repvgg.py
    '''

    def __init__(self, in_channels, out_channels, kernel_size=3,
                 stride=1, padding=1, dilation=1, groups=1, padding_mode='zeros', deploy=False, use_se=False):
        super(RepVGGBlock, self).__init__()
        """ Initialization of the class.
        Args:
            in_channels (int): Number of channels in the input image
            out_channels (int): Number of channels produced by the convolution
            kernel_size (int or tuple): Size of the convolving kernel
            stride (int or tuple, optional): Stride of the convolution. Default: 1
            padding (int or tuple, optional): Zero-padding added to both sides of
                the input. Default: 1
            dilation (int or tuple, optional): Spacing between kernel elements. Default: 1
            groups (int, optional): Number of blocked connections from input
                channels to output channels. Default: 1
            padding_mode (string, optional): Default: 'zeros'
            deploy: Whether to be deploy status or training status. Default: False
            use_se: Whether to use se. Default: False
        """
        self.deploy = deploy
        self.groups = groups
        self.in_channels = in_channels
        self.out_channels = out_channels

        assert kernel_size == 3
        assert padding == 1

        padding_11 = padding - kernel_size // 2

        self.nonlinearity = nn.ReLU()

        if use_se:
            raise NotImplementedError("se block not supported yet")
        else:
            self.se = nn.Identity()

        if deploy:
            self.rbr_reparam = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                         stride=stride,
                                         padding=padding, dilation=dilation, groups=groups, bias=True,
                                         padding_mode=padding_mode)

        else:
            self.rbr_identity = nn.BatchNorm2d(
                num_features=in_channels) if out_channels == in_channels and stride == 1 else None
            self.rbr_dense = conv_bn(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                     stride=stride, padding=padding, groups=groups)
            self.rbr_1x1 = conv_bn(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=stride,
                                   padding=padding_11, groups=groups)

    def forward(self, inputs):
        '''Forward process'''
        if hasattr(self, 'rbr_reparam'):
            return self.nonlinearity(self.se(self.rbr_reparam(inputs)))

        if self.rbr_identity is None:
            id_out = 0
        else:
            id_out = self.rbr_identity(inputs)

        return self.nonlinearity(self.se(self.rbr_dense(inputs) + self.rbr_1x1(inputs) + id_out))

    def get_equivalent_kernel_bias(self):
        kernel3x3, bias3x3 = self._fuse_bn_tensor(self.rbr_dense)
        kernel1x1, bias1x1 = self._fuse_bn_tensor(self.rbr_1x1)
        kernelid, biasid = self._fuse_bn_tensor(self.rbr_identity)
        return kernel3x3 + self._pad_1x1_to_3x3_tensor(kernel1x1) + kernelid, bias3x3 + bias1x1 + biasid

    def _pad_1x1_to_3x3_tensor(self, kernel1x1):
        if kernel1x1 is None:
            return 0
        else:
            return torch.nn.functional.pad(kernel1x1, [1, 1, 1, 1])

    def _fuse_bn_tensor(self, branch):
        if branch is None:
            return 0, 0
        if isinstance(branch, nn.Sequential):
            kernel = branch.conv.weight
            running_mean = branch.bn.running_mean
            running_var = branch.bn.running_var
            gamma = branch.bn.weight
            beta = branch.bn.bias
            eps = branch.bn.eps
        else:
            assert isinstance(branch, nn.BatchNorm2d)
            if not hasattr(self, 'id_tensor'):
                input_dim = self.in_channels // self.groups
                kernel_value = np.zeros((self.in_channels, input_dim, 3, 3), dtype=np.float32)
                for i in range(self.in_channels):
                    kernel_value[i, i % input_dim, 1, 1] = 1
                self.id_tensor = torch.from_numpy(kernel_value).to(branch.weight.device)
            kernel = self.id_tensor
            running_mean = branch.running_mean
            running_var = branch.running_var
            gamma = branch.weight
            beta = branch.bias
            eps = branch.eps
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * t, beta - running_mean * gamma / std

    def switch_to_deploy(self):
        if hasattr(self, 'rbr_reparam'):
            return
        kernel, bias = self.get_equivalent_kernel_bias()
        self.rbr_reparam = nn.Conv2d(in_channels=self.rbr_dense.conv.in_channels,
                                     out_channels=self.rbr_dense.conv.out_channels,
                                     kernel_size=self.rbr_dense.conv.kernel_size, stride=self.rbr_dense.conv.stride,
                                     padding=self.rbr_dense.conv.padding, dilation=self.rbr_dense.conv.dilation,
                                     groups=self.rbr_dense.conv.groups, bias=True)
        self.rbr_reparam.weight.data = kernel
        self.rbr_reparam.bias.data = bias
        for para in self.parameters():
            para.detach_()
        self.__delattr__('rbr_dense')
        self.__delattr__('rbr_1x1')
        if hasattr(self, 'rbr_identity'):
            self.__delattr__('rbr_identity')
        if hasattr(self, 'id_tensor'):
            self.__delattr__('id_tensor')
        self.deploy = True


class RepVGGBlocks(nn.Module):
    def __init__(self, channel, fuse_block_num=3) -> None:
        super().__init__()

        self.conv = nn.Sequential(
            *[RepVGGBlock(channel, channel) for _ in range(fuse_block_num)],
        )

    def forward(self, x):
        return self.conv(x)


class align_3In(nn.Module):
    def __init__(self, c1, c2, size_sign):
        super().__init__()
        c_ = c2//2
        self.ss = size_sign
        self.convs = nn.ModuleList(
            [nn.Conv2d(i, c_, kernel_size=3, stride=1, padding=1) for i in c1])
        self.conv2 = Conv(3*c_, c2)

    def forward(self, x):
        y = list([])
        target_size = x[self.ss].shape[2:]

        for i, x1 in enumerate(x):
            if x1.shape[-1] > target_size[-1]:
                x1 = F.adaptive_avg_pool2d(x1, (target_size[0], target_size[1]))
            elif x1.shape[-1] < target_size[-1]:
                x1 = F.interpolate(x1, size=(target_size[0], target_size[1]),
                                  mode='bilinear', align_corners=True)

            y.append(self.convs[i](x1))
        return self.conv2(torch.cat(y, dim=1))



__all__ = ['align_3In', 'RepVGGBlocks', 'SPPF_WD', 'SF_model', 'GhostModuleV3', 'PConv'
           ]


if __name__ == '__main__':
    x = torch.randn(1, 32, 16, 16)
    # model = RCRep2A_ScConv(32, 64)
    # print(model(x).shape)

