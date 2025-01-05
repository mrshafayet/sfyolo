import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.conv import _ConvNd
from torch.nn.modules.utils import _pair
from typing import Optional, List, Tuple

from ultralytics.nn.modules.conv import Conv, DWConv, GhostConv, LightConv, RepConv, autopad
from ultralytics.nn.modules import TransformerBlock

try:
    import math
    from einops import rearrange
except ImportError:
    pass


__all__ = ['Conv', 'DWConv', 'GhostConv', 'LightConv', 'RepConv', 'TransformerBlock', 'ScConv', 'SCConv',
           'AKConv', 'DS_Conv', 'DySnakeConv', 'HWD', 'SPD', 'CARAFE', 'DySample', 'DropPath', 'GhostModuleV2',
           'GhostModuleV3', 'PConv']


class PConv(nn.Module):
    def __init__(self, c1, c2, n_div=4, forward='split_cat'):
        super().__init__()
        self.dim_conv3 = c1 // n_div
        self.dim_untouched = c1 - self.dim_conv3
        self.partial_conv3 = nn.Conv2d(self.dim_conv3, self.dim_conv3, 3, 1, 1, bias=False)
        self.conv = Conv(c1, c2, k=1)

        if forward == 'slicing':
            self.forward = self.forward_slicing
        elif forward == 'split_cat':
            self.forward = self.forward_split_cat
        else:
            raise NotImplementedError

    def forward_slicing(self, x):
        # only for inference
        x = x.clone()  # !!! Keep the original input intact for the residual connection later
        x[:, :self.dim_conv3, :, :] = self.partial_conv3(x[:, :self.dim_conv3, :, :])
        x = self.conv(x)
        return x

    def forward_split_cat(self, x):
        # for training/inference
        x1, x2 = torch.split(x, [self.dim_conv3, self.dim_untouched], dim=1)
        x1 = self.partial_conv3(x1)
        x = torch.cat((x1, x2), 1)
        x = self.conv(x)
        return x


def _make_divisible(v, divisor, min_value=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


def hard_sigmoid(x, inplace: bool = False):
    if inplace:
        return x.add_(3.).clamp_(0., 6.).div_(6.)
    else:
        return F.relu6(x + 3.) / 6.


class SqueezeExcite(nn.Module):
    def __init__(self, in_chs, se_ratio=0.25, reduced_base_chs=None,
                 act_layer=nn.ReLU, gate_fn=hard_sigmoid, divisor=4, **_):
        super(SqueezeExcite, self).__init__()
        self.gate_fn = gate_fn
        reduced_chs = _make_divisible((reduced_base_chs or in_chs) * se_ratio, divisor)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_reduce = nn.Conv2d(in_chs, reduced_chs, 1, bias=True)
        self.act1 = act_layer(inplace=True)
        self.conv_expand = nn.Conv2d(reduced_chs, in_chs, 1, bias=True)

    def forward(self, x):
        x_se = self.avg_pool(x)
        x_se = self.conv_reduce(x_se)
        x_se = self.act1(x_se)
        x_se = self.conv_expand(x_se)
        x = x * self.gate_fn(x_se)
        return x


class ConvBnAct(nn.Module):
    def __init__(self, in_chs, out_chs, kernel_size,
                 stride=1, act_layer=nn.ReLU):
        super(ConvBnAct, self).__init__()
        self.conv = nn.Conv2d(in_chs, out_chs, kernel_size, stride, kernel_size // 2, bias=False)
        self.bn1 = nn.BatchNorm2d(out_chs)
        self.act1 = act_layer(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn1(x)
        x = self.act1(x)
        return x


def gcd(a, b):
    if a < b:
        a, b = b, a
    while (a % b != 0):
        c = a % b
        a = b
        b = c
    return b


def MyNorm(dim):
    return nn.GroupNorm(1, dim)


class GhostModuleV3(nn.Module):
    def __init__(self, inp, oup, kernel_size=1, ratio=2, dw_size=3, stride=1, relu=True, mode='ori', args=None):
        super(GhostModuleV3, self).__init__()
        # self.args=args
        mode = 'ori_shortcut_mul_conv15'
        self.mode = mode
        self.gate_loc = 'before'

        self.inter_mode = 'nearest'
        self.scale = 1.0

        self.infer_mode = False
        self.num_conv_branches = 3
        self.dconv_scale = True
        self.gate_fn = nn.Sigmoid()

        # if args.gate_fn=='hard_sigmoid':
        #     self.gate_fn=hard_sigmoid
        # elif args.gate_fn=='sigmoid':
        #     self.gate_fn=nn.Sigmoid()
        # elif args.gate_fn=='relu':
        #     self.gate_fn=nn.ReLU()
        # elif args.gate_fn=='clip':
        #     self.gate_fn=myclip
        # elif args.gate_fn=='tanh':
        #     self.gate_fn=nn.Tanh()

        if self.mode in ['ori']:
            self.oup = oup
            init_channels = math.ceil(oup / ratio)
            new_channels = init_channels * (ratio - 1)
            if self.infer_mode:
                self.primary_conv = nn.Sequential(
                    nn.Conv2d(inp, init_channels, kernel_size, stride, kernel_size // 2, bias=False),
                    nn.BatchNorm2d(init_channels),
                    nn.ReLU(inplace=True) if relu else nn.Sequential(),
                )
                self.cheap_operation = nn.Sequential(
                    nn.Conv2d(init_channels, new_channels, dw_size, 1, dw_size // 2, groups=init_channels, bias=False),
                    nn.BatchNorm2d(new_channels),
                    nn.ReLU(inplace=True) if relu else nn.Sequential(),
                )
            else:
                self.primary_rpr_skip = nn.BatchNorm2d(inp) \
                    if inp == init_channels and stride == 1 else None
                primary_rpr_conv = list()
                for _ in range(self.num_conv_branches):
                    primary_rpr_conv.append(
                        self._conv_bn(inp, init_channels, kernel_size, stride, kernel_size // 2, bias=False))
                self.primary_rpr_conv = nn.ModuleList(primary_rpr_conv)
                # Re-parameterizable scale branch
                self.primary_rpr_scale = None
                if kernel_size > 1:
                    self.primary_rpr_scale = self._conv_bn(inp, init_channels, 1, 1, 0, bias=False)
                self.primary_activation = nn.ReLU(inplace=True) if relu else None

                self.cheap_rpr_skip = nn.BatchNorm2d(init_channels) \
                    if init_channels == new_channels else None
                cheap_rpr_conv = list()
                for _ in range(self.num_conv_branches):
                    cheap_rpr_conv.append(
                        self._conv_bn(init_channels, new_channels, dw_size, 1, dw_size // 2, groups=init_channels,
                                      bias=False))
                self.cheap_rpr_conv = nn.ModuleList(cheap_rpr_conv)
                # Re-parameterizable scale branch
                self.cheap_rpr_scale = None
                if dw_size > 1:
                    self.cheap_rpr_scale = self._conv_bn(init_channels, new_channels, 1, 1, 0, groups=init_channels,
                                                         bias=False)
                self.cheap_activation = nn.ReLU(inplace=True) if relu else None
                self.in_channels = init_channels
                self.groups = init_channels
                self.kernel_size = dw_size

        elif self.mode in ['ori_shortcut_mul_conv15']:
            self.oup = oup
            init_channels = math.ceil(oup / ratio)
            new_channels = init_channels * (ratio - 1)
            self.short_conv = nn.Sequential(
                nn.Conv2d(inp, oup, kernel_size, stride, kernel_size // 2, bias=False),
                nn.BatchNorm2d(oup),
                nn.Conv2d(oup, oup, kernel_size=(1, 5), stride=1, padding=(0, 2), groups=oup, bias=False),
                nn.BatchNorm2d(oup),
                nn.Conv2d(oup, oup, kernel_size=(5, 1), stride=1, padding=(2, 0), groups=oup, bias=False),
                nn.BatchNorm2d(oup),
            )
            if self.infer_mode:
                self.primary_conv = nn.Sequential(
                    nn.Conv2d(inp, init_channels, kernel_size, stride, kernel_size // 2, bias=False),
                    nn.BatchNorm2d(init_channels),
                    nn.ReLU(inplace=True) if relu else nn.Sequential(),
                )
                self.cheap_operation = nn.Sequential(
                    nn.Conv2d(init_channels, new_channels, dw_size, 1, dw_size // 2, groups=init_channels, bias=False),
                    nn.BatchNorm2d(new_channels),
                    nn.ReLU(inplace=True) if relu else nn.Sequential(),
                )
            else:
                self.primary_rpr_skip = nn.BatchNorm2d(inp) \
                    if inp == init_channels and stride == 1 else None
                primary_rpr_conv = list()
                for _ in range(self.num_conv_branches):
                    primary_rpr_conv.append(
                        self._conv_bn(inp, init_channels, kernel_size, stride, kernel_size // 2, bias=False))
                self.primary_rpr_conv = nn.ModuleList(primary_rpr_conv)
                # Re-parameterizable scale branch
                self.primary_rpr_scale = None
                if kernel_size > 1:
                    self.primary_rpr_scale = self._conv_bn(inp, init_channels, 1, 1, 0, bias=False)
                self.primary_activation = nn.ReLU(inplace=True) if relu else None

                self.cheap_rpr_skip = nn.BatchNorm2d(init_channels) \
                    if init_channels == new_channels else None
                cheap_rpr_conv = list()
                for _ in range(self.num_conv_branches):
                    cheap_rpr_conv.append(
                        self._conv_bn(init_channels, new_channels, dw_size, 1, dw_size // 2, groups=init_channels,
                                      bias=False))
                self.cheap_rpr_conv = nn.ModuleList(cheap_rpr_conv)
                # Re-parameterizable scale branch
                self.cheap_rpr_scale = None
                if dw_size > 1:
                    self.cheap_rpr_scale = self._conv_bn(init_channels, new_channels, 1, 1, 0, groups=init_channels,
                                                         bias=False)
                self.cheap_activation = nn.ReLU(inplace=True) if relu else None
                self.in_channels = init_channels
                self.groups = init_channels
                self.kernel_size = dw_size

    def forward(self, x):
        if self.mode in ['ori']:
            if self.infer_mode:
                x1 = self.primary_conv(x)
                x2 = self.cheap_operation(x1)
            else:
                identity_out = 0
                if self.primary_rpr_skip is not None:
                    identity_out = self.primary_rpr_skip(x)
                scale_out = 0
                if self.primary_rpr_scale is not None and self.dconv_scale:
                    scale_out = self.primary_rpr_scale(x)
                x1 = scale_out + identity_out
                for ix in range(self.num_conv_branches):
                    x1 += self.primary_rpr_conv[ix](x)
                if self.primary_activation is not None:
                    x1 = self.primary_activation(x1)

                cheap_identity_out = 0
                if self.cheap_rpr_skip is not None:
                    cheap_identity_out = self.cheap_rpr_skip(x1)
                cheap_scale_out = 0
                if self.cheap_rpr_scale is not None and self.dconv_scale:
                    cheap_scale_out = self.cheap_rpr_scale(x1)
                x2 = cheap_scale_out + cheap_identity_out
                for ix in range(self.num_conv_branches):
                    x2 += self.cheap_rpr_conv[ix](x1)
                if self.cheap_activation is not None:
                    x2 = self.cheap_activation(x2)

            out = torch.cat([x1, x2], dim=1)
            return out

        elif self.mode in ['ori_shortcut_mul_conv15']:
            res = self.short_conv(F.avg_pool2d(x, kernel_size=2, stride=2))

            if self.infer_mode:
                x1 = self.primary_conv(x)
                x2 = self.cheap_operation(x1)
            else:
                identity_out = 0
                if self.primary_rpr_skip is not None:
                    identity_out = self.primary_rpr_skip(x)
                scale_out = 0
                if self.primary_rpr_scale is not None and self.dconv_scale:
                    scale_out = self.primary_rpr_scale(x)
                x1 = scale_out + identity_out
                for ix in range(self.num_conv_branches):
                    x1 += self.primary_rpr_conv[ix](x)
                if self.primary_activation is not None:
                    x1 = self.primary_activation(x1)

                cheap_identity_out = 0
                if self.cheap_rpr_skip is not None:
                    cheap_identity_out = self.cheap_rpr_skip(x1)
                cheap_scale_out = 0
                if self.cheap_rpr_scale is not None and self.dconv_scale:
                    cheap_scale_out = self.cheap_rpr_scale(x1)
                x2 = cheap_scale_out + cheap_identity_out
                for ix in range(self.num_conv_branches):
                    x2 += self.cheap_rpr_conv[ix](x1)
                if self.cheap_activation is not None:
                    x2 = self.cheap_activation(x2)

            out = torch.cat([x1, x2], dim=1)

            if self.gate_loc == 'before':
                return out[:, :self.oup, :, :] * F.interpolate(self.gate_fn(res / self.scale), size=out.shape[-2:],
                                                               mode=self.inter_mode)  # 'nearest'
            #                 return out*F.interpolate(self.gate_fn(res/self.scale),size=out.shape[-1].item(),mode=self.inter_mode) # 'nearest'
            else:
                return out[:, :self.oup, :, :] * self.gate_fn(
                    F.interpolate(res, size=out.shape[-2:], mode=self.inter_mode))
            #                 return out*self.gate_fn(F.interpolate(res,size=out.shape[-1],mode=self.inter_mode))

    def reparameterize(self):
        """ Following works like `RepVGG: Making VGG-style ConvNets Great Again` -
        https://arxiv.org/pdf/2101.03697.pdf. We re-parameterize multi-branched
        architecture used at training time to obtain a plain CNN-like structure
        for inference.
        """
        if self.infer_mode:
            return
        primary_kernel, primary_bias = self._get_kernel_bias_primary()
        self.primary_conv = nn.Conv2d(in_channels=self.primary_rpr_conv[0].conv.in_channels,
                                      out_channels=self.primary_rpr_conv[0].conv.out_channels,
                                      kernel_size=self.primary_rpr_conv[0].conv.kernel_size,
                                      stride=self.primary_rpr_conv[0].conv.stride,
                                      padding=self.primary_rpr_conv[0].conv.padding,
                                      dilation=self.primary_rpr_conv[0].conv.dilation,
                                      groups=self.primary_rpr_conv[0].conv.groups,
                                      bias=True)
        self.primary_conv.weight.data = primary_kernel
        self.primary_conv.bias.data = primary_bias
        self.primary_conv = nn.Sequential(
            self.primary_conv,
            self.primary_activation if self.primary_activation is not None else nn.Sequential()
        )

        cheap_kernel, cheap_bias = self._get_kernel_bias_cheap()
        self.cheap_operation = nn.Conv2d(in_channels=self.cheap_rpr_conv[0].conv.in_channels,
                                         out_channels=self.cheap_rpr_conv[0].conv.out_channels,
                                         kernel_size=self.cheap_rpr_conv[0].conv.kernel_size,
                                         stride=self.cheap_rpr_conv[0].conv.stride,
                                         padding=self.cheap_rpr_conv[0].conv.padding,
                                         dilation=self.cheap_rpr_conv[0].conv.dilation,
                                         groups=self.cheap_rpr_conv[0].conv.groups,
                                         bias=True)
        self.cheap_operation.weight.data = cheap_kernel
        self.cheap_operation.bias.data = cheap_bias

        self.cheap_operation = nn.Sequential(
            self.cheap_operation,
            self.cheap_activation if self.cheap_activation is not None else nn.Sequential()
        )

        # Delete un-used branches
        for para in self.parameters():
            para.detach_()
        if hasattr(self, 'primary_rpr_conv'):
            self.__delattr__('primary_rpr_conv')
        if hasattr(self, 'primary_rpr_scale'):
            self.__delattr__('primary_rpr_scale')
        if hasattr(self, 'primary_rpr_skip'):
            self.__delattr__('primary_rpr_skip')

        if hasattr(self, 'cheap_rpr_conv'):
            self.__delattr__('cheap_rpr_conv')
        if hasattr(self, 'cheap_rpr_scale'):
            self.__delattr__('cheap_rpr_scale')
        if hasattr(self, 'cheap_rpr_skip'):
            self.__delattr__('cheap_rpr_skip')

        self.infer_mode = True

    def _get_kernel_bias_primary(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """ Method to obtain re-parameterized kernel and bias.
        Reference: https://github.com/DingXiaoH/RepVGG/blob/main/repvgg.py#L83

        :return: Tuple of (kernel, bias) after fusing branches.
        """
        # get weights and bias of scale branch
        kernel_scale = 0
        bias_scale = 0
        if self.primary_rpr_scale is not None:
            kernel_scale, bias_scale = self._fuse_bn_tensor(self.primary_rpr_scale)
            # Pad scale branch kernel to match conv branch kernel size.
            pad = self.kernel_size // 2
            kernel_scale = torch.nn.functional.pad(kernel_scale,
                                                   [pad, pad, pad, pad])

        # get weights and bias of skip branch
        kernel_identity = 0
        bias_identity = 0
        if self.primary_rpr_skip is not None:
            kernel_identity, bias_identity = self._fuse_bn_tensor(self.primary_rpr_skip)

        # get weights and bias of conv branches
        kernel_conv = 0
        bias_conv = 0
        for ix in range(self.num_conv_branches):
            _kernel, _bias = self._fuse_bn_tensor(self.primary_rpr_conv[ix])
            kernel_conv += _kernel
            bias_conv += _bias

        kernel_final = kernel_conv + kernel_scale + kernel_identity
        bias_final = bias_conv + bias_scale + bias_identity
        return kernel_final, bias_final

    def _get_kernel_bias_cheap(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """ Method to obtain re-parameterized kernel and bias.
        Reference: https://github.com/DingXiaoH/RepVGG/blob/main/repvgg.py#L83

        :return: Tuple of (kernel, bias) after fusing branches.
        """
        # get weights and bias of scale branch
        kernel_scale = 0
        bias_scale = 0
        if self.cheap_rpr_scale is not None:
            kernel_scale, bias_scale = self._fuse_bn_tensor(self.cheap_rpr_scale)
            # Pad scale branch kernel to match conv branch kernel size.
            pad = self.kernel_size // 2
            kernel_scale = torch.nn.functional.pad(kernel_scale,
                                                   [pad, pad, pad, pad])

        # get weights and bias of skip branch
        kernel_identity = 0
        bias_identity = 0
        if self.cheap_rpr_skip is not None:
            kernel_identity, bias_identity = self._fuse_bn_tensor(self.cheap_rpr_skip)

        # get weights and bias of conv branches
        kernel_conv = 0
        bias_conv = 0
        for ix in range(self.num_conv_branches):
            _kernel, _bias = self._fuse_bn_tensor(self.cheap_rpr_conv[ix])
            kernel_conv += _kernel
            bias_conv += _bias

        kernel_final = kernel_conv + kernel_scale + kernel_identity
        bias_final = bias_conv + bias_scale + bias_identity
        return kernel_final, bias_final

    def _fuse_bn_tensor(self, branch) -> Tuple[torch.Tensor, torch.Tensor]:
        """ Method to fuse batchnorm layer with preceeding conv layer.
        Reference: https://github.com/DingXiaoH/RepVGG/blob/main/repvgg.py#L95

        :param branch:
        :return: Tuple of (kernel, bias) after fusing batchnorm.
        """
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
                kernel_value = torch.zeros((self.in_channels,
                                            input_dim,
                                            self.kernel_size,
                                            self.kernel_size),
                                           dtype=branch.weight.dtype,
                                           device=branch.weight.device)
                for i in range(self.in_channels):
                    kernel_value[i, i % input_dim,
                                    self.kernel_size // 2,
                                    self.kernel_size // 2] = 1
                self.id_tensor = kernel_value
            kernel = self.id_tensor
            running_mean = branch.running_mean
            running_var = branch.running_var
            gamma = branch.weight
            beta = branch.bias
            eps = branch.eps
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * t, beta - running_mean * gamma / std

    def _conv_bn(self, in_channels, out_channels, kernel_size, stride, padding, groups=1, bias=False):
        """ Helper method to construct conv-batchnorm layers.

        :param kernel_size: Size of the convolution kernel.
        :param padding: Zero-padding size.
        :return: Conv-BN module.
        """
        mod_list = nn.Sequential()
        mod_list.add_module('conv', nn.Conv2d(in_channels=in_channels,
                                              out_channels=out_channels,
                                              kernel_size=kernel_size,
                                              stride=stride,
                                              padding=padding,
                                              groups=groups,
                                              bias=bias))
        mod_list.add_module('bn', nn.BatchNorm2d(out_channels))
        return mod_list


class GhostModuleV2(nn.Module):
    def __init__(self, inp, oup, kernel_size=1, ratio=2, dw_size=3, stride=1, args=1):
        super(GhostModuleV2, self).__init__()
        self.gate_fn = nn.Sigmoid()
        self.ages = args
        if self.ages == 1:
            self.oup = oup
            init_channels = math.ceil(oup / ratio)
            new_channels = init_channels * (ratio - 1)
            self.primary_conv = Conv(inp, init_channels, kernel_size, stride, kernel_size // 2)
            self.cheap_operation =Conv(init_channels, new_channels, dw_size, 1, dw_size // 2, g=init_channels,)

        elif self.ages == 2:
            self.oup = oup
            init_channels = math.ceil(oup / ratio)
            new_channels = init_channels * (ratio - 1)
            self.primary_conv = Conv(inp, init_channels, kernel_size, stride, kernel_size // 2)
            self.cheap_operation = Conv(init_channels, new_channels, dw_size, 1, dw_size // 2, g=init_channels, )
            self.short_conv = nn.Sequential(
                Conv(inp, oup, kernel_size, stride, kernel_size // 2),
                nn.Conv2d(oup, oup, kernel_size=(1, 5), stride=1, padding=(0, 2), groups=oup, bias=False),
                nn.BatchNorm2d(oup),
                nn.Conv2d(oup, oup, kernel_size=(5, 1), stride=1, padding=(2, 0), groups=oup, bias=False),
                nn.BatchNorm2d(oup),
            )

    def forward(self, x):
        if self.ages == 1:
            x1 = self.primary_conv(x)
            x2 = self.cheap_operation(x1)
            out = torch.cat([x1, x2], dim=1)
            return out[:, :self.oup, :, :]
        elif  self.ages == 2:
            res = self.short_conv(F.avg_pool2d(x, kernel_size=2, stride=2))
            x1 = self.primary_conv(x)
            x2 = self.cheap_operation(x1)
            out = torch.cat([x1, x2], dim=1)
            return out[:, :self.oup, :, :] * F.interpolate(self.gate_fn(res), size=(out.shape[-2], out.shape[-1]),

                                                           mode='nearest')


class DropPath(nn.Module):
    """
    Implements stochastic depth regularization for neural networks during training.

    Attributes:
        drop_prob (float): Probability of dropping a path during training.
        scale_by_keep (bool): Whether to scale the output by the keep probability.

    Methods:
        forward: Applies stochastic depth to input tensor during training, with optional scaling.

    Examples:
        >>> drop_path = DropPath(drop_prob=0.2, scale_by_keep=True)
        >>> x = torch.randn(32, 64, 224, 224)
        >>> output = drop_path(x)
    """

    def __init__(self, drop_prob=0.0, scale_by_keep=True):
        """Initialize DropPath module for stochastic depth regularization during training."""
        super().__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def forward(self, x):
        """Applies stochastic depth to input tensor during training, with optional scaling."""
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
        if keep_prob > 0.0 and self.scale_by_keep:
            random_tensor.div_(keep_prob)
        return x * random_tensor


def normal_init(module, mean=0, std=1, bias=0):
    if hasattr(module, 'weight') and module.weight is not None:
        nn.init.normal_(module.weight, mean, std)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)


def constant_init(module, val, bias=0):
    if hasattr(module, 'weight') and module.weight is not None:
        nn.init.constant_(module.weight, val)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)


class DySample(nn.Module):
    def __init__(self, c1, scale=2, style='lp', groups=4, dyscope=False):
        super().__init__()
        self.scale = scale
        self.style = style
        self.groups = groups
        assert style in ['lp', 'pl']
        if style == 'pl':
            assert c1 >= scale ** 2 and c1 % scale ** 2 == 0
        assert c1 >= groups and c1 % groups == 0

        if style == 'pl':
            in_channels = c1 // scale ** 2
            out_channels = 2 * groups
        else:
            out_channels = 2 * groups * scale ** 2

        self.offset = nn.Conv2d(in_channels, out_channels, 1)
        normal_init(self.offset, std=0.001)
        if dyscope:
            self.scope = nn.Conv2d(in_channels, out_channels, 1, bias=False)
            constant_init(self.scope, val=0.)

        self.register_buffer('init_pos', self._init_pos())

    def _init_pos(self):
        h = torch.arange((-self.scale + 1) / 2, (self.scale - 1) / 2 + 1) / self.scale
        return torch.stack(torch.meshgrid([h, h], indexing='ij')).transpose(1, 2).repeat(1, self.groups, 1).reshape(1, -1, 1, 1)

    def sample(self, x, offset):
        B, _, H, W = offset.shape
        offset = offset.view(B, 2, -1, H, W)
        coords_h = torch.arange(H) + 0.5
        coords_w = torch.arange(W) + 0.5
        coords = torch.stack(torch.meshgrid([coords_w, coords_h], indexing='ij')
                             ).transpose(1, 2).unsqueeze(1).unsqueeze(0).type(x.dtype).to(x.device)
        normalizer = torch.tensor([W, H], dtype=x.dtype, device=x.device).view(1, 2, 1, 1, 1)
        coords = 2 * (coords + offset) / normalizer - 1
        coords = F.pixel_shuffle(coords.view(B, -1, H, W), self.scale).view(
            B, 2, -1, self.scale * H, self.scale * W).permute(0, 2, 3, 4, 1).contiguous().flatten(0, 1)
        return F.grid_sample(x.reshape(B * self.groups, -1, H, W), coords, mode='bilinear',
                             align_corners=False, padding_mode="border").view(B, -1, self.scale * H, self.scale * W)

    def forward_lp(self, x):
        if hasattr(self, 'scope'):
            offset = self.offset(x) * self.scope(x).sigmoid() * 0.5 + self.init_pos
        else:
            offset = self.offset(x) * 0.25 + self.init_pos
        return self.sample(x, offset)

    def forward_pl(self, x):
        x_ = F.pixel_shuffle(x, self.scale)
        if hasattr(self, 'scope'):
            offset = F.pixel_unshuffle(self.offset(x_) * self.scope(x_).sigmoid(), self.scale) * 0.5 + self.init_pos
        else:
            offset = F.pixel_unshuffle(self.offset(x_), self.scale) * 0.25 + self.init_pos
        return self.sample(x, offset)

    def forward(self, x):
        if self.style == 'pl':
            return self.forward_pl(x)
        return self.forward_lp(x)


class CARAFE(nn.Module):
    def __init__(self, c, c_mid=64, k_enc=3, k_up=5, scale=2):
        """ The unofficial implementation of the CARAFE module.
        The details are in "https://arxiv.org/abs/1905.02188".
        Args:
            c: The channel number of the input and the output.
            c_mid: The channel number after compression.
            scale: The expected upsample scale.
            k_up: The size of the reassembly kernel.
            k_enc: The kernel size of the encoder.
        Returns:
            X: The upsampled feature map.
        """
        super(CARAFE, self).__init__()
        self.scale = scale

        self.comp = Conv(c, c_mid)
        self.enc = Conv(c_mid, (scale * k_up) ** 2, k=k_enc, act=False)
        self.pix_shf = nn.PixelShuffle(scale)

        self.upsmp = nn.Upsample(scale_factor=scale, mode='nearest')
        self.unfold = nn.Unfold(kernel_size=k_up, dilation=scale,
                                padding=k_up // 2 * scale)

    def forward(self, x):
        b, c, h, w = x.size()
        h_, w_ = h * self.scale, w * self.scale

        W = self.comp(x)  # b * m * h * w
        W = self.enc(W)  # b * 100 * h * w
        W = self.pix_shf(W)  # b * 25 * h_ * w_
        W = torch.softmax(W, dim=1)  # b * 25 * h_ * w_

        x = self.upsmp(x)  # b * c * h_ * w_
        x = self.unfold(x)  # b * 25c * h_ * w_
        x = x.view(b, c, -1, h_, w_)  # b * 25 * c * h_ * w_

        x = torch.einsum('bkhw,bckhw->bchw', [W, x])  # b * c * h_ * w_
        return x


class SPD(nn.Module):
    # Changing the dimension of the Tensor
    def __init__(self, dimension=1):
        super().__init__()
        self.d = dimension

    def forward(self, x):
         return torch.cat([x[..., ::2, ::2], x[..., 1::2, ::2], x[..., ::2, 1::2], x[..., 1::2, 1::2]], 1)


class HWD(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(HWD, self).__init__()
        from pytorch_wavelets import DWTForward
        self.wt = DWTForward(J=1, mode='zero', wave='haar')
        self.conv = Conv(in_ch * 4, out_ch, 1, 1)

    def forward(self, x):
        yL, yH = self.wt(x)
        y_HL = yH[0][:, :, 0, ::]
        y_LH = yH[0][:, :, 1, ::]
        y_HH = yH[0][:, :, 2, ::]
        x = torch.cat([yL, y_HL, y_LH, y_HH], dim=1)
        x = self.conv(x)

        return x


class DySnakeConv(nn.Module):
    def __init__(self, inc, ouc, k=3) -> None:
        super().__init__()
        c_ = ouc//3
        self.conv_0 = Conv(inc, ouc-2*c_, k)
        self.conv_x = DSConv(inc, c_, 0, k)
        self.conv_y = DSConv(inc, c_, 1, k)

    def forward(self, x):
        return torch.cat([self.conv_0(x), self.conv_x(x), self.conv_y(x)], dim=1)


class DSConv(nn.Module):
    def __init__(self, in_ch, out_ch, morph, kernel_size=3, if_offset=True, extend_scope=1):
        """
        The Dynamic Snake Convolution
        :param in_ch: input channel
        :param out_ch: output channel
        :param kernel_size: the size of kernel
        :param extend_scope: the range to expand (default 1 for this method)
        :param morph: the morphology of the convolution kernel is mainly divided into two types
                        along the x-axis (0) and the y-axis (1) (see the paper for details)
        :param if_offset: whether deformation is required, if it is False, it is the standard convolution kernel
        """
        super(DSConv, self).__init__()
        # use the <offset_conv> to learn the deformable offset
        self.offset_conv = nn.Conv2d(in_ch, 2 * kernel_size, 3, padding=1)
        self.bn = nn.BatchNorm2d(2 * kernel_size)
        self.kernel_size = kernel_size

        # two types of the DSConv (along x-axis and y-axis)
        self.dsc_conv_x = nn.Conv2d(
            in_ch,
            out_ch,
            kernel_size=(kernel_size, 1),
            stride=(kernel_size, 1),
            padding=0,
        )
        self.dsc_conv_y = nn.Conv2d(
            in_ch,
            out_ch,
            kernel_size=(1, kernel_size),
            stride=(1, kernel_size),
            padding=0,
        )

        self.gn = nn.GroupNorm([i + 1 for i in range(4) if out_ch % (i+1) == 0][-1], out_ch)
        self.act = Conv.default_act

        self.extend_scope = extend_scope
        self.morph = morph
        self.if_offset = if_offset

    def forward(self, f):
        offset = self.offset_conv(f)
        offset = self.bn(offset)
        # We need a range of deformation between -1 and 1 to mimic the snake's swing
        offset = torch.tanh(offset)
        input_shape = f.shape
        dsc = DSC(input_shape, self.kernel_size, self.extend_scope, self.morph)
        deformed_feature = dsc.deform_conv(f, offset, self.if_offset)
        if self.morph == 0:
            x = self.dsc_conv_x(deformed_feature.type(f.dtype))
            x = self.gn(x)
            x = self.act(x)
            return x
        else:
            x = self.dsc_conv_y(deformed_feature.type(f.dtype))
            x = self.gn(x)
            x = self.act(x)
            return x


class DSC(object):
    def __init__(self, input_shape, kernel_size, extend_scope, morph):
        self.num_points = kernel_size
        self.width = input_shape[2]
        self.height = input_shape[3]
        self.morph = morph
        self.extend_scope = extend_scope  # offset (-1 ~ 1) * extend_scope

        # define feature map shape
        """
        B: Batch size  C: Channel  W: Width  H: Height
        """
        self.num_batch = input_shape[0]
        self.num_channels = input_shape[1]

    """
    input: offset [B,2*K,W,H]  K: Kernel size (2*K: 2D image, deformation contains <x_offset> and <y_offset>)
    output_x: [B,1,W,K*H]   coordinate map
    output_y: [B,1,K*W,H]   coordinate map
    """

    def _coordinate_map_3D(self, offset, if_offset):
        device = offset.device
        # offset
        y_offset, x_offset = torch.split(offset, self.num_points, dim=1)

        y_center = torch.arange(0, self.width).repeat([self.height])
        y_center = y_center.reshape(self.height, self.width)
        y_center = y_center.permute(1, 0)
        y_center = y_center.reshape([-1, self.width, self.height])
        y_center = y_center.repeat([self.num_points, 1, 1]).float()
        y_center = y_center.unsqueeze(0)

        x_center = torch.arange(0, self.height).repeat([self.width])
        x_center = x_center.reshape(self.width, self.height)
        x_center = x_center.permute(0, 1)
        x_center = x_center.reshape([-1, self.width, self.height])
        x_center = x_center.repeat([self.num_points, 1, 1]).float()
        x_center = x_center.unsqueeze(0)

        if self.morph == 0:
            """
            Initialize the kernel and flatten the kernel
                y: only need 0
                x: -num_points//2 ~ num_points//2 (Determined by the kernel size)
                !!! The related PPT will be submitted later, and the PPT will contain the whole changes of each step
            """
            y = torch.linspace(0, 0, 1)
            x = torch.linspace(
                -int(self.num_points // 2),
                int(self.num_points // 2),
                int(self.num_points),
            )

            y, x = torch.meshgrid(y, x, indexing='ij')
            y_spread = y.reshape(-1, 1)
            x_spread = x.reshape(-1, 1)

            y_grid = y_spread.repeat([1, self.width * self.height])
            y_grid = y_grid.reshape([self.num_points, self.width, self.height])
            y_grid = y_grid.unsqueeze(0)  # [B*K*K, W,H]

            x_grid = x_spread.repeat([1, self.width * self.height])
            x_grid = x_grid.reshape([self.num_points, self.width, self.height])
            x_grid = x_grid.unsqueeze(0)  # [B*K*K, W,H]

            y_new = y_center + y_grid
            x_new = x_center + x_grid

            y_new = y_new.repeat(self.num_batch, 1, 1, 1).to(device)
            x_new = x_new.repeat(self.num_batch, 1, 1, 1).to(device)

            y_offset_new = y_offset.detach().clone()

            if if_offset:
                y_offset = y_offset.permute(1, 0, 2, 3)
                y_offset_new = y_offset_new.permute(1, 0, 2, 3)
                center = int(self.num_points // 2)

                # The center position remains unchanged and the rest of the positions begin to swing
                # This part is quite simple. The main idea is that "offset is an iterative process"
                y_offset_new[center] = 0
                for index in range(1, center):
                    y_offset_new[center + index] = (y_offset_new[center + index - 1] + y_offset[center + index])
                    y_offset_new[center - index] = (y_offset_new[center - index + 1] + y_offset[center - index])
                y_offset_new = y_offset_new.permute(1, 0, 2, 3).to(device)
                y_new = y_new.add(y_offset_new.mul(self.extend_scope))

            y_new = y_new.reshape(
                [self.num_batch, self.num_points, 1, self.width, self.height])
            y_new = y_new.permute(0, 3, 1, 4, 2)
            y_new = y_new.reshape([
                self.num_batch, self.num_points * self.width, 1 * self.height
            ])
            x_new = x_new.reshape(
                [self.num_batch, self.num_points, 1, self.width, self.height])
            x_new = x_new.permute(0, 3, 1, 4, 2)
            x_new = x_new.reshape([
                self.num_batch, self.num_points * self.width, 1 * self.height
            ])
            return y_new, x_new

        else:
            """
            Initialize the kernel and flatten the kernel
                y: -num_points//2 ~ num_points//2 (Determined by the kernel size)
                x: only need 0
            """
            y = torch.linspace(
                -int(self.num_points // 2),
                int(self.num_points // 2),
                int(self.num_points),
            )
            x = torch.linspace(0, 0, 1)

            y, x = torch.meshgrid(y, x, indexing='ij')
            y_spread = y.reshape(-1, 1)
            x_spread = x.reshape(-1, 1)

            y_grid = y_spread.repeat([1, self.width * self.height])
            y_grid = y_grid.reshape([self.num_points, self.width, self.height])
            y_grid = y_grid.unsqueeze(0)

            x_grid = x_spread.repeat([1, self.width * self.height])
            x_grid = x_grid.reshape([self.num_points, self.width, self.height])
            x_grid = x_grid.unsqueeze(0)

            y_new = y_center + y_grid
            x_new = x_center + x_grid

            y_new = y_new.repeat(self.num_batch, 1, 1, 1)
            x_new = x_new.repeat(self.num_batch, 1, 1, 1)

            y_new = y_new.to(device)
            x_new = x_new.to(device)
            x_offset_new = x_offset.detach().clone()

            if if_offset:
                x_offset = x_offset.permute(1, 0, 2, 3)
                x_offset_new = x_offset_new.permute(1, 0, 2, 3)
                center = int(self.num_points // 2)
                x_offset_new[center] = 0
                for index in range(1, center):
                    x_offset_new[center + index] = (x_offset_new[center + index - 1] + x_offset[center + index])
                    x_offset_new[center - index] = (x_offset_new[center - index + 1] + x_offset[center - index])
                x_offset_new = x_offset_new.permute(1, 0, 2, 3).to(device)
                x_new = x_new.add(x_offset_new.mul(self.extend_scope))

            y_new = y_new.reshape(
                [self.num_batch, 1, self.num_points, self.width, self.height])
            y_new = y_new.permute(0, 3, 1, 4, 2)
            y_new = y_new.reshape([
                self.num_batch, 1 * self.width, self.num_points * self.height
            ])
            x_new = x_new.reshape(
                [self.num_batch, 1, self.num_points, self.width, self.height])
            x_new = x_new.permute(0, 3, 1, 4, 2)
            x_new = x_new.reshape([
                self.num_batch, 1 * self.width, self.num_points * self.height
            ])
            return y_new, x_new

    """
    input: input feature map [N,C,D,W,H]；coordinate map [N,K*D,K*W,K*H] 
    output: [N,1,K*D,K*W,K*H]  deformed feature map
    """

    def _bilinear_interpolate_3D(self, input_feature, y, x):
        device = input_feature.device
        y = y.reshape([-1]).float()
        x = x.reshape([-1]).float()

        zero = torch.zeros([]).int()
        max_y = self.width - 1
        max_x = self.height - 1

        # find 8 grid locations
        y0 = torch.floor(y).int()
        y1 = y0 + 1
        x0 = torch.floor(x).int()
        x1 = x0 + 1

        # clip out coordinates exceeding feature map volume
        y0 = torch.clamp(y0, zero, max_y)
        y1 = torch.clamp(y1, zero, max_y)
        x0 = torch.clamp(x0, zero, max_x)
        x1 = torch.clamp(x1, zero, max_x)

        input_feature_flat = input_feature.flatten()
        input_feature_flat = input_feature_flat.reshape(
            self.num_batch, self.num_channels, self.width, self.height)
        input_feature_flat = input_feature_flat.permute(0, 2, 3, 1)
        input_feature_flat = input_feature_flat.reshape(-1, self.num_channels)
        dimension = self.height * self.width

        base = torch.arange(self.num_batch) * dimension
        base = base.reshape([-1, 1]).float()

        repeat = torch.ones([self.num_points * self.width * self.height
                             ]).unsqueeze(0)
        repeat = repeat.float()

        base = torch.matmul(base, repeat)
        base = base.reshape([-1])

        base = base.to(device)

        base_y0 = base + y0 * self.height
        base_y1 = base + y1 * self.height

        # top rectangle of the neighbourhood volume
        index_a0 = base_y0 - base + x0
        index_c0 = base_y0 - base + x1

        # bottom rectangle of the neighbourhood volume
        index_a1 = base_y1 - base + x0
        index_c1 = base_y1 - base + x1

        # get 8 grid values
        value_a0 = input_feature_flat[index_a0.type(torch.int64)].to(device)
        value_c0 = input_feature_flat[index_c0.type(torch.int64)].to(device)
        value_a1 = input_feature_flat[index_a1.type(torch.int64)].to(device)
        value_c1 = input_feature_flat[index_c1.type(torch.int64)].to(device)

        # find 8 grid locations
        y0 = torch.floor(y).int()
        y1 = y0 + 1
        x0 = torch.floor(x).int()
        x1 = x0 + 1

        # clip out coordinates exceeding feature map volume
        y0 = torch.clamp(y0, zero, max_y + 1)
        y1 = torch.clamp(y1, zero, max_y + 1)
        x0 = torch.clamp(x0, zero, max_x + 1)
        x1 = torch.clamp(x1, zero, max_x + 1)

        x0_float = x0.float()
        x1_float = x1.float()
        y0_float = y0.float()
        y1_float = y1.float()

        vol_a0 = ((y1_float - y) * (x1_float - x)).unsqueeze(-1).to(device)
        vol_c0 = ((y1_float - y) * (x - x0_float)).unsqueeze(-1).to(device)
        vol_a1 = ((y - y0_float) * (x1_float - x)).unsqueeze(-1).to(device)
        vol_c1 = ((y - y0_float) * (x - x0_float)).unsqueeze(-1).to(device)

        outputs = (value_a0 * vol_a0 + value_c0 * vol_c0 + value_a1 * vol_a1 +
                   value_c1 * vol_c1)

        if self.morph == 0:
            outputs = outputs.reshape([
                self.num_batch,
                self.num_points * self.width,
                1 * self.height,
                self.num_channels,
            ])
            outputs = outputs.permute(0, 3, 1, 2)
        else:
            outputs = outputs.reshape([
                self.num_batch,
                1 * self.width,
                self.num_points * self.height,
                self.num_channels,
            ])
            outputs = outputs.permute(0, 3, 1, 2)
        return outputs

    def deform_conv(self, input, offset, if_offset):
        y, x = self._coordinate_map_3D(offset, if_offset)
        deformed_feature = self._bilinear_interpolate_3D(input, y, x)
        return deformed_feature


class DS_Conv(nn.Module):
    def __init__(self, c1, c2, k=3, s=1, p=None, g=1, d=1, b=False):
        super(DS_Conv, self).__init__()
        self.DSconv = DSConv2d(c1, c2, kernel_size=(k, k), stride=s, padding=p, groups=g, dilation=d, bias=b)

    def forward(self, x):
        return self.DSconv(x)


# https://github.com/ActiveVisionLab/DSConv/tree/master/truncmodules
class DSConv2d(_ConvNd):  # https://arxiv.org/pdf/1901.01928v1.pdf
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=None, dilation=1, groups=1, padding_mode='zeros', bias=False, block_size=32, KDSBias=False,
                 CDS=False):
        padding = _pair(autopad(kernel_size, padding, dilation))
        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        dilation = _pair(dilation)

        blck_numb = math.ceil(((in_channels) / (block_size * groups)))
        super(DSConv2d, self).__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation,
            False, _pair(0), groups, bias, padding_mode)

        # KDS weight From Paper
        self.intweight = torch.Tensor(out_channels, in_channels, *kernel_size)
        self.alpha = torch.Tensor(out_channels, blck_numb, *kernel_size)

        # KDS bias From Paper
        self.KDSBias = KDSBias
        self.CDS = CDS

        if KDSBias:
            self.KDSb = torch.Tensor(out_channels, blck_numb, *kernel_size)
        if CDS:
            self.CDSw = torch.Tensor(out_channels)
            self.CDSb = torch.Tensor(out_channels)

        self.reset_parameters()

    def get_weight_res(self):
        # Include expansion of alpha and multiplication with weights to include in the convolution layer here
        alpha_res = torch.zeros(self.weight.shape).to(self.alpha.device)

        # Include KDSBias
        if self.KDSBias:
            KDSBias_res = torch.zeros(self.weight.shape).to(self.alpha.device)

        # Handy definitions:
        nmb_blocks = self.alpha.shape[1]
        total_depth = self.weight.shape[1]
        bs = total_depth // nmb_blocks

        llb = total_depth - (nmb_blocks - 1) * bs

        # Casting the Alpha values as same tensor shape as weight
        for i in range(nmb_blocks):
            length_blk = llb if i == nmb_blocks - 1 else bs

            shp = self.alpha.shape  # Notice this is the same shape for the bias as well
            to_repeat = self.alpha[:, i, ...].view(shp[0], 1, shp[2], shp[3]).clone()
            repeated = to_repeat.expand(shp[0], length_blk, shp[2], shp[3]).clone()
            alpha_res[:, i * bs:(i * bs + length_blk), ...] = repeated.clone()

            if self.KDSBias:
                to_repeat = self.KDSb[:, i, ...].view(shp[0], 1, shp[2], shp[3]).clone()
                repeated = to_repeat.expand(shp[0], length_blk, shp[2], shp[3]).clone()
                KDSBias_res[:, i * bs:(i * bs + length_blk), ...] = repeated.clone()

        if self.CDS:
            to_repeat = self.CDSw.view(-1, 1, 1, 1)
            repeated = to_repeat.expand_as(self.weight)
            print(repeated.shape)

        # Element-wise multiplication of alpha and weight
        weight_res = torch.mul(alpha_res, self.weight)
        if self.KDSBias:
            weight_res = torch.add(weight_res, KDSBias_res)
        return weight_res

    def forward(self, input):
        # Get resulting weight
        # weight_res = self.get_weight_res()

        # Returning convolution
        return F.conv2d(input, self.weight, self.bias,
                        self.stride, self.padding, self.dilation,
                        self.groups)


class AKConv(nn.Module):
    def __init__(self, c1, c2, stride=1, num_param=5):
        """
        初始化参数说明：
            inc: 输入通道数, outc: 输出通道数, num_param:（卷积核）参数量, stride = 1:卷积步长默认为1, bias = None：默认无偏执
            """
        super(AKConv, self).__init__()
        self.num_param = num_param
        self.stride = stride
        self.conv = Conv(c1, c2, k=(num_param, 1), s=(num_param, 1))
        self.p_conv = nn.Conv2d(c1, 2 * num_param, kernel_size=3, padding=1, stride=stride)
        nn.init.constant_(self.p_conv.weight, 0)
        self.p_conv.register_full_backward_hook(self._set_lr)

    @staticmethod
    def _set_lr(module, grad_input, grad_output):
        grad_input = (grad_input[i] * 0.1 for i in range(len(grad_input)))
        grad_output = (grad_output[i] * 0.1 for i in range(len(grad_output)))

    def forward(self, x):
        # N is num_param.
        offset = self.p_conv(x)
        dtype = offset.data.type()
        N = offset.size(1) // 2
        # (b, 2N, h, w)
        p = self._get_p(offset, dtype)

        # (b, h, w, 2N)
        p = p.contiguous().permute(0, 2, 3, 1)
        q_lt = p.detach().floor()
        q_rb = q_lt + 1

        q_lt = torch.cat([torch.clamp(q_lt[..., :N], 0, x.size(2) - 1), torch.clamp(q_lt[..., N:], 0, x.size(3) - 1)],
                         dim=-1).long()
        q_rb = torch.cat([torch.clamp(q_rb[..., :N], 0, x.size(2) - 1), torch.clamp(q_rb[..., N:], 0, x.size(3) - 1)],
                         dim=-1).long()
        q_lb = torch.cat([q_lt[..., :N], q_rb[..., N:]], dim=-1)
        q_rt = torch.cat([q_rb[..., :N], q_lt[..., N:]], dim=-1)

        # clip p
        p = torch.cat([torch.clamp(p[..., :N], 0, x.size(2) - 1), torch.clamp(p[..., N:], 0, x.size(3) - 1)], dim=-1)

        # bilinear kernel (b, h, w, N)
        g_lt = (1 + (q_lt[..., :N].type_as(p) - p[..., :N])) * (1 + (q_lt[..., N:].type_as(p) - p[..., N:]))
        g_rb = (1 - (q_rb[..., :N].type_as(p) - p[..., :N])) * (1 - (q_rb[..., N:].type_as(p) - p[..., N:]))
        g_lb = (1 + (q_lb[..., :N].type_as(p) - p[..., :N])) * (1 - (q_lb[..., N:].type_as(p) - p[..., N:]))
        g_rt = (1 - (q_rt[..., :N].type_as(p) - p[..., :N])) * (1 + (q_rt[..., N:].type_as(p) - p[..., N:]))

        # resampling the features based on the modified coordinates.
        x_q_lt = self._get_x_q(x, q_lt, N)
        x_q_rb = self._get_x_q(x, q_rb, N)
        x_q_lb = self._get_x_q(x, q_lb, N)
        x_q_rt = self._get_x_q(x, q_rt, N)

        # bilinear
        x_offset = g_lt.unsqueeze(dim=1) * x_q_lt + \
                   g_rb.unsqueeze(dim=1) * x_q_rb + \
                   g_lb.unsqueeze(dim=1) * x_q_lb + \
                   g_rt.unsqueeze(dim=1) * x_q_rt

        x_offset = self._reshape_x_offset(x_offset, self.num_param)
        out = self.conv(x_offset)

        return out

    # generating the inital sampled shapes for the AKConv with different sizes.
    def _get_p_n(self, N, dtype):
        base_int = round(math.sqrt(self.num_param))
        row_number = self.num_param // base_int
        mod_number = self.num_param % base_int
        p_n_x, p_n_y = torch.meshgrid(
            torch.arange(0, row_number),
            torch.arange(0, base_int), indexing='xy')
        p_n_x = torch.flatten(p_n_x)
        p_n_y = torch.flatten(p_n_y)
        if mod_number > 0:
            mod_p_n_x, mod_p_n_y = torch.meshgrid(
                torch.arange(row_number, row_number + 1),
                torch.arange(0, mod_number), indexing='xy')

            mod_p_n_x = torch.flatten(mod_p_n_x)
            mod_p_n_y = torch.flatten(mod_p_n_y)
            p_n_x, p_n_y = torch.cat((p_n_x, mod_p_n_x)), torch.cat((p_n_y, mod_p_n_y))
        p_n = torch.cat([p_n_x, p_n_y], 0)
        p_n = p_n.view(1, 2 * N, 1, 1).type(dtype)
        return p_n

    # no zero-padding
    def _get_p_0(self, h, w, N, dtype):
        p_0_x, p_0_y = torch.meshgrid(
            torch.arange(0, h * self.stride, self.stride),
            torch.arange(0, w * self.stride, self.stride), indexing='xy')

        p_0_x = torch.flatten(p_0_x).view(1, 1, h, w).repeat(1, N, 1, 1)
        p_0_y = torch.flatten(p_0_y).view(1, 1, h, w).repeat(1, N, 1, 1)
        p_0 = torch.cat([p_0_x, p_0_y], 1).type(dtype)

        return p_0

    def _get_p(self, offset, dtype):
        N, h, w = offset.size(1) // 2, offset.size(2), offset.size(3)

        # (1, 2N, 1, 1)
        p_n = self._get_p_n(N, dtype)
        # (1, 2N, h, w)
        p_0 = self._get_p_0(h, w, N, dtype)
        p = p_0 + p_n + offset
        return p

    def _get_x_q(self, x, q, N):
        b, h, w, _ = q.size()
        padded_w = x.size(3)
        c = x.size(1)
        # (b, c, h*w)
        x = x.contiguous().view(b, c, -1)

        # (b, h, w, N)
        index = q[..., :N] * padded_w + q[..., N:]  # offset_x*w + offset_y
        # (b, c, h*w*N)
        index = index.contiguous().unsqueeze(dim=1).expand(-1, c, -1, -1, -1).contiguous().view(b, c, -1)
        index = index.clamp(min=0, max=x.shape[-1] - 1)
        x_offset = x.gather(dim=-1, index=index).contiguous().view(b, c, h, w, N)

        return x_offset

    #  Stacking resampled features in the row direction.
    @staticmethod
    def _reshape_x_offset(x_offset, num_param):
        b, c, h, w, n = x_offset.size()
        x_offset = rearrange(x_offset, 'b c h w n -> b c (h n) w')
        return x_offset


class GroupBatchnorm2d(nn.Module):
    def __init__(self, c_num: int,
                 group_num: int = 16,
                 eps: float = 1e-10
                 ):
        super(GroupBatchnorm2d, self).__init__()
        assert c_num >= group_num
        self.group_num = group_num
        self.weight = nn.Parameter(torch.randn(c_num, 1, 1))
        self.bias = nn.Parameter(torch.zeros(c_num, 1, 1))
        self.eps = eps

    def forward(self, x):
        N, C, H, W = x.size()
        x = x.view(N, self.group_num, -1)
        mean = x.mean(dim=2, keepdim=True)
        std = x.std(dim=2, keepdim=True)
        x = (x - mean) / (std + self.eps)
        x = x.view(N, C, H, W)
        return x * self.weight + self.bias


class SRU(nn.Module):
    def __init__(self,
                 oup_channels: int,
                 group_num: int = 16,
                 gate_treshold: float = 0.5,
                 torch_gn: bool = True
                 ):
        super().__init__()

        self.gn = nn.GroupNorm(num_channels=oup_channels, num_groups=group_num) if torch_gn else GroupBatchnorm2d(
            c_num=oup_channels, group_num=group_num)
        self.gate_treshold = gate_treshold
        self.sigomid = nn.Sigmoid()

    def forward(self, x):
        gn_x = self.gn(x)
        w_gamma = self.gn.weight / sum(self.gn.weight)
        w_gamma = w_gamma.view(1, -1, 1, 1)
        reweigts = self.sigomid(gn_x * w_gamma)
        # Gate
        w1 = torch.where(reweigts > self.gate_treshold, torch.ones_like(reweigts), reweigts)  # 大于门限值的设为1，否则保留原值
        w2 = torch.where(reweigts > self.gate_treshold, torch.zeros_like(reweigts), reweigts)  # 大于门限值的设为0，否则保留原值
        x_1 = w1 * x
        x_2 = w2 * x
        y = self.reconstruct(x_1, x_2)
        return y

    def reconstruct(self, x_1, x_2):
        x_11, x_12 = torch.split(x_1, x_1.size(1) // 2, dim=1)
        x_21, x_22 = torch.split(x_2, x_2.size(1) // 2, dim=1)
        return torch.cat([x_11 + x_22, x_12 + x_21], dim=1)


class CRU(nn.Module):
    '''
    alpha: 0<alpha<1
    '''

    def __init__(self,
                 op_channel: int,
                 alpha: float = 1 / 2,
                 squeeze_radio: int = 2,
                 group_size: int = 2,
                 group_kernel_size: int = 3,
                 ):
        super().__init__()
        self.up_channel = up_channel = int(alpha * op_channel)
        self.low_channel = low_channel = op_channel - up_channel
        self.squeeze1 = nn.Conv2d(up_channel, up_channel // squeeze_radio, kernel_size=1, bias=False)
        self.squeeze2 = nn.Conv2d(low_channel, low_channel // squeeze_radio, kernel_size=1, bias=False)
        # up
        self.GWC = nn.Conv2d(up_channel // squeeze_radio, op_channel, kernel_size=group_kernel_size, stride=1,
                             padding=group_kernel_size // 2, groups=group_size)
        self.PWC1 = nn.Conv2d(up_channel // squeeze_radio, op_channel, kernel_size=1, bias=False)
        # low
        self.PWC2 = nn.Conv2d(low_channel // squeeze_radio, op_channel - low_channel // squeeze_radio, kernel_size=1,
                              bias=False)
        self.advavg = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        # Split
        up, low = torch.split(x, [self.up_channel, self.low_channel], dim=1)
        up, low = self.squeeze1(up), self.squeeze2(low)
        # Transform
        Y1 = self.GWC(up) + self.PWC1(up)
        Y2 = torch.cat([self.PWC2(low), low], dim=1)
        # Fuse
        out = torch.cat([Y1, Y2], dim=1)
        out = F.softmax(self.advavg(out), dim=1) * out
        out1, out2 = torch.split(out, out.size(1) // 2, dim=1)
        return out1 + out2


# https://github.com/cheng-haha/ScConv/blob/main/
class ScConv(nn.Module):
    def __init__(self,
                 op_channel: int,
                 group_num: int = 4,
                 gate_treshold: float = 0.5,
                 alpha: float = 1 / 2,
                 squeeze_radio: int = 2,
                 group_size: int = 2,
                 group_kernel_size: int = 3,
                 ):
        super().__init__()
        self.SRU = SRU(op_channel,
                       group_num=group_num,
                       gate_treshold=gate_treshold)
        self.CRU = CRU(op_channel,
                       alpha=alpha,
                       squeeze_radio=squeeze_radio,
                       group_size=group_size,
                       group_kernel_size=group_kernel_size)

    def forward(self, x):
        x = self.SRU(x)
        x = self.CRU(x)
        return x


# https://gitcode.com/MCG-NKU/SCNet/blob/master/scnet.py
class SCConv(nn.Module):
    def __init__(self, c1, c2, stride=1, dilation=1, groups=1, pooling_r=4):
        super(SCConv, self).__init__()
        self.k2 = nn.Sequential(
                    nn.AvgPool2d(kernel_size=pooling_r, stride=pooling_r),
                    Conv(c1, c1, k=3, s=1, d=dilation, g=groups, act=False),
                    )
        self.k3 = Conv(c1, c1, k=3, s=1, d=dilation, g=groups, act=False)
        self.k4 = Conv(c1, c2, k=3, s=stride, d=dilation, g=groups, act=False)

    def forward(self, x):
        identity = x

        out = torch.sigmoid(torch.add(identity, F.interpolate(self.k2(x), identity.size()[2:]))) # sigmoid(identity + k2)
        out = torch.mul(self.k3(x), out) # k3 * sigmoid(identity + k2)
        out = self.k4(out) # k4

        return out



if __name__ == '__main__':
    x = torch.randn(1, 32, 16, 16)
    model = DySample(32,)
    print(model(x).shape)