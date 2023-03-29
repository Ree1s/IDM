import math
import numpy as np
import random
import torch
from torch import nn
from torch.nn import functional as F
from ..ops.fused_act.fused_act import fused_leaky_relu, FusedLeakyReLU
from ..ops.upfirdn2d.upfirdn2d import upfirdn2d
def make_resample_kernel(k):
    """Make resampling kernel for UpFirDn.

    Args:
        k (list[int]): A list indicating the 1D resample kernel magnitude.

    Returns:
        Tensor: 2D resampled kernel.
    """
    k = torch.tensor(k, dtype=torch.float32)
    if k.ndim == 1:
        k = k[None, :] * k[:, None]  # to 2D kernel, outer product
    # normalize
    k /= k.sum()
    return k
class UpFirDnSmooth(nn.Module):
    """Upsample, FIR filter, and downsample (smooth version).

    Args:
        resample_kernel (list[int]): A list indicating the 1D resample kernel
            magnitude.
        upsample_factor (int): Upsampling scale factor. Default: 1.
        downsample_factor (int): Downsampling scale factor. Default: 1.
        kernel_size (int): Kernel size: Deafult: 1.
    """

    def __init__(self, resample_kernel, upsample_factor=1, downsample_factor=1, kernel_size=1):
        super(UpFirDnSmooth, self).__init__()
        self.upsample_factor = upsample_factor
        self.downsample_factor = downsample_factor
        self.kernel = make_resample_kernel(resample_kernel)
        if upsample_factor > 1:
            self.kernel = self.kernel * (upsample_factor**2)

        if upsample_factor > 1:
            pad = (self.kernel.shape[0] - upsample_factor) - (kernel_size - 1)
            self.pad = ((pad + 1) // 2 + upsample_factor - 1, pad // 2 + 1)
        elif downsample_factor > 1:
            pad = (self.kernel.shape[0] - downsample_factor) + (kernel_size - 1)
            # self.pad = ((pad + 1) // 2, pad // 2)
            self.pad = ((pad - 1) // 2, pad   // 2)
        else:
            raise NotImplementedError

    def forward(self, x):
        out = upfirdn2d(x, self.kernel.type_as(x), up=1, down=1, pad=self.pad)
        return out

    def __repr__(self):
        return (f'{self.__class__.__name__}(upsample_factor={self.upsample_factor}'
                f', downsample_factor={self.downsample_factor})')
class ScaledLeakyReLU(nn.Module):
    """Scaled LeakyReLU.

    Args:
        negative_slope (float): Negative slope. Default: 0.2.
    """

    def __init__(self, negative_slope=0.2):
        super(ScaledLeakyReLU, self).__init__()
        self.negative_slope = negative_slope

    def forward(self, x):
        out = F.leaky_relu(x, negative_slope=self.negative_slope)
        return out * math.sqrt(2)
class EqualConv2d(nn.Module):
    """Equalized Linear as StyleGAN2.

    Args:
        in_channels (int): Channel number of the input.
        out_channels (int): Channel number of the output.
        kernel_size (int): Size of the convolving kernel.
        stride (int): Stride of the convolution. Default: 1
        padding (int): Zero-padding added to both sides of the input.
            Default: 0.
        bias (bool): If ``True``, adds a learnable bias to the output.
            Default: ``True``.
        bias_init_val (float): Bias initialized value. Default: 0.
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True, bias_init_val=0):
        super(EqualConv2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.scale = 1 / math.sqrt(in_channels * kernel_size**2)

        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size, kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channels).fill_(bias_init_val))
        else:
            self.register_parameter('bias', None)

    def forward(self, x):
        out = F.conv2d(
            x,
            self.weight * self.scale,
            bias=self.bias,
            stride=self.stride,
            padding=self.padding,
        )

        return out

    def __repr__(self):
        return (f'{self.__class__.__name__}(in_channels={self.in_channels}, '
                f'out_channels={self.out_channels}, '
                f'kernel_size={self.kernel_size},'
                f' stride={self.stride}, padding={self.padding}, '
                f'bias={self.bias is not None})')
class StyleLayer(nn.Sequential):
    """Conv Layer used in StyleGAN2 Discriminator.

    Args:
        in_channels (int): Channel number of the input.
        out_channels (int): Channel number of the output.
        kernel_size (int): Kernel size.
        downsample (bool): Whether downsample by a factor of 2.
            Default: False.
        resample_kernel (list[int]): A list indicating the 1D resample
            kernel magnitude. A cross production will be applied to
            extent 1D resample kenrel to 2D resample kernel.
            Default: (1, 3, 3, 1).
        bias (bool): Whether with bias. Default: True.
        activate (bool): Whether use activateion. Default: True.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 downsample=False,
                 resample_kernel=(1, 3, 1), # (1, 3, 3, 1),
                 bias=True,
                 activate=True):
        layers = []
        # downsample
        if downsample:
            layers.append(
                UpFirDnSmooth(resample_kernel, upsample_factor=1, downsample_factor=2, kernel_size=kernel_size))
            stride = 2
            self.padding = 1
        else:
            stride = 1
            self.padding = kernel_size // 2
        # conv
        layers.append(
            EqualConv2d(
                in_channels, out_channels, kernel_size, stride=stride, padding=self.padding, bias=bias
                and not activate))
        # activation
        if activate:
            if bias:
                layers.append(FusedLeakyReLU(out_channels))
            else:
                layers.append(ScaledLeakyReLU(0.2))

        super(StyleLayer, self).__init__(*layers)
class EqualLinear(nn.Module):
    """Equalized Linear as StyleGAN2.

    Args:
        in_channels (int): Size of each sample.
        out_channels (int): Size of each output sample.
        bias (bool): If set to ``False``, the layer will not learn an additive
            bias. Default: ``True``.
        bias_init_val (float): Bias initialized value. Default: 0.
        lr_mul (float): Learning rate multiplier. Default: 1.
        activation (None | str): The activation after ``linear`` operation.
            Supported: 'fused_lrelu', None. Default: None.
    """

    def __init__(self, in_channels, out_channels, bias=True, bias_init_val=0, lr_mul=1, activation=None):
        super(EqualLinear, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.lr_mul = lr_mul
        self.activation = activation
        if self.activation not in ['fused_lrelu', None]:
            raise ValueError(f'Wrong activation value in EqualLinear: {activation}'
                             "Supported ones are: ['fused_lrelu', None].")
        self.scale = (1 / math.sqrt(in_channels)) * lr_mul

        self.weight = nn.Parameter(torch.randn(out_channels, in_channels).div_(lr_mul))
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channels).fill_(bias_init_val))
        else:
            self.register_parameter('bias', None)

    def forward(self, x):
        if self.bias is None:
            bias = None
        else:
            bias = self.bias * self.lr_mul
        if self.activation == 'fused_lrelu':
            out = F.linear(x, self.weight * self.scale)
            out = fused_leaky_relu(out, bias)
        else:
            out = F.linear(x, self.weight * self.scale, bias=bias)
        return out

    def __repr__(self):
        return (f'{self.__class__.__name__}(in_channels={self.in_channels}, '
                f'out_channels={self.out_channels}, bias={self.bias is not None})')
class ModulatedLayer(nn.Module):
    """Modulated Conv2d used in StyleGAN2.

    There is no bias in ModulatedLayer.

    Args:
        in_channels (int): Channel number of the input.
        out_channels (int): Channel number of the output.
        kernel_size (int): Size of the convolving kernel.
        num_style_feat (int): Channel number of style features.
        demodulate (bool): Whether to demodulate in the conv layer.
            Default: True.
        sample_mode (str | None): Indicating 'upsample', 'downsample' or None.
            Default: None.
        resample_kernel (list[int]): A list indicating the 1D resample kernel
            magnitude. Default: (1, 3, 3, 1).
        eps (float): A value added to the denominator for numerical stability.
            Default: 1e-8.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 num_style_feat,
                 demodulate=True,
                 sample_mode=None,
                 resample_kernel=(1, 3, 3, 1),
                 eps=1e-8):
        super(ModulatedLayer, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.demodulate = demodulate
        self.sample_mode = sample_mode
        self.eps = eps

        if self.sample_mode == 'upsample':
            self.smooth = UpFirDnSmooth(
                resample_kernel, upsample_factor=2, downsample_factor=1, kernel_size=kernel_size)
        elif self.sample_mode == 'downsample':
            self.smooth = UpFirDnSmooth(
                resample_kernel, upsample_factor=1, downsample_factor=2, kernel_size=kernel_size)
        elif self.sample_mode is None:
            pass
        else:
            raise ValueError(f'Wrong sample mode {self.sample_mode}, '
                             "supported ones are ['upsample', 'downsample', None].")

        self.scale = 1 / math.sqrt(in_channels * kernel_size**2)
        # modulation inside each modulated conv
        self.modulation = EqualLinear(
            num_style_feat, in_channels, bias=True, bias_init_val=1, lr_mul=1, activation=None)

        self.weight = nn.Parameter(torch.randn(1, out_channels, in_channels, kernel_size, kernel_size))
        self.padding = kernel_size // 2

    def forward(self, x, style):
        """Forward function.

        Args:
            x (Tensor): Tensor with shape (b, c, h, w).
            style (Tensor): Tensor with shape (b, num_style_feat).

        Returns:
            Tensor: Modulated tensor after convolution.
        """
        b, c, h, w = x.shape  # c = c_in
        # weight modulation
        style = self.modulation(style).view(b, 1, c, 1, 1)
        # self.weight: (1, c_out, c_in, k, k); style: (b, 1, c, 1, 1)
        weight = self.scale * self.weight * style  # (b, c_out, c_in, k, k)

        if self.demodulate:
            demod = torch.rsqrt(weight.pow(2).sum([2, 3, 4]) + self.eps)
            weight = weight * demod.view(b, self.out_channels, 1, 1, 1)

        weight = weight.view(b * self.out_channels, c, self.kernel_size, self.kernel_size)

        if self.sample_mode == 'upsample':
            x = x.view(1, b * c, h, w)
            weight = weight.view(b, self.out_channels, c, self.kernel_size, self.kernel_size)
            weight = weight.transpose(1, 2).reshape(b * c, self.out_channels, self.kernel_size, self.kernel_size)
            out = F.conv_transpose2d(x, weight, padding=0, stride=2, groups=b)
            out = out.view(b, self.out_channels, *out.shape[2:4])
            out = self.smooth(out)
        elif self.sample_mode == 'downsample':
            x = self.smooth(x)
            x = x.view(1, b * c, *x.shape[2:4])
            out = F.conv2d(x, weight, padding=0, stride=2, groups=b)
            out = out.view(b, self.out_channels, *out.shape[2:4])
        else:
            x = x.view(1, b * c, h, w)
            # weight: (b*c_out, c_in, k, k), groups=b
            out = F.conv2d(x, weight, padding=self.padding, groups=b)
            out = out.view(b, self.out_channels, *out.shape[2:4])

        return out

    def __repr__(self):
        return (f'{self.__class__.__name__}(in_channels={self.in_channels}, '
                f'out_channels={self.out_channels}, '
                f'kernel_size={self.kernel_size}, '
                f'demodulate={self.demodulate}, sample_mode={self.sample_mode})')

class Norm2Scale(nn.Module):
    def forward(self, scale1, scale2):
        scales_norm = scale1**2 + scale2**2 + 1e-8
        return scale1 * torch.rsqrt(scales_norm), scale2 * torch.rsqrt(scales_norm)   
             
class StyleLayer_norm_scale_shift(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 num_style_feat,
                 demodulate=True,
                 sample_mode=None,
                 resample_kernel=(1, 3, 3, 1)):
        super(StyleLayer_norm_scale_shift, self).__init__()
        self.modulated_conv = ModulatedLayer(
            in_channels,
            out_channels,
            kernel_size,
            num_style_feat,
            demodulate=demodulate,
            sample_mode=sample_mode,
            resample_kernel=resample_kernel)
        self.weight = nn.Parameter(torch.zeros(1))  # for noise injection
        self.activate = FusedLeakyReLU(out_channels)
        self.norm = Norm2Scale()

    def forward(self, x, noise=None, scale1=None, scale2=None, shift=None):
        # modulate
        # out = self.modulated_conv(x, style)
        # noise injection
        # if noise is None:
            # b, _, h, w = out.shape
            # noise = out.new_empty(b, 1, h, w).normal_()
        # out = out + self.weight * noise

        scale1, scale2 = self.norm(scale1, scale2)

        out = x * scale1.view(-1, x.size(1), 1, 1) + shift * scale2.view(-1, x.size(1), 1, 1)

        # activation (with bias)
        out = self.activate(out)
        return out