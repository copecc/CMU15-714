"""The module."""

from typing import List, Callable, Any
from needle.autograd import Tensor
from needle import ops
import needle.init as init
import numpy as np
from .nn_basic import Parameter, Module


class Conv(Module):
    """
    Multi-channel 2D convolutional layer
    IMPORTANT: Accepts inputs in NCHW format, outputs also in NCHW format
    Only supports padding=same
    No grouped convolution or dilation
    Only supports square kernels
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, bias=True, device=None, dtype="float32"):
        super().__init__()
        if isinstance(kernel_size, tuple):
            kernel_size = kernel_size[0]
        if isinstance(stride, tuple):
            stride = stride[0]
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride

        ### BEGIN YOUR SOLUTION
        self.padding = kernel_size // 2

        shape = (kernel_size, kernel_size, in_channels, out_channels)
        weight_data = init.kaiming_uniform(
            in_channels, out_channels, shape, device=device, dtype=dtype, requires_grad=True
        )
        self.weight = Parameter(weight_data)

        if bias:
            bound = 1 / np.sqrt(in_channels * kernel_size**2)
            bias_data = init.rand(out_channels, low=-bound, high=bound, device=device, dtype=dtype, requires_grad=True)
            self.bias = Parameter(bias_data)
        ### END YOUR SOLUTION

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        x = ops.permute(x, (0, 2, 3, 1))  # NCHW -> NHWC
        x = ops.conv(x, self.weight, stride=self.stride, padding=self.padding)
        if self.bias is not None:
            bias = self.bias.broadcast_to(x.shape)
            x = x + bias
        x = ops.permute(x, (0, 3, 1, 2))  # NHWC -> NCHW
        return x
        ### END YOUR SOLUTION
