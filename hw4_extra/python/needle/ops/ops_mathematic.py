"""Operator implementations."""

from numbers import Number
from typing import Optional, List, Tuple, Union

from ..autograd import NDArray
from ..autograd import Op, Tensor, Value, TensorOp
from ..autograd import TensorTuple, TensorTupleOp
import numpy

# NOTE: we will import numpy as the array_api
# as the backend for our computations, this line will change in later homeworks

from ..backend_selection import array_api, BACKEND
from .ops_tuple import *


class EWiseAdd(TensorOp):
    def compute(self, a: NDArray, b: NDArray):
        return a + b

    def gradient(self, out_grad: Tensor, node: Tensor):
        return out_grad, out_grad


def add(a, b):
    return EWiseAdd()(a, b)


class AddScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a: NDArray):
        return a + self.scalar

    def gradient(self, out_grad: Tensor, node: Tensor):
        return out_grad


def add_scalar(a, scalar):
    return AddScalar(scalar)(a)


class EWiseMul(TensorOp):
    def compute(self, a: NDArray, b: NDArray):
        return a * b

    def gradient(self, out_grad: Tensor, node: Tensor):
        lhs, rhs = node.inputs
        return out_grad * rhs, out_grad * lhs


def multiply(a, b):
    return EWiseMul()(a, b)


class MulScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a: NDArray):
        return a * self.scalar

    def gradient(self, out_grad: Tensor, node: Tensor):
        return (out_grad * self.scalar,)


def mul_scalar(a, scalar):
    return MulScalar(scalar)(a)


class EWisePow(TensorOp):
    """Op to element-wise raise a tensor to a power."""

    def compute(self, a: NDArray, b: NDArray) -> NDArray:
        ### BEGIN YOUR SOLUTION
        return a**b
        ### END YOUR SOLUTION

    def gradient(self, out_grad: Tensor, node: Tensor):
        ### BEGIN YOUR SOLUTION
        a, b = node.inputs
        return out_grad * (b * a ** (b - 1)), out_grad * log(a) * a**b
        ### END YOUR SOLUTION


def power(a, b):
    return EWisePow()(a, b)


class PowerScalar(TensorOp):
    """Op raise a tensor to an (integer) power."""

    def __init__(self, scalar: int):
        self.scalar = scalar

    def compute(self, a: NDArray) -> NDArray:
        ### BEGIN YOUR SOLUTION
        return a**self.scalar
        ### END YOUR SOLUTION

    def gradient(self, out_grad: Tensor, node: Tensor):
        ### BEGIN YOUR SOLUTION
        (a,) = node.inputs
        return (out_grad * (self.scalar * a ** (self.scalar - 1)),)
        ### END YOUR SOLUTION


def power_scalar(a, scalar):
    return PowerScalar(scalar)(a)


class EWiseDiv(TensorOp):
    """Op to element-wise divide two nodes."""

    def compute(self, a: NDArray, b: NDArray):
        ### BEGIN YOUR SOLUTION
        return a / b
        ### END YOUR SOLUTION

    def gradient(self, out_grad: Tensor, node: Tensor):
        ### BEGIN YOUR SOLUTION
        a, b = node.inputs
        return out_grad / b, -out_grad * a / (b**2)
        ### END YOUR SOLUTION


def divide(a, b):
    return EWiseDiv()(a, b)


class DivScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a: NDArray):
        ### BEGIN YOUR SOLUTION
        return a / self.scalar
        ### END YOUR SOLUTION

    def gradient(self, out_grad: Tensor, node: Tensor):
        ### BEGIN YOUR SOLUTION
        return out_grad / self.scalar
        ### END YOUR SOLUTION


def divide_scalar(a, scalar):
    return DivScalar(scalar)(a)


class Transpose(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, a: NDArray):
        ### BEGIN YOUR SOLUTION
        axis1, axis2 = self.axes or (a.ndim - 2, a.ndim - 1)
        return array_api.swapaxes(a, axis1, axis2)
        ### END YOUR SOLUTION

    def gradient(self, out_grad: Tensor, node: Tensor):
        ### BEGIN YOUR SOLUTION
        return out_grad.transpose(self.axes)
        ### END YOUR SOLUTION


def transpose(a, axes=None):
    return Transpose(axes)(a)


class Permute(TensorOp):
    def __init__(self, axes: tuple):
        self.axes = axes

    def compute(self, a: NDArray):
        ### BEGIN YOUR SOLUTION
        return array_api.permute(a, self.axes)
        ### END YOUR SOLUTION

    def gradient(self, out_grad: Tensor, node: Tensor):
        ### BEGIN YOUR SOLUTION
        inverse_axes = tuple(self.axes.index(i) for i in range(len(self.axes)))
        return permute(out_grad, inverse_axes)
        ### END YOUR SOLUTION


def permute(a, axes=None):
    return Permute(axes)(a)


class Reshape(TensorOp):
    def __init__(self, shape):
        self.shape = shape

    def compute(self, a: NDArray):
        ### BEGIN YOUR SOLUTION
        return array_api.reshape(a, self.shape)
        ### END YOUR SOLUTION

    def gradient(self, out_grad: Tensor, node: Tensor):
        ### BEGIN YOUR SOLUTION
        (a,) = node.inputs
        return out_grad.reshape(a.shape)
        ### END YOUR SOLUTION


def reshape(a, shape):
    return Reshape(shape)(a)


class BroadcastTo(TensorOp):
    def __init__(self, shape):
        self.shape = shape

    def compute(self, a: NDArray):
        ### BEGIN YOUR SOLUTION
        return array_api.broadcast_to(a, self.shape)
        ### END YOUR SOLUTION

    def gradient(self, out_grad: Tensor, node: Tensor):
        ### BEGIN YOUR SOLUTION
        (a,) = node.inputs
        input_shape = a.shape
        out_shape = out_grad.shape
        input_shape_ = (1,) * (len(out_shape) - len(input_shape)) + input_shape
        axes = tuple(i for i, (a, b) in enumerate(zip(input_shape_, out_shape)) if a == 1 and b > 1)
        grad = out_grad.sum(axes) if axes else out_grad
        return grad.reshape(input_shape)
        ### END YOUR SOLUTION


def broadcast_to(a, shape):
    return BroadcastTo(shape)(a)


class Summation(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, a: NDArray):
        ### BEGIN YOUR SOLUTION
        return array_api.sum(a, axis=self.axes)
        ### END YOUR SOLUTION

    def gradient(self, out_grad: Tensor, node: Tensor):
        ### BEGIN YOUR SOLUTION
        (a,) = node.inputs
        input_shape = a.shape
        axes = (self.axes,) if isinstance(self.axes, int) else self.axes
        axes = axes or tuple(range(len(input_shape)))
        # Normalize negative axes
        axes = tuple(ax if ax >= 0 else ax + len(input_shape) for ax in axes)
        shape = [size if i not in axes else 1 for i, size in enumerate(input_shape)]
        grad = out_grad.reshape(shape)
        return grad.broadcast_to(input_shape)
        ### END YOUR SOLUTION


def summation(a, axes=None):
    return Summation(axes)(a)


class MatMul(TensorOp):
    def compute(self, a: NDArray, b: NDArray):
        ### BEGIN YOUR SOLUTION
        return a @ b
        ### END YOUR SOLUTION

    def gradient(self, out_grad: Tensor, node: Tensor):
        ### BEGIN YOUR SOLUTION
        a, b = node.inputs
        grad_a = out_grad @ b.transpose()
        grad_b = a.transpose() @ out_grad
        if grad_a.shape != a.shape:
            axes = tuple(i for i, (ga, aa) in enumerate(zip(grad_a.shape, a.shape)) if ga != aa)
            grad_a = grad_a.sum(axes) if axes else grad_a
        if grad_b.shape != b.shape:
            axes = tuple(i for i, (gb, bb) in enumerate(zip(grad_b.shape, b.shape)) if gb != bb)
            grad_b = grad_b.sum(axes) if axes else grad_b
        return grad_a, grad_b
        ### END YOUR SOLUTION


def matmul(a, b):
    return MatMul()(a, b)


class Negate(TensorOp):
    def compute(self, a: NDArray):
        ### BEGIN YOUR SOLUTION
        return -a
        ### END YOUR SOLUTION

    def gradient(self, out_grad: Tensor, node: Tensor):
        ### BEGIN YOUR SOLUTION
        return -out_grad
        ### END YOUR SOLUTION


def negate(a):
    return Negate()(a)


class Log(TensorOp):
    def compute(self, a: NDArray):
        ### BEGIN YOUR SOLUTION
        return array_api.log(a)
        ### END YOUR SOLUTION

    def gradient(self, out_grad: Tensor, node: Tensor):
        ### BEGIN YOUR SOLUTION
        (a,) = node.inputs
        return out_grad / a
        ### END YOUR SOLUTION


def log(a):
    return Log()(a)


class Exp(TensorOp):
    def compute(self, a: NDArray):
        ### BEGIN YOUR SOLUTION
        return array_api.exp(a)
        ### END YOUR SOLUTION

    def gradient(self, out_grad: Tensor, node: Tensor):
        ### BEGIN YOUR SOLUTION
        (a,) = node.inputs
        return out_grad * exp(a)
        ### END YOUR SOLUTION


def exp(a):
    return Exp()(a)


class ReLU(TensorOp):
    def compute(self, a: NDArray):
        ### BEGIN YOUR SOLUTION
        return array_api.maximum(a, 0)
        ### END YOUR SOLUTION

    def gradient(self, out_grad: Tensor, node: Tensor):
        ### BEGIN YOUR SOLUTION
        (a,) = node.inputs
        return out_grad * (a.realize_cached_data() > 0)
        ### END YOUR SOLUTION


def relu(a):
    return ReLU()(a)


class Tanh(TensorOp):
    def compute(self, a: NDArray):
        ### BEGIN YOUR SOLUTION
        return array_api.tanh(a)
        ### END YOUR SOLUTION

    def gradient(self, out_grad: Tensor, node: Tensor):
        ### BEGIN YOUR SOLUTION
        (a,) = node.inputs
        return out_grad * (-tanh(a) ** 2 + 1)
        ### END YOUR SOLUTION


def tanh(a):
    return Tanh()(a)


class Stack(TensorOp):
    def __init__(self, axis: int):
        """
        Concatenates a sequence of arrays along a new dimension.
        Parameters:
        axis - dimension to concatenate along
        All arrays need to be of the same size.
        """
        self.axis = axis

    def compute(self, args: TensorTuple):
        ### BEGIN YOUR SOLUTION
        arg, *_ = args
        shape, device = arg.shape, arg.device
        n = len(args)
        out_shape = shape[: self.axis] + (n,) + shape[self.axis :]
        out = array_api.empty(out_shape, dtype=arg.dtype, device=arg.device)
        for i, a in enumerate(args):
            arr = array_api.reshape(a, shape[: self.axis] + (1,) + shape[self.axis :])
            slices = [slice(None)] * len(out_shape)
            slices[self.axis] = i
            out[tuple(slices)] = arr
        return out
        ### END YOUR SOLUTION

    def gradient(self, out_grad: Tensor, node: Tensor):
        ### BEGIN YOUR SOLUTION
        return split(out_grad, self.axis)
        ### END YOUR SOLUTION


def stack(args, axis):
    return Stack(axis)(make_tuple(*args))


class Split(TensorTupleOp):
    def __init__(self, axis: int):
        """
        Splits a tensor along an axis into a tuple of tensors.
        (The "inverse" of Stack)
        Parameters:
        axis - dimension to split
        """
        self.axis = axis

    def compute(self, A: Tensor):
        ### BEGIN YOUR SOLUTION
        n = A.shape[self.axis]
        split_shape = A.shape[: self.axis] + A.shape[self.axis + 1 :]
        results = []
        for i in range(n):
            slices = [slice(None)] * len(A.shape)
            slices[self.axis] = i
            results.append(A[tuple(slices)].reshape(split_shape))
        return tuple(results)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return stack(out_grad, self.axis)
        ### END YOUR SOLUTION


def split(a, axis):
    return Split(axis)(a)


class Flip(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, a: NDArray):
        ### BEGIN YOUR SOLUTION
        return array_api.flip(a, self.axes)
        ### END YOUR SOLUTION

    def gradient(self, out_grad: Tensor, node: Tensor):
        ### BEGIN YOUR SOLUTION
        return flip(out_grad, self.axes)
        ### END YOUR SOLUTION


def flip(a, axes):
    return Flip(axes)(a)


class Dilate(TensorOp):
    def __init__(self, axes: tuple, dilation: int):
        self.axes = axes
        self.dilation = dilation

    def compute(self, a: NDArray):
        ### BEGIN YOUR SOLUTION
        if self.dilation == 0:
            return a
        shape = list(a.shape)
        for ax in self.axes:
            shape[ax] = shape[ax] * (self.dilation + 1)
        out = array_api.full(tuple(shape), 0, dtype=a.dtype, device=a.device)
        slices = [slice(None)] * len(shape)
        for ax in self.axes:
            slices[ax] = slice(None, None, self.dilation + 1)
        out[tuple(slices)] = a
        return out
        ### END YOUR SOLUTION

    def gradient(self, out_grad: Tensor, node: Tensor):
        ### BEGIN YOUR SOLUTION
        return undilate(out_grad, self.axes, self.dilation)
        ### END YOUR SOLUTION


def dilate(a, axes, dilation):
    return Dilate(axes, dilation)(a)


class UnDilate(TensorOp):
    def __init__(self, axes: tuple, dilation: int):
        self.axes = axes
        self.dilation = dilation

    def compute(self, a: NDArray):
        ### BEGIN YOUR SOLUTION
        if self.dilation == 0:
            return a
        shape = list(a.shape)
        for ax in self.axes:
            shape[ax] = shape[ax] // (self.dilation + 1)
        out = array_api.empty(tuple(shape), dtype=a.dtype, device=a.device)
        slices = [slice(None)] * len(a.shape)
        for ax in self.axes:
            slices[ax] = slice(None, None, self.dilation + 1)
        out = a[tuple(slices)]
        return out
        ### END YOUR SOLUTION

    def gradient(self, out_grad: Tensor, node: Tensor):
        ### BEGIN YOUR SOLUTION
        return dilate(out_grad, self.axes, self.dilation)
        ### END YOUR SOLUTION


def undilate(a, axes, dilation):
    return UnDilate(axes, dilation)(a)


class Conv(TensorOp):
    def __init__(self, stride: Optional[int] = 1, padding: Optional[int] = 0):
        self.stride = stride
        self.padding = padding

    def compute(self, A: NDArray, B: NDArray):
        ### BEGIN YOUR SOLUTION
        # A: (N, H, W, C_in), B: (KH, KW, C_in, C_out)

        # padding spatial dims
        if self.padding > 0:
            pad_width = ((0, 0), (self.padding, self.padding), (self.padding, self.padding), (0, 0))
            A = array_api.pad(A, pad_width)
        N, H, W, C_in = A.shape
        KH, KW, _, C_out = B.shape
        Ns, Hs, Ws, Cs = A.strides

        OH = (H - KH) // self.stride + 1
        OW = (W - KW) // self.stride + 1

        # im2col: as_strided
        shape = (N, OH, OW, KH, KW, C_in)
        strides = (Ns, Hs * self.stride, Ws * self.stride, Hs, Ws, Cs)
        cols = A.as_strided(shape, strides)
        cols = cols.reshape((N * OH * OW, KH * KW * C_in))  # (N*OH*OW, KH*KW*C_in)
        B_col = B.reshape((KH * KW * C_in, C_out))  # (KH*KW*C_in, C_out)
        out = cols @ B_col  # (N*OH*OW, C_out)
        out = out.reshape((N, OH, OW, C_out))
        return out
        ### END YOUR SOLUTION

    def gradient(self, out_grad: Tensor, node: Tensor):
        ### BEGIN YOUR SOLUTION
        # A: (N, H, W, C_in), B: (KH, KW, C_in, C_out)
        A, B = node.inputs
        KH, *_ = B.shape

        # grad wrt input
        B_flip = flip(B, (0, 1))  # (KH, KW, C_in, C_out)
        B_flip = transpose(B_flip, (2, 3))  # (KW, KH, C_out, C_in)
        out_grad = dilate(out_grad, (1, 2), self.stride - 1) # (N, OH, OW, C_out)
        padding = KH - 1 - self.padding
        grad_A = conv(out_grad, B_flip, padding=padding) # (N, H, W, C_in)

        # grad wrt weight
        out_grad = permute(out_grad, (1, 2, 0, 3)) # (OH, OW, N, C_out)
        A = permute(A, (3, 1, 2, 0)) # (C_in, H, W, N)
        grad_B = conv(A, out_grad, padding=self.padding) # (C_in, KH, KW, C_out)
        grad_B = permute(grad_B, (1, 2, 0, 3)) # (KH, KW, C_in, C_out)
        return grad_A, grad_B
        ### END YOUR SOLUTION


def conv(a, b, stride=1, padding=1):
    return Conv(stride, padding)(a, b)
