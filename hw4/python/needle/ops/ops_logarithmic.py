from typing import Optional
from ..autograd import NDArray
from ..autograd import Op, Tensor, Value, TensorOp
from ..autograd import TensorTuple, TensorTupleOp

from .ops_mathematic import *

from ..backend_selection import array_api, BACKEND


class LogSoftmax(TensorOp):
    def compute(self, Z: NDArray):
        ### BEGIN YOUR SOLUTION
        max_Z = array_api.max(Z, axis=1, keepdims=True).broadcast_to(Z.shape)
        sum_exp = array_api.sum(array_api.exp(Z - max_Z), axis=-1, keepdims=True).broadcast_to(Z.shape)
        lse = array_api.log(sum_exp) + max_Z
        return Z - lse
        ### END YOUR SOLUTION

    def gradient(self, out_grad: Tensor, node: Tensor):
        ### BEGIN YOUR SOLUTION
        (Z,) = node.inputs
        batch, _ = Z.shape
        softmax = exp(logsoftmax(Z))
        sum_grad = out_grad.sum(axes=-1).reshape((batch, 1)).broadcast_to(softmax.shape)
        return out_grad - softmax * sum_grad
        ### END YOUR SOLUTION


def logsoftmax(a):
    return LogSoftmax()(a)


class LogSumExp(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, Z: NDArray):
        ### BEGIN YOUR SOLUTION
        max_Z = array_api.max(Z, axis=self.axes, keepdims=True)
        sum_exp = array_api.sum(array_api.exp(Z - max_Z.broadcast_to(Z.shape)), axis=self.axes, keepdims=True)
        out = array_api.log(sum_exp) + max_Z
        out = array_api.squeeze(out, axis=self.axes)
        return out
        ### END YOUR SOLUTION

    def gradient(self, out_grad: Tensor, node: Tensor):
        ### BEGIN YOUR SOLUTION
        (Z,) = node.inputs
        lse = logsumexp(Z, axes=self.axes)
        if self.axes is not None:
            shape = list(Z.shape)
            axes = [self.axes] if isinstance(self.axes, int) else list(self.axes)
            for ax in axes:
                shape[ax] = 1
            # reshape the logsumexp output and the gradient to match the input shape
            lse = lse.reshape(shape).broadcast_to(Z.shape)
            out_grad = out_grad.reshape(shape).broadcast_to(Z.shape)
        softmax = exp(Z - lse)  # softmax(z) = exp(z - lse)
        return out_grad * softmax
        ### END YOUR SOLUTION


def logsumexp(a, axes=None):
    return LogSumExp(axes=axes)(a)
