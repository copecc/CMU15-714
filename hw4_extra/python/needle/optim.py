"""Optimization module"""

import needle as ndl
import numpy as np


class Optimizer:
    def __init__(self, params):
        self.params = params

    def step(self):
        raise NotImplementedError()

    def reset_grad(self):
        for p in self.params:
            p.grad = None


class SGD(Optimizer):
    def __init__(self, params, lr=0.01, momentum=0.0, weight_decay=0.0):
        super().__init__(params)
        self.lr = lr
        self.momentum = momentum
        self.u = {}
        self.weight_decay = weight_decay

    def step(self):
        ### BEGIN YOUR SOLUTION
        for p in self.params:
            if p.grad is None:
                continue
            grad = p.grad.detach()
            if self.weight_decay != 0.0:
                grad = grad + self.weight_decay * p.data
            u = self.u.get(p, ndl.init.zeros(*p.shape, device=p.device, dtype=p.dtype))
            u = self.momentum * u + (1 - self.momentum) * grad
            p.data = p.data - self.lr * u
            self.u[p] = u
        ### END YOUR SOLUTION

    def clip_grad_norm(self, max_norm=0.25):
        """
        Clips gradient norm of parameters.
        """
        ### BEGIN YOUR SOLUTION
        total_norm = 0.0
        for p in self.params:
            if p.grad is None:
                continue
            param_norm = (p.grad**2).sum().numpy() ** 0.5
            total_norm += param_norm**2
        total_norm = total_norm**0.5
        clip_coef = max_norm / (total_norm + 1e-6)
        if clip_coef < 1:
            for p in self.params:
                if p.grad is None:
                    continue
                p.grad = p.grad * clip_coef
        ### END YOUR SOLUTION


class Adam(Optimizer):
    def __init__(self, params, lr=0.01, beta1=0.9, beta2=0.999, eps=1e-8, weight_decay=0.0):
        super().__init__(params)
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.weight_decay = weight_decay
        self.t = 0

        self.m = {}
        self.v = {}

    def step(self):
        ### BEGIN YOUR SOLUTION
        self.t += 1
        for p in self.params:
            if p.grad is None:
                continue
            grad = p.grad.detach()
            if self.weight_decay != 0.0:
                grad = grad + self.weight_decay * p.data
            m = self.m.get(p, ndl.init.zeros(*p.shape, device=p.device, dtype=p.dtype))
            v = self.v.get(p, ndl.init.zeros(*p.shape, device=p.device, dtype=p.dtype))
            m = self.beta1 * m + (1 - self.beta1) * grad
            v = self.beta2 * v + (1 - self.beta2) * grad**2
            self.m[p] = m
            self.v[p] = v
            m_hat = m / (1 - self.beta1**self.t)
            v_hat = v / (1 - self.beta2**self.t)
            p.data = p.data - self.lr * m_hat / (v_hat**0.5 + self.eps)
        ### END YOUR SOLUTION
