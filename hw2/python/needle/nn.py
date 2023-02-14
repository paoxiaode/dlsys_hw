"""The module.
"""
from typing import List, Callable, Any
from needle.autograd import Tensor
from needle import ops
import needle.init as init
import numpy as np


class Parameter(Tensor):
    """A special kind of tensor that represents parameters."""


def _unpack_params(value: object) -> List[Tensor]:
    if isinstance(value, Parameter):
        return [value]
    elif isinstance(value, Module):
        return value.parameters()
    elif isinstance(value, dict):
        params = []
        for k, v in value.items():
            params += _unpack_params(v)
        return params
    elif isinstance(value, (list, tuple)):
        params = []
        for v in value:
            params += _unpack_params(v)
        return params
    else:
        return []


def _child_modules(value: object) -> List["Module"]:
    if isinstance(value, Module):
        modules = [value]
        modules.extend(_child_modules(value.__dict__))
        return modules
    if isinstance(value, dict):
        modules = []
        for k, v in value.items():
            modules += _child_modules(v)
        return modules
    elif isinstance(value, (list, tuple)):
        modules = []
        for v in value:
            modules += _child_modules(v)
        return modules
    else:
        return []


class Module:
    def __init__(self):
        self.training = True

    def parameters(self) -> List[Tensor]:
        """Return the list of parameters in the module."""
        return _unpack_params(self.__dict__)

    def _children(self) -> List["Module"]:
        return _child_modules(self.__dict__)

    def eval(self):
        self.training = False
        for m in self._children():
            m.training = False

    def train(self):
        self.training = True
        for m in self._children():
            m.training = True

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


class Identity(Module):
    def forward(self, x):
        return x


class Linear(Module):
    def __init__(
        self, in_features, out_features, bias=True, device=None, dtype="float32"
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        # BEGIN YOUR SOLUTION
        self.weight = Parameter(
            init.kaiming_uniform(in_features, out_features), device=device, dtype=dtype
        )
        self.bias = Parameter(
            init.kaiming_uniform(out_features, 1).transpose()
            if bias
            else init.zeros(1, out_features)
        )
        # END YOUR SOLUTION

    def forward(self, X: Tensor) -> Tensor:
        # BEGIN YOUR SOLUTION
        return X @ self.weight + self.bias.broadcast_to((X.shape[0], self.out_features))
        # END YOUR SOLUTION


class Flatten(Module):
    def forward(self, X):
        # BEGIN YOUR SOLUTION
        return X.reshape((X.shape[0], -1))
        # END YOUR SOLUTION


class ReLU(Module):
    def forward(self, x: Tensor) -> Tensor:
        # BEGIN YOUR SOLUTION
        return ops.relu(x)
        # END YOUR SOLUTION


class Sequential(Module):
    def __init__(self, *modules):
        super().__init__()
        self.modules = modules

    def forward(self, x: Tensor) -> Tensor:
        # BEGIN YOUR SOLUTION
        for module in self.modules:
            x = module(x)
        return x
        # END YOUR SOLUTION


class SoftmaxLoss(Module):
    def forward(self, logits: Tensor, y: Tensor):
        # BEGIN YOUR SOLUTION
        log_exp_sum = ops.logsumexp(logits, axes=(1,)).sum()
        z_y_sum = (logits * init.one_hot(logits.shape[1], y)).sum()
        return (log_exp_sum - z_y_sum) / logits.shape[0]
        # END YOUR SOLUTION


class BatchNorm1d(Module):
    def __init__(self, dim, eps=1e-5, momentum=0.1, device=None, dtype="float32"):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.momentum = momentum
        # BEGIN YOUR SOLUTION
        self.weight = Parameter(
            init.ones(dim, device=device, dtype=dtype, requires_grad=True)
        )
        self.bias = Parameter(
            init.zeros(dim, device=device, dtype=dtype, requires_grad=True)
        )
        self.running_mean = init.zeros(dim, device=device, dtype=dtype)
        self.running_var = init.ones(dim, device=device, dtype=dtype)
        # END YOUR SOLUTION

    def forward(self, x: Tensor) -> Tensor:
        # BEGIN YOUR SOLUTION
        if self.training:
            n, k = x.shape
            mean = x.sum(axes=0) / n
            self.running_mean.data = (
                (1 - self.momentum) * self.running_mean + self.momentum * mean
            ).data

            mean = mean.reshape((1, self.dim)).broadcast_to(x.shape)
            var = ((x - mean) ** 2).sum(axes=0) / n
            self.running_var.data = (
                (1 - self.momentum) * self.running_var + self.momentum * var
            ).data
            var = var.reshape((1, self.dim)).broadcast_to(x.shape)
        else:
            mean = self.running_mean.reshape((1, self.dim)).broadcast_to(x.shape)
            var = self.running_var.reshape((1, self.dim)).broadcast_to(x.shape)

        weight = self.weight.reshape((1, self.dim)).broadcast_to(x.shape)
        bias = self.bias.reshape((1, self.dim)).broadcast_to(x.shape)
        return weight * (x - mean) / ((var + self.eps) ** 0.5) + bias
        # END YOUR SOLUTION


class LayerNorm1d(Module):
    def __init__(self, dim, eps=1e-5, device=None, dtype="float32"):
        super().__init__()
        self.dim = dim
        self.eps = eps
        # BEGIN YOUR SOLUTION
        self.weight = Parameter(init.ones(dim, device=device, dtype=dtype))
        self.bias = Parameter(init.zeros(dim, device=device, dtype=dtype))
        # END YOUR SOLUTION

    def forward(self, x: Tensor) -> Tensor:
        # BEGIN YOUR SOLUTION
        n, k = x.shape
        # Layer mean
        mean = x.sum(axes=1) / k
        mean = mean.reshape((n, 1)).broadcast_to(x.shape)
        # Layer variance
        var = ((x - mean) ** 2).sum(axes=1) / k
        var = var.reshape((n, 1)).broadcast_to(x.shape)

        weight = self.weight.reshape((1, self.dim)).broadcast_to(x.shape)
        bias = self.bias.reshape((1, self.dim)).broadcast_to(x.shape)

        y = weight * (x - mean) / ((var + self.eps) ** 0.5) + bias
        return y
        # END YOUR SOLUTION


class Dropout(Module):
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    def forward(self, x: Tensor) -> Tensor:
        # BEGIN YOUR SOLUTION
        if self.training:
            mask = init.randb(*x.shape, p=self.p)
            return x * mask / (1 - self.p)
        else:
            return x
        # END YOUR SOLUTION


class Residual(Module):
    def __init__(self, fn: Module):
        super().__init__()
        self.fn = fn

    def forward(self, x: Tensor) -> Tensor:
        # BEGIN YOUR SOLUTION
        res = self.fn(x)
        return x + res
        # END YOUR SOLUTION
