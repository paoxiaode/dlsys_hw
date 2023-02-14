import sys
sys.path.append("./python")
import numpy as np
import needle as ndl
import needle.nn as nn

sys.path.append("./apps")

def get_tensor(*shape, entropy=1):
    np.random.seed(np.prod(shape) * len(shape) * entropy)
    return ndl.Tensor(np.random.randint(0, 100, size=shape) / 20, dtype="float32")

def batchnorm_forward(*shape, affine=False):
    x = get_tensor(*shape)
    bn = ndl.nn.BatchNorm1d(shape[1])
    if affine:
        bn.weight.data = get_tensor(shape[1], entropy=42)
        bn.bias.data = get_tensor(shape[1], entropy=1337)
    return bn(x).cached_data

print(batchnorm_forward(4, 4))