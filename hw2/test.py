import sys
sys.path.append("./python")
import numpy as np
import needle as ndl
import needle.nn as nn

sys.path.append("./apps")
from mlp_resnet import *


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

def mlp_resnet_forward(dim, hidden_dim, num_blocks, num_classes, norm, drop_prob):
    np.random.seed(4)
    input_tensor = ndl.Tensor(np.random.randn(2, dim), dtype=np.float32)
    output_tensor = MLPResNet(dim, hidden_dim, num_blocks, num_classes, norm, drop_prob)(input_tensor)
    return output_tensor.numpy()


def test_mlp_resnet_forward_2():
    np.testing.assert_allclose(
        mlp_resnet_forward(15, 25, 5, 14, nn.BatchNorm1d, 0.0),
        np.array([[
            0.92448235, -2.745743, -1.5077105, 1.130784, -1.2078242,
            -0.09833566, -0.69301605, 2.8945382, 1.259397, 0.13866742,
            -2.963875, -4.8566914, 1.7062538, -4.846424
        ],
        [
            0.6653336, -2.4708004, 2.0572243, -1.0791507, 4.3489094,
            3.1086435, 0.0304327, -1.9227124, -1.416201, -7.2151937,
            -1.4858506, 7.1039696, -2.1589825, -0.7593413
        ]],
        dtype=np.float32),
        rtol=1e-5,
        atol=1e-5)

# print(batchnorm_forward(4, 4))
test_mlp_resnet_forward_2()