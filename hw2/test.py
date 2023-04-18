import sys
sys.path.append("./python")
import numpy as np
import needle as ndl
import needle.nn as nn

sys.path.append("./apps")

def get_tensor(*shape, entropy=1):
    np.random.seed(np.prod(shape) * len(shape) * entropy)
    return ndl.Tensor(np.random.randint(0, 100, size=shape) / 20, dtype="float32")

def get_int_tensor(*shape, low=0, high=10, entropy=1):
    np.random.seed(np.prod(shape) * len(shape) * entropy)
    return ndl.Tensor(np.random.randint(low, high, size=shape))

def logsumexp_forward(shape, axes):
    x = get_tensor(*shape)
    return (ndl.ops.logsumexp(x,axes=axes)).cached_data

def batchnorm_forward(*shape, affine=False):
    x = get_tensor(*shape)
    bn = ndl.nn.BatchNorm1d(shape[1])
    if affine:
        bn.weight.data = get_tensor(shape[1], entropy=42)
        bn.bias.data = get_tensor(shape[1], entropy=1337)
    return bn(x).cached_data

def logsumexp_backward(shape, axes):
    x = get_tensor(*shape)
    y = (ndl.ops.logsumexp(x, axes=axes)**2).sum()
    y.backward()
    return x.grad.cached_data

def softmax_loss_forward(rows, classes):
    x = get_tensor(rows, classes)
    y = get_int_tensor(rows, low=0, high=classes)
    f = ndl.nn.SoftmaxLoss()
    return np.array(f(x, y).cached_data)

def softmax_loss_backward(rows, classes):
    x = get_tensor(rows, classes)
    y = get_int_tensor(rows, low=0, high=classes)
    f = ndl.nn.SoftmaxLoss()
    loss = f(x, y)
    loss.backward()
    return x.grad.cached_data

def layernorm_forward(shape, dim):
    f = ndl.nn.LayerNorm1d(dim)
    x = get_tensor(*shape)
    return f(x).cached_data

def flatten_forward(*shape):
    x = get_tensor(*shape)
    tform = ndl.nn.Flatten()
    return tform(x).cached_data

def dropout_forward(shape, prob=0.5):
    np.random.seed(3)
    x = get_tensor(*shape)
    f = nn.Dropout(prob)
    return f(x).cached_data

def learn_model_1d(feature_size, nclasses, _model, optimizer, epochs=1, **kwargs):
    np.random.seed(42)
    model = _model([])
    X = get_tensor(1024, feature_size).cached_data
    y = get_int_tensor(1024, low=0, high=nclasses).cached_data.astype(np.uint8)
    m = X.shape[0]
    batch = 32

    loss_func = nn.SoftmaxLoss()
    opt = optimizer(model.parameters(), **kwargs)

    for _ in range(epochs):
        for i, (X0, y0) in enumerate(zip(np.array_split(X, m//batch), np.array_split(y, m//batch))):
            opt.reset_grad()
            X0, y0 = ndl.Tensor(X0, dtype="float32"), ndl.Tensor(y0)
            out = model(X0)
            loss = loss_func(out, y0)
            loss.backward()
            # Opt should not change gradients.
            grad_before = model.parameters()[0].grad.detach().cached_data
            opt.step()
            grad_after = model.parameters()[0].grad.detach().cached_data
            np.testing.assert_allclose(grad_before, grad_after, rtol=1e-5, atol=1e-5, \
                                       err_msg="Optim should not modify gradients in place")


    return np.array(loss.cached_data)


def test_flip_horizontal():
    tform = ndl.data.RandomFlipHorizontal()
    np.random.seed(0)
    a = np.array([[[0.6788795301189603, 0.7206326547259168, 0.5820197920751071, 0.5373732294490107, 0.7586156243223572], [0.10590760718779213, 0.4736004193466574, 0.18633234332675996, 0.7369181771289581, 0.21655035442437187]], [[0.13521817340545206, 0.3241410077932141, 0.14967486718368317, 0.22232138825158765, 0.38648898112586194], [0.9025984755294046, 0.4499499899112276, 0.6130634578841324, 0.9023485831739843, 0.09928035035897387]], [[0.9698090677467488, 0.6531400357979377, 0.17090958513604515, 0.358152166969525, 0.7506861412184562], [0.6078306687154678, 0.3250472290083525, 0.038425426472734725, 0.634274057957335, 0.9589492686245203]], [[0.6527903170054908, 0.6350588736035638, 0.9952995676778876, 0.5818503294385343, 0.4143685882263688], [0.4746975022884129, 0.6235101011318682, 0.33800761483889175, 0.6747523222590207, 0.3172017420692961]], [[0.778345482025909, 0.9495710534507421, 0.6625268669500443, 0.013571635612109834, 0.6228460955466695], [0.6736596308357894, 0.9719450024996658, 0.878193471347177, 0.5096243767199001, 0.05571469370160631]], [[0.4511592145209281, 0.019987665408758737, 0.44171092124884537, 0.9795867288127285, 0.3594444639693215], [0.4808935308361628, 0.6886611828057704, 0.8804758892525955, 0.9182354663621447, 0.21682213762754288]]])
    b = np.array([[[0.6788795301189603, 0.7206326547259168, 0.5820197920751071, 0.5373732294490107, 0.7586156243223572], [0.10590760718779213, 0.4736004193466574, 0.18633234332675996, 0.7369181771289581, 0.21655035442437187]], [[0.13521817340545206, 0.3241410077932141, 0.14967486718368317, 0.22232138825158765, 0.38648898112586194], [0.9025984755294046, 0.4499499899112276, 0.6130634578841324, 0.9023485831739843, 0.09928035035897387]], [[0.9698090677467488, 0.6531400357979377, 0.17090958513604515, 0.358152166969525, 0.7506861412184562], [0.6078306687154678, 0.3250472290083525, 0.038425426472734725, 0.634274057957335, 0.9589492686245203]], [[0.6527903170054908, 0.6350588736035638, 0.9952995676778876, 0.5818503294385343, 0.4143685882263688], [0.4746975022884129, 0.6235101011318682, 0.33800761483889175, 0.6747523222590207, 0.3172017420692961]], [[0.778345482025909, 0.9495710534507421, 0.6625268669500443, 0.013571635612109834, 0.6228460955466695], [0.6736596308357894, 0.9719450024996658, 0.878193471347177, 0.5096243767199001, 0.05571469370160631]], [[0.4511592145209281, 0.019987665408758737, 0.44171092124884537, 0.9795867288127285, 0.3594444639693215], [0.4808935308361628, 0.6886611828057704, 0.8804758892525955, 0.9182354663621447, 0.21682213762754288]]])
    np.testing.assert_allclose(tform(a), b)
    
def test_mnist_dataset():
    # Test dataset sizing
    mnist_train_dataset = ndl.data.MNISTDataset("data/train-images-idx3-ubyte.gz",
                                                "data/train-labels-idx1-ubyte.gz")
    assert len(mnist_train_dataset) == 60000

    sample_norms = np.array([np.linalg.norm(mnist_train_dataset[idx][0]) for idx in [1,42,1000,2000,3000,4000,5000,5005]])
    compare_against = np.array([10.188792, 6.261355, 8.966858, 9.4346485, 9.086626, 9.214664, 10.208544, 10.649756])
    sample_labels = np.array([mnist_train_dataset[idx][1] for idx in [1,42,1000,2000,3000,4000,5000,5005]])
    compare_labels = np.array([0,7,0,5,9,7,7,8])

    np.testing.assert_allclose(sample_norms, compare_against)
    np.testing.assert_allclose(sample_labels, compare_labels)

    mnist_train_dataset = ndl.data.MNISTDataset("data/t10k-images-idx3-ubyte.gz",
                                               "data/t10k-labels-idx1-ubyte.gz")
    assert len(mnist_train_dataset) == 10000

    sample_norms = np.array([np.linalg.norm(mnist_train_dataset[idx][0]) for idx in [1,42,1000,2000,3000,4000,5000,5005]])
    compare_against = np.array([9.857545, 8.980832, 8.57207 , 6.891522, 8.192135, 9.400087, 8.645003, 7.405202])
    sample_labels = np.array([mnist_train_dataset[idx][1] for idx in [1,42,1000,2000,3000,4000,5000,5005]])
    compare_labels = np.array([2, 4, 9, 6, 6, 9, 3, 1])

    np.testing.assert_allclose(sample_norms, compare_against, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(sample_labels, compare_labels)

    # test a transform
    np.random.seed(0)
    tforms = [ndl.data.RandomCrop(28), ndl.data.RandomFlipHorizontal()]
    mnist_train_dataset = ndl.data.MNISTDataset("data/train-images-idx3-ubyte.gz",
                                                "data/train-labels-idx1-ubyte.gz",
                                                transforms=tforms)

    sample_norms = np.array([np.linalg.norm(mnist_train_dataset[idx][0]) for idx in [1,42,1000,2000,3000,4000,5000,5005]])
    compare_against = np.array([2.0228338 ,0.        ,7.4892044 ,0.,0.,3.8012788,9.583429,4.2152724])
    sample_labels = np.array([mnist_train_dataset[idx][1] for idx in [1,42,1000,2000,3000,4000,5000,5005]])
    compare_labels = np.array([0,7,0,5,9,7,7,8])

    np.testing.assert_allclose(sample_norms, compare_against, rtol=5e-07)
    np.testing.assert_allclose(sample_labels, compare_labels)


    # test a transform
    tforms = [ndl.data.RandomCrop(12), ndl.data.RandomFlipHorizontal(0.4)]
    mnist_train_dataset = ndl.data.MNISTDataset("data/train-images-idx3-ubyte.gz",
                                                "data/train-labels-idx1-ubyte.gz",
                                                transforms=tforms)
    sample_norms = np.array([np.linalg.norm(mnist_train_dataset[idx][0]) for idx in [1,42,1000,2000,3000,4000,5000,5005]])
    compare_against = np.array([5.369537, 5.5454974, 8.966858, 7.547235, 8.785921, 7.848442, 7.1654058, 9.361828])
    sample_labels = np.array([mnist_train_dataset[idx][1] for idx in [1,42,1000,2000,3000,4000,5000,5005]])
    compare_labels = np.array([0,7,0,5,9,7,7,8])

    np.testing.assert_allclose(sample_norms, compare_against, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(sample_labels, compare_labels)


# print(learn_model_1d(64, 16, lambda z: nn.Sequential(nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 16)), ndl.optim.SGD, lr=0.01, momentum=0.0))
# test_flip_horizontal()
test_mnist_dataset()