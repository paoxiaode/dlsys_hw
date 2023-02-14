import sys

sys.path.append("../python")
import needle as ndl
import needle.nn as nn
import numpy as np
import time
import os

np.random.seed(0)


def ResidualBlock(dim, hidden_dim, norm=nn.BatchNorm1d, drop_prob=0.1):
    ### BEGIN YOUR SOLUTION
    model = nn.Sequential(
        nn.Linear(dim, hidden_dim),
        norm(dim=hidden_dim),
        nn.ReLU(),
        nn.Dropout(p=drop_prob),
        nn.Linear(hidden_dim, dim),
        norm(dim=dim),
    )
    return nn.Sequential(nn.Residual(model), nn.ReLU())
    ### END YOUR SOLUTION


def MLPResNet(
    dim,
    hidden_dim=100,
    num_blocks=3,
    num_classes=10,
    norm=nn.BatchNorm1d,
    drop_prob=0.1,
):
    ### BEGIN YOUR SOLUTION
    layers = [nn.Linear(dim, hidden_dim), nn.ReLU()]
    for _ in range(num_blocks):
        layers.append(
            ResidualBlock(hidden_dim, hidden_dim // 2, norm=norm, drop_prob=drop_prob)
        )
    layers.append(nn.Linear(in_features=hidden_dim, out_features=num_classes))
    return nn.Sequential(*layers)
    ### END YOUR SOLUTION


def epoch(dataloader, model, opt=None):
    np.random.seed(4)
    ### BEGIN YOUR SOLUTION
    if opt:
        model.train()
    else:
        model.eval()
    loss_func = nn.SoftmaxLoss()
    total_err, total_loss = 0, 0
    total_count = 0
    for data in dataloader:
        X, y = data
        n, *_ = X.shape
        X = X.reshape((n, -1))
        y_hat = model(X)
        loss = loss_func(y_hat, y)
        if opt:
            loss.backward()
            opt.step()
        total_loss += loss.numpy() * n
        y_pred = np.argmax(y_hat.numpy(), axis=1)
        total_err += (y_pred != y.numpy()).sum()
        total_count += n
    return total_err / total_count, total_loss / total_count
    ### END YOUR SOLUTION


def train_mnist(
    batch_size=100,
    epochs=10,
    optimizer=ndl.optim.Adam,
    lr=0.001,
    weight_decay=0.001,
    hidden_dim=100,
    data_dir="data",
):
    np.random.seed(4)
    ### BEGIN YOUR SOLUTION
    train_dataset = ndl.data.MNISTDataset(
        f"{data_dir}/train-images-idx3-ubyte.gz",
        f"{data_dir}/train-labels-idx1-ubyte.gz",
    )
    test_dataset = ndl.data.MNISTDataset(
        f"{data_dir}/t10k-images-idx3-ubyte.gz", f"{data_dir}/t10k-labels-idx1-ubyte.gz"
    )
    train_dataloader = ndl.data.DataLoader(train_dataset, batch_size, shuffle=True)
    test_dataloader = ndl.data.DataLoader(test_dataset, batch_size, shuffle=False)

    model = MLPResNet(784, hidden_dim=hidden_dim)
    opt = optimizer(model.parameters(), lr=lr, weight_decay=weight_decay)
    for i in range(epochs):
        train_err, train_loss = epoch(train_dataloader, model, opt)
        test_err, test_loss = epoch(test_dataloader, model)
    return train_err, train_loss, test_err, test_loss
    ### END YOUR SOLUTION


if __name__ == "__main__":
    train_mnist(data_dir="../data")
