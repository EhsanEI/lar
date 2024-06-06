import os
import numpy as np
from sklearn.preprocessing import StandardScaler
from copy import deepcopy
import torchvision.datasets as datasets
import torch
from torchvision import transforms


def subsample(X, y, ratio):
    if ratio is None:
        return X, y
    
    neg_inds = np.argwhere(y == 0)[:, 0]
    neg_inds = neg_inds[:int(len(neg_inds)*ratio)]
    pos_inds = np.argwhere(y == 1)[:, 0]
    inds = np.hstack((neg_inds, pos_inds))
    np.random.shuffle(inds)
    return X[inds], y[inds]


def get_data_synthetic_clf(meta_data, seed, classification=True):
    n_source = meta_data['data_cnt']
    n_target = meta_data['data_cnt']
    n_val = meta_data['val_cnt']

    d = 2
    # cov_mat = np.zeros((d, d)) + 0.
    # cov_mat[0, 0] = cov_mat[1, 1] = .5
    cov_mat = np.zeros((d, d)) + 0.5
    cov_mat[0, 0] = 1


    rng = np.random.default_rng(seed=seed)

    # Source
    X_source = rng.multivariate_normal(np.zeros(d), cov_mat, n_source)

    eigvals_s, eigvecs_s = np.linalg.eigh(cov_mat)
    idx_s = eigvals_s.argsort()[::-1]
    eigvals_s = eigvals_s[idx_s]
    eigvecs_s = eigvecs_s[:, idx_s]

    if classification:
        y_source = 1.0 * ((X_source @ eigvecs_s[:, 0]) > 0)
    else:
        y_source = (X_source @ (1.0 * eigvecs_s[:, 0] + 1. * eigvecs_s[:, 0]))

    theta = np.pi * 1/4 #- 1e-3#1/4

    r = np.array(((np.cos(theta), -np.sin(theta)),
                 (np.sin(theta),  np.cos(theta))))

    if classification:
        X_source, y_source = subsample(X_source, y_source, meta_data['ratio'])

    # Target
    X_target = rng.multivariate_normal(np.zeros(d), cov_mat, n_target)

    eigvals, eigvecs = np.linalg.eigh(cov_mat)
    idx = eigvals.argsort()[::-1]
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]

    if classification:
        y_target = 1.0 * ((X_target @ eigvecs[:, 0]) > 0)
    else:
        y_target = (X_target @ (1.0 * eigvecs[:, 0] + 0.0 * eigvecs[:, 0]))

    X_target = X_target @ r

    # Val
    X_val = rng.multivariate_normal(np.zeros(d), cov_mat, n_val)

    eigvals, eigvecs = np.linalg.eigh(cov_mat)
    idx = eigvals.argsort()[::-1]
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]

    if classification:
        y_val = 1.0 * ((X_val @ eigvecs[:, 0]) > 0)
    else:
        y_val = (X_val @ (1.0 * eigvecs[:, 0] + 0.0 * eigvecs[:, 0]))

    X_val = X_val @ r

    # y_source = np.reshape(y_source, newshape=(y_source.size, 1))
    # y_target = np.reshape(y_target, newshape=(y_target.size, 1))
    # y_val = np.reshape(y_val, newshape=(y_val.size, 1))

    return X_source, y_source, X_target, y_target, X_val, y_val


def binary_imbalanced_mnist(data, labels, neg_class, pos_class, ratio=None, seed=0):
    mask = (labels == pos_class)
    idx = np.argwhere(labels == neg_class)[:, 0]
    if ratio is not None:
        rng = torch.Generator()
        rng.manual_seed(seed)
        idx = idx[torch.randperm(len(idx), generator=rng)[:int(len(idx)*ratio)]]
        # print(idx[:10])
    mask[idx] = True
    binary_labels = labels[mask]
    binary_labels[binary_labels == neg_class] = 0
    binary_labels[binary_labels == pos_class] = 1
    binary_data = data[mask]
    return binary_data, binary_labels


def binary_imbalanced_usps(data, labels, neg_class, pos_class, ratio=None):
    assert ratio is None
    binary_labels = torch.Tensor(labels)
    mask = (binary_labels == neg_class) | (binary_labels == pos_class)
    binary_labels = binary_labels[mask]
    binary_labels[binary_labels == neg_class] = 0
    binary_labels[binary_labels == pos_class] = 1
    binary_data = data[mask]
    binary_labels = binary_labels.tolist()
    return binary_data, binary_labels


def load_mnist(task, ratio, seed):
    mnist_train_dataset = datasets.MNIST(root='~/data', train=True, download=True)
    mnist_test_dataset = datasets.MNIST(root='~/data', train=False)

    for dataset in [mnist_train_dataset, mnist_test_dataset]:
        dataset.data, dataset.targets = binary_imbalanced_mnist(
            dataset.data, dataset.targets, neg_class=task[0], pos_class=task[1], ratio=ratio, seed=seed)
        transform = transforms.Resize([28, ])
        dataset.data = transform(dataset.data)
        print('mnist', dataset.data.shape, dataset.targets.shape, len(dataset), torch.min(dataset.targets), torch.max(dataset.targets), torch.mean(dataset.targets.float()))
    
    return mnist_train_dataset.data.cpu().numpy(), mnist_train_dataset.targets.cpu().numpy(), \
        mnist_test_dataset.data.cpu().numpy(), mnist_test_dataset.targets.cpu().numpy()


def load_usps(task, ratio, seed):
    usps_train_dataset = datasets.USPS(root='~/data', train=True, download=True)
    usps_test_dataset = datasets.USPS(root='~/data', train=False, download=True) # Will test on the training portion of this dataset

    for dataset in [usps_train_dataset, usps_test_dataset]:
        dataset.data, dataset.targets = binary_imbalanced_usps(
            dataset.data, dataset.targets, neg_class=task[0], pos_class=task[1], ratio=ratio)

        dataset.data = torch.tensor(dataset.data)
        transform = transforms.Resize([28, ])
        dataset.data = transform(dataset.data)
        dataset.targets = torch.tensor(dataset.targets)
        print('usps', dataset.data.shape, dataset.targets.shape, len(dataset), torch.min(dataset.targets), torch.max(dataset.targets), torch.mean(dataset.targets.float()))
    
    return usps_train_dataset.data.cpu().numpy(), usps_train_dataset.targets.cpu().numpy(), \
        usps_test_dataset.data.cpu().numpy(), usps_test_dataset.targets.cpu().numpy()


load_data = {
    'mnist': load_mnist,
    'usps': load_usps,
}

###################


def normalize_len(X):
    X_len = ((X**2).sum(axis=1)**0.5)[:, None]
    return X / X_len


def normalize_mean_var(X):
    if len(X.shape) < 2:
        return StandardScaler().fit_transform(X[:, None])[:, 0]
    return StandardScaler().fit_transform(X)


def add_bias_feature(X):
    return np.hstack((X, np.ones((X.shape[0], 1))))  # instead of bias unit


def flatten_X(X):
    return X.reshape((X.shape[0], np.prod(X.shape[1:])))
