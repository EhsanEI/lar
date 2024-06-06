import torchvision.datasets as datasets
from torch.utils.data import Dataset, SubsetRandomSampler, DataLoader
from torchvision import transforms
import torch
import params
from datautils import binary_imbalanced_usps as binary_imbalanced
import numpy as np
from sklearn.model_selection import train_test_split


class NumpyDataset(Dataset):
    def __init__(self, data, targets):
        self.data = data
        self.targets = torch.LongTensor(targets)
        
    def __getitem__(self, index):
        x = self.data[index]
        y = self.targets[index]
        
        return x, y
    
    def __len__(self):
        return len(self.data)


def subsample(X, y, ratio):
    if ratio is None:
        return X, y
    
    neg_inds = np.argwhere(y == 0)[:, 0]
    neg_inds = neg_inds[:int(len(neg_inds)*ratio)]
    pos_inds = np.argwhere(y == 1)[:, 0]
    inds = np.hstack((neg_inds, pos_inds))
    np.random.shuffle(inds)
    return X[inds], y[inds]


def create_loaders(meta_data, target=False, seed=0):
    n_train = meta_data['data_cnt']
    n_test = meta_data['data_cnt']
    n_val = meta_data['val_cnt']

    d = 2
    cov_mat = np.zeros((d, d)) + 0.5
    cov_mat[0, 0] = 1

    rng = np.random.default_rng(seed=seed)

    # Train
    X_train = rng.multivariate_normal(np.zeros(d), cov_mat, n_train)

    eigvals_s, eigvecs_s = np.linalg.eigh(cov_mat)
    idx_s = eigvals_s.argsort()[::-1]
    eigvals_s = eigvals_s[idx_s]
    eigvecs_s = eigvecs_s[:, idx_s]

    y_train = 1.0 * ((X_train @ eigvecs_s[:, 0]) > 0)

    theta = np.pi * 1/4

    r = np.array(((np.cos(theta), -np.sin(theta)),
                 (np.sin(theta),  np.cos(theta))))

    if target:
        X_train = X_train @ r
    else:
        X_train, y_train = subsample(X_train, y_train, meta_data['ratio'])

    # Target
    X_test = rng.multivariate_normal(np.zeros(d), cov_mat, n_test)

    eigvals, eigvecs = np.linalg.eigh(cov_mat)
    idx = eigvals.argsort()[::-1]
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]

    y_test = 1.0 * ((X_test @ eigvecs[:, 0]) > 0)

    if target:
        X_test = X_test @ r
    else:
        X_test, y_test = subsample(X_test, y_test, meta_data['ratio'])

    # Val
    X_val = rng.multivariate_normal(np.zeros(d), cov_mat, n_val)

    eigvals, eigvecs = np.linalg.eigh(cov_mat)
    idx = eigvals.argsort()[::-1]
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]

    y_val = 1.0 * ((X_val @ eigvecs[:, 0]) > 0)

    if target:
        X_val = X_val @ r
    else:
        X_val, y_val = subsample(X_val, y_val, meta_data['ratio'])

    train_dataset = NumpyDataset(X_train, y_train)
    test_dataset = NumpyDataset(X_test, y_test)
    val_dataset = NumpyDataset(X_val, y_val)

    train_loader = DataLoader(
        train_dataset,
        shuffle=True,
        batch_size=params.batch_size,
        num_workers=params.num_workers
    )

    test_loader = DataLoader(
        test_dataset,
        shuffle=True,
        batch_size=params.batch_size,
        num_workers=params.num_workers
    )

    val_loader = DataLoader(
        val_dataset,
        shuffle=True,
        batch_size=params.batch_size,
        num_workers=params.num_workers
    )

    return train_loader, val_loader, test_loader