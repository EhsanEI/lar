import numpy as np
import os
import matplotlib.pyplot as plt
from tabulate import tabulate


def add_bias_feature(X):
    return np.hstack((X, np.ones((X.shape[0], 1))))  # instead of bias unit


PP_FOLDER = 'preprocessed'

datasets = ['ctscan', 'yearpred', 'bike',  'mnist', 'usps']
# datasets = ['cifar10', 'cifar100', 'STL10', 'XED_en', 'agnews']
# datasets = []

threshold = .9

table = []
for dataset in datasets:
    X = np.load(os.path.join(PP_FOLDER, f"{dataset}_X.npy"))
    y = np.load(os.path.join(PP_FOLDER, f"{dataset}_y.npy"))
    X = add_bias_feature(X)
    # print(X.shape, y.shape, y.min(), y.max())

    if y.min() > -1e-5:
        y = (y - 0.5) * 2

    rank = np.linalg.matrix_rank(X)
    
    y_norm = (y ** 2).sum() ** 0.5
    u, s, vt = np.linalg.svd(X)
    dot_prods = (u.T @ y) ** 2
    cumu = np.cumsum(dot_prods) ** 0.5
    cumu_norm = cumu/cumu[X.shape[1]-1]#y_norm
    alignment_rank = np.argwhere(cumu_norm > threshold)[0][0]
    row = [dataset, X.shape[0], X.shape[1], rank, alignment_rank + 1]
    print(row)
    table.append(row)
    # print(cumu_norm[:200])
    # plt.clf()
    # plt.plot(cumu_norm[:300])
    # plt.savefig('dot_prods.png')
print(tabulate(table, headers=['Task', 'n', 'd', 'rank{\Phi}', 'k(0.9)'], tablefmt='latex'))