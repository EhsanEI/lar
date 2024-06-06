import numpy as np


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


def add_bias_feature(X):
    return np.hstack((X, np.ones((X.shape[0], 1))))  # instead of bias unit
