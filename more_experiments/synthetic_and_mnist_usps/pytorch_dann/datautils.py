import torch


def binary_imbalanced_mnist(data, labels, neg_class, pos_class, ratio=None, seed=0):
    mask = (labels == pos_class)
    idx = torch.argwhere(labels == neg_class)[:, 0]
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
