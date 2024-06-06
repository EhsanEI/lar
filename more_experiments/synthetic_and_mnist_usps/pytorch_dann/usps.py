import torchvision.datasets as datasets
from torch.utils.data import SubsetRandomSampler, DataLoader
from torchvision import transforms
import torch
import params
from datautils import binary_imbalanced_usps as binary_imbalanced
import numpy as np
from sklearn.model_selection import train_test_split


def create_loaders(classes, validation=False):
    # transform = transforms.Compose([transforms.ToTensor(),
    #                                 transforms.Normalize((0.29730626, 0.29918741, 0.27534935),
    #                                                      (0.32780124, 0.32292358, 0.32056796)),
    #                                 ])
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Resize([28, ]),
                                    transforms.Normalize((0.29730626),
                                                         (0.32780124)),
                                    ])

    usps_train_dataset = datasets.USPS(root='~/data', train=True, download=True,
                                         transform=transform)
    usps_test_dataset = datasets.USPS(root='~/data', train=False, transform=transform) 

    for dataset in [usps_train_dataset, usps_test_dataset]:
        dataset.data, dataset.targets = binary_imbalanced(dataset.data, dataset.targets, neg_class=classes[0], pos_class=classes[1], ratio=None)
        print('usps', dataset.data.shape, len(dataset.targets), len(dataset), np.min(dataset.targets), np.max(dataset.targets), np.mean(dataset.targets))

    # Validation data
    usps_valid_dataset = datasets.USPS(root='~/data', train=True, download=True, # Split doesn't matter. Will replace its data.
                                         transform=transform)
    if validation:
        X_train = usps_train_dataset.data
        y_train = np.array(usps_train_dataset.targets)
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=100, random_state=0)
        usps_train_dataset.data = X_train
        usps_train_dataset.targets = y_train.tolist()
        usps_valid_dataset.data = X_val
        usps_valid_dataset.targets = y_val.tolist()
        print('usps', usps_valid_dataset.data.shape, len(usps_valid_dataset.targets), len(usps_valid_dataset), np.min(usps_valid_dataset.targets), np.max(usps_valid_dataset.targets), np.mean(usps_valid_dataset.targets))

    indices = list(range(len(usps_train_dataset)))
    train_idx = indices
    train_sampler = SubsetRandomSampler(train_idx)

    usps_train_loader = DataLoader(
        usps_train_dataset,
        batch_size=params.batch_size,
        sampler=train_sampler,
        num_workers=params.num_workers
    )

    usps_valid_loader = DataLoader(
        usps_valid_dataset,
        batch_size=params.batch_size,
        num_workers=params.num_workers
    )

    usps_test_loader = DataLoader(
        usps_test_dataset,
        batch_size=params.batch_size,
        num_workers=params.num_workers
    )

    return usps_train_loader, usps_valid_loader, usps_test_loader


# usps_train_all = (usps_train_dataset.train_data[5000:].reshape(55000, 28, 28, 1))
# usps_concat = torch.cat((usps_train_all, usps_train_all, usps_train_all), 3)
# print(usps_test_dataset.test_labels.shape, usps_test_dataset.test_labels)


def one_hot_embedding(labels, num_classes=10):
    """Embedding labels to one-hot form.

    Args:
      labels: (LongTensor) class labels, sized [N,].
      num_classes: (int) number of classes.

    Returns:
      (tensor) encoded labels, sized [N, #classes].
    """
    y = torch.eye(num_classes)
    return y[labels]


# print(one_hot_embedding(usps_test_dataset.test_labels))

# print(usps_concat.shape)


# def test():
    # print(usps_train_loader.shape)
    # print(len(train_sampler), len(usps_test_loader), len(valid_sampler))
    # print(len(usps_train_loader), len(usps_valid_loader), len(usps_test_loader))
    # for i, train_data in enumerate(usps_train_loader):
    #     img, label = train_data
    #     print(img.shape)
    # for i in range(1):
    #     # for batch_idx, (inputs, labels) in enumerate(train_loader):
    #     #     print(i, batch_idx, labels, len(labels))
    # usps_train_all = (usps_train_dataset.train_data[5000:].reshape(55000, 28, 28, 1))
    # usps_concat = torch.cat((usps_train_all, usps_train_all, usps_train_all), 3)
    # print(usps_concat.shape)
    # print(list(usps_train_dataset.train_data[5000:].size()))
    # print(usps_train_dataset.train_data.float().mean()/255)
    # print(usps_train_dataset.train_data.float().std()/255)
    # for batch_idx, (train_data, test_data) in enumerate(zip(usps_train_loader, usps_valid_loader)):
    #     train_image, train_label = train_data
    #     test_image, test_label = test_data
    #     print(train_image.shape)
    #     # print(train_label, len(train_label))
    #     # print(test_label, len(test_label))
    #     # exit()

# test()
