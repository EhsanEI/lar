import torchvision.datasets as datasets
from torch.utils.data import SubsetRandomSampler, DataLoader
from torchvision import transforms
import torch
import params
from datautils import binary_imbalanced_mnist as binary_imbalanced
import numpy as np
from sklearn.model_selection import train_test_split


def create_loaders(classes, ratio, validation=False):
    # transform = transforms.Compose([transforms.ToTensor(),
    #                                 transforms.Normalize((0.29730626, 0.29918741, 0.27534935),
    #                                                      (0.32780124, 0.32292358, 0.32056796)),
    #                                 ])
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.29730626),
                                                         (0.32780124)),
                                    ])

    mnist_train_dataset = datasets.MNIST(root='~/data', train=True, download=True,
                                         transform=transform)
    mnist_valid_dataset = datasets.MNIST(root='~/data', train=True, download=True,
                                         transform=transform)
    mnist_test_dataset = datasets.MNIST(root='~/data', train=False, transform=transform)

    for dataset in [mnist_train_dataset, mnist_test_dataset]:
        dataset.data, dataset.targets = binary_imbalanced(
            dataset.data, dataset.targets, neg_class=classes[0], pos_class=classes[1], ratio=ratio)
        print('mnist', dataset.data.shape, dataset.targets.shape, len(dataset), torch.min(dataset.targets), torch.max(dataset.targets), torch.mean(dataset.targets.float()))

    # Validation data
    mnist_valid_dataset = datasets.MNIST(root='~/data', train=True, download=True, # Split doesn't matter. Will replace its data.
                                         transform=transform)
    if validation:
        X_train = mnist_train_dataset.data.numpy()
        y_train = mnist_train_dataset.targets.numpy()
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=100, random_state=0)
        mnist_train_dataset.data = torch.from_numpy(X_train)
        mnist_train_dataset.targets = torch.from_numpy(y_train)
        mnist_valid_dataset.data = torch.from_numpy(X_val)
        mnist_valid_dataset.targets = torch.from_numpy(y_val)
        print('mnist', mnist_valid_dataset.data.shape, mnist_valid_dataset.targets.shape, len(mnist_valid_dataset), torch.min(mnist_valid_dataset.targets), torch.max(mnist_valid_dataset.targets), torch.mean(mnist_valid_dataset.targets.float()))

    indices = list(range(len(mnist_train_dataset))) 
    train_idx = indices
    train_sampler = SubsetRandomSampler(train_idx)

    mnist_train_loader = DataLoader(
        mnist_train_dataset,
        batch_size=params.batch_size,
        sampler=train_sampler,
        num_workers=params.num_workers
    )

    mnist_valid_loader = DataLoader(
        mnist_valid_dataset,
        batch_size=params.batch_size,
        num_workers=params.num_workers
    )

    mnist_test_loader = DataLoader(
        mnist_test_dataset,
        batch_size=params.batch_size,
        num_workers=params.num_workers
    )

    return mnist_train_loader, mnist_valid_loader, mnist_test_loader


# mnist_train_all = (mnist_train_dataset.train_data[5000:].reshape(55000, 28, 28, 1))
# mnist_concat = torch.cat((mnist_train_all, mnist_train_all, mnist_train_all), 3)
# print(mnist_test_dataset.test_labels.shape, mnist_test_dataset.test_labels)


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


# print(one_hot_embedding(mnist_test_dataset.test_labels))

# print(mnist_concat.shape)


# def test():
    # print(mnist_train_loader.shape)
    # print(len(train_sampler), len(mnist_test_loader), len(valid_sampler))
    # print(len(mnist_train_loader), len(mnist_valid_loader), len(mnist_test_loader))
    # for i, train_data in enumerate(mnist_train_loader):
    #     img, label = train_data
    #     print(img.shape)
    # for i in range(1):
    #     # for batch_idx, (inputs, labels) in enumerate(train_loader):
    #     #     print(i, batch_idx, labels, len(labels))
    # mnist_train_all = (mnist_train_dataset.train_data[5000:].reshape(55000, 28, 28, 1))
    # mnist_concat = torch.cat((mnist_train_all, mnist_train_all, mnist_train_all), 3)
    # print(mnist_concat.shape)
    # print(list(mnist_train_dataset.train_data[5000:].size()))
    # print(mnist_train_dataset.train_data.float().mean()/255)
    # print(mnist_train_dataset.train_data.float().std()/255)
    # for batch_idx, (train_data, test_data) in enumerate(zip(mnist_train_loader, mnist_valid_loader)):
    #     train_image, train_label = train_data
    #     test_image, test_label = test_data
    #     print(train_image.shape)
    #     # print(train_label, len(train_label))
    #     # print(test_label, len(test_label))
    #     # exit()

# test()
