import torch
import pandas as pd
import numpy as np
import os
import psutil
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torchvision.models import resnet18
from torchvision.models.feature_extraction import create_feature_extractor
from tqdm import tqdm
from transformers import BertTokenizer, BertModel


DATA_FOLDER = '~/data/'
PP_FOLDER = 'preprocessed'


def get_regression_data(dataset, cnt=10000, normalize_y=True, verbose=True):
    # dataset = e.g. 'ctscan' or 'yearpred'

    if dataset == 'yearpred':
        dat = pd.read_csv(os.path.join(DATA_FOLDER, 'YearPredictionMSD.txt'), header=None)

        X = dat.iloc[:, 1:].values.astype('float')
        y = dat.iloc[:, 0].values.astype('float')

    elif dataset == 'bike':
        dat = pd.read_csv(os.path.join(DATA_FOLDER, 'bike_sharing', 'hour.csv')).iloc[:, 2:]
        for col in ['casual', 'registered']:
            del dat[col]

        X = dat.iloc[:, :-1].values.astype('float')
        y = dat.iloc[:, -1].values.astype('float')

    elif dataset == 'ctscan':
        dat = pd.read_csv(os.path.join(DATA_FOLDER, 'slice_localization_data.csv')).iloc[:, 1:]

        X = dat.iloc[:, :-1].values.astype('float')
        y = dat.iloc[:, -1].values.astype('float')
    else:
        raise NotImplementedError

    if normalize_y:  # normalizes based on the whole dataset
        y -= y.min()
        y /= (y.max()/2)
        y -= 1.0

    np.random.seed(0)
    inds = np.random.choice(range(y.shape[0]), size=min(y.shape[0], cnt), replace=False)

    X = X[inds, ...]
    y = y[inds, ...]

    if verbose:
        # print(dat.shape)

        # process = psutil.Process(os.getpid())
        # print('memory:', process.memory_info().rss)

        print(dataset, X.shape, y.shape)

    return X, y


def pp_reg():
    for dataset in ['yearpred', 'bike', 'ctscan']:
        X, y = get_regression_data(dataset)
        np.save(os.path.join(PP_FOLDER, f"{dataset}_X.npy"), X)
        np.save(os.path.join(PP_FOLDER, f"{dataset}_y.npy"), y)

#######################################################


def load_mnist(task, seed=0):
    dataset = datasets.MNIST(root='~/data', train=True, download=False)

    dataset.data, dataset.targets = binary_mnist(
        dataset.data, dataset.targets, neg_class=task[0], pos_class=task[1], seed=seed)
    print('mnist', dataset.data.shape, dataset.targets.shape, len(dataset), torch.min(dataset.targets), torch.max(dataset.targets), torch.mean(dataset.targets.float()))
    
    return dataset.data.cpu().numpy(), dataset.targets.cpu().numpy()


def binary_mnist(data, labels, neg_class=0, pos_class=1, seed=0):
    mask = (labels == pos_class)
    idx = torch.argwhere(labels == neg_class)[:, 0]
    mask[idx] = True
    binary_labels = labels[mask]
    binary_labels[binary_labels == neg_class] = 0
    binary_labels[binary_labels == pos_class] = 1
    binary_data = data[mask]
    return binary_data, binary_labels


def pp_mnist():
    X, y = load_mnist((0, 1))
    X = X.reshape(*X.shape[:-2], -1)
    print(X.shape, y.shape)
    np.save(os.path.join(PP_FOLDER, f"mnist_X.npy"), X)
    np.save(os.path.join(PP_FOLDER, f"mnist_y.npy"), y)


def load_usps(task, seed=0):
    dataset = datasets.USPS(root='~/data', train=True, download=False)

    dataset.data, dataset.targets = binary_usps(
        dataset.data, dataset.targets, neg_class=task[0], pos_class=task[1])

    dataset.data = torch.tensor(dataset.data)
    dataset.targets = torch.tensor(dataset.targets)
    print('usps', dataset.data.shape, dataset.targets.shape, len(dataset), torch.min(dataset.targets), torch.max(dataset.targets), torch.mean(dataset.targets.float()))
    
    return dataset.data.cpu().numpy(), dataset.targets.cpu().numpy()


def binary_usps(data, labels, neg_class, pos_class):
    binary_labels = torch.Tensor(labels)
    mask = (binary_labels == neg_class) | (binary_labels == pos_class)
    binary_labels = binary_labels[mask]
    binary_labels[binary_labels == neg_class] = 0
    binary_labels[binary_labels == pos_class] = 1
    binary_data = data[mask]
    binary_labels = binary_labels.tolist()
    return binary_data, binary_labels


def pp_usps():
    X, y = load_usps((0, 1))
    X = X.reshape(*X.shape[:-2], -1)
    print(X.shape, y.shape)
    np.save(os.path.join(PP_FOLDER, f"usps_X.npy"), X)
    np.save(os.path.join(PP_FOLDER, f"usps_y.npy"), y)


#######################################################


def load_vision(dataset_name, task, seed=0):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    transform = transforms.Compose([
                transforms.Resize(256), 
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize,
            ])

    if dataset_name == 'cifar10':
        dataset = datasets.CIFAR10(root='~/data', train=True, download=False, transform=transform)
        dataset.data, dataset.targets = binary_vision(
            dataset.data, dataset.targets, neg_class=task[0], pos_class=task[1])
        print(dataset_name, dataset.data.shape, len(dataset.targets), len(dataset), min(dataset.targets), max(dataset.targets))
    elif dataset_name == 'cifar100':
        dataset = datasets.CIFAR100(root='~/data', train=True, download=False, transform=transform)
        dataset.data, dataset.targets = binary_vision(
            dataset.data, dataset.targets, neg_class=task[0], pos_class=task[1])
        print(dataset_name, dataset.data.shape, len(dataset.targets), len(dataset), min(dataset.targets), max(dataset.targets))
    elif dataset_name == 'STL10':
        dataset = datasets.STL10(root='~/data', split='train', download=False, transform=transform)
        dataset.data, dataset.labels = binary_vision(
            dataset.data, dataset.labels, neg_class=task[0], pos_class=task[1])
        print(dataset_name, dataset.data.shape, len(dataset.labels), len(dataset), min(dataset.labels), max(dataset.labels))
    elif dataset_name == 'SVHN':
        dataset = datasets.SVHN(root='~/data', split='train', download=False, transform=transform)
        dataset.data, dataset.labels = binary_vision(
            dataset.data, dataset.labels, neg_class=task[0], pos_class=task[1])
        print(dataset_name, dataset.data.shape, len(dataset.labels), len(dataset), min(dataset.labels), max(dataset.labels))
    
    return dataset


def binary_vision(data, labels, neg_class, pos_class):
    binary_labels = torch.Tensor(labels)
    mask = (binary_labels == neg_class) | (binary_labels == pos_class)
    binary_labels = binary_labels[mask]
    binary_labels[binary_labels == neg_class] = 0
    binary_labels[binary_labels == pos_class] = 1
    binary_data = data[mask]
    binary_labels = binary_labels.tolist()
    return binary_data, binary_labels


def pp_vision(dataset_name):
    torch.manual_seed(0)
    np.random.seed(0)
    dataset = load_vision(dataset_name, (0, 1))
    loader = torch.utils.data.DataLoader(dataset, batch_size=1024,
                                         shuffle=True, num_workers=8)
    model = resnet18().cuda()
    model.load_state_dict(torch.load("/home/weights/hub/checkpoints/resnet18-f37072fd.pth"))
    # print(model)

    layer = 'avgpool'
    feature_extractor = create_feature_extractor(
        model, return_nodes=[layer])

    rep = []
    y = []
    with torch.no_grad():
        for batch_X, batch_y in tqdm(loader):
            batch_X = batch_X.cuda()
            batch_rep = torch.squeeze(feature_extractor(batch_X)[layer])
            rep.append(batch_rep.detach().cpu().numpy())
            y.append(batch_y)

    rep = np.vstack(rep)
    y = np.hstack(y)
    print(rep.shape, y.shape)
    np.save(os.path.join(PP_FOLDER, f"{dataset_name}_X.npy"), rep)
    np.save(os.path.join(PP_FOLDER, f"{dataset_name}_y.npy"), y)


#######################################################


def get_agnews(task, cnt=10000):
    dat = pd.read_csv(os.path.join(DATA_FOLDER, 'agnews', 'train.csv'))
    dat['Class Index'] = dat['Class Index'] - 1    

    mask = (dat['Class Index'] == task[0]) | (dat['Class Index'] == task[1])
    dat = dat[mask]
    dat.loc[dat['Class Index'] == task[0], 'Class Index'] = 0
    dat.loc[dat['Class Index'] == task[1], 'Class Index'] = 1

    np.random.seed(0)
    inds = np.random.choice(range(len(dat)), size=min(len(dat), cnt), replace=False)
    dat = dat.iloc[inds]

    y = dat['Class Index'].values.astype('float')
    print(y.shape, y)
    # X = dat.iloc[:, 1:].values.astype('float')

    tokenizer = BertTokenizer.from_pretrained('bert-base-cased', do_lowercase=False)
    model = BertModel.from_pretrained('bert-base-cased').cuda()

    lens = []
    for sentence in dat['Description']:
        lens.append(len(tokenizer.encode(sentence)))
    print('max leng:', max(lens))

    batch_size = 32
    ind = 0
    X = np.zeros((dat.shape[0], 768))
    while ind < dat.shape[0]:
        encoding = tokenizer.batch_encode_plus(
              dat['Description'][ind:ind+batch_size],
              add_special_tokens=True,
              truncation=True,
              max_length=max(lens),
              return_token_type_ids=False,
              pad_to_max_length=True,
              return_attention_mask=True,
              return_tensors='pt',
        )
    #     print(encoding['input_ids'].shape, encoding['attention_mask'].shape)
        _, pooled_output = model(
          input_ids=encoding['input_ids'].cuda(),
          attention_mask=encoding['attention_mask'].cuda(),
          return_dict=False,
        )
        X[ind:ind+batch_size] = pooled_output.detach().cpu().numpy()
        ind += batch_size
        # if ind % 100 == 0: 
        print(f"{ind}/{dat.shape[0]}")

    np.save(os.path.join(PP_FOLDER, f"agnews_X.npy"), X)
    np.save(os.path.join(PP_FOLDER, f"agnews_y.npy"), y)

get_agnews(task=(0, 1))