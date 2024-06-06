import io
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import matplotlib.pyplot as plt
import os
import json
from itertools import product
from sklearn.metrics import f1_score
import sys
from sklearn.model_selection import train_test_split


def config_str(config):
    return f"{config['lamda']}_{config['k']}_{config['ktilde']}"


def add_bias_feature(X):
    return np.hstack((X, np.ones((X.shape[0], 1))))  # instead of bias unit


device = torch.device(f'cuda:0') if torch.cuda.is_available() else torch.device('cpu')
loss = 'mse'

if len(sys.argv) > 2:
    src_lang = sys.argv[1]
    tgt_lang = sys.argv[2]
    seed = int(sys.argv[3])
else:
    src_lang = 'en'
    tgt_lang = 'es'
    seed = 0

print('languages', src_lang, tgt_lang)

aligned = False
n_epochs = 5000
lamdas = [0, 1e-1, 1e1, 1e3]
ks = [8, 16, 32, 64, 128, 256, 512, 768]
ktildes = [8, 16, 32, 64, 128, 256, 512]

f1averages = ['binary', 'macro', 'weighted']

hyperparams = {
    'lamdas': lamdas,
    'ks': ks,
    'ktildes': ktildes,
}
print('saving hyperparams')
with open(os.path.join('results_new_alg', f"{src_lang}_{tgt_lang}_hyperparams.json"), 'w') as f:
    json.dump(hyperparams, f)


configs = []

config = {
    'lamda': 0.0,
    'remove_implicit': False,
    'k': 0,
    'ktilde': 0,
    'n_epochs': n_epochs
}
configs.append(config)

if loss == 'mse' and not aligned:
    for lamda, k, ktilde in product(lamdas[1:], ks, ktildes):
        config = {
            'lamda': lamda,
            'remove_implicit': True,
            'k': k,
            'ktilde': ktilde,
            'n_epochs': n_epochs
        }
        configs.append(config)

results = {
    'src_acc': {},
    'tgt_acc': {},
    'val_acc': {},
    'src_f1': {},
    'tgt_f1': {},
    'val_f1': {},
}
for config in configs:
    for key in results.keys():
        results[key][config_str(config)] = []

with open(os.path.join('results_new_alg', f"configs.json"), 'w') as f:
    json.dump(configs, f)

##############################################################

# src_X = np.load('embeddings/en_X.npy')
# src_y = np.load('embeddings/en_y.npy')
# tgt_X = np.load('embeddings/es_X.npy')
# tgt_y = np.load('embeddings/es_y.npy')

if aligned:
    src_X = np.load(f"embeddings/bert_aligned/{src_lang}_{tgt_lang}/vectors-{src_lang}.npy")
    tgt_X = np.load(f"embeddings/bert_aligned/{src_lang}_{tgt_lang}/vectors-{tgt_lang}.npy")
else:
    src_X = np.load(f'embeddings/bert_{src_lang}_X.npy')
    tgt_X = np.load(f'embeddings/bert_{tgt_lang}_X.npy')

src_y = np.load(f'embeddings/bert_{src_lang}_y.npy')
tgt_y = np.load(f'embeddings/bert_{tgt_lang}_y.npy')


# from sklearn.svm import LinearSVC
# from sklearn.model_selection import train_test_split
# from sklearn import metrics
# X_train, X_test, y_train, y_test = train_test_split(tgt_X, tgt_y, test_size=0.3,random_state=0)
# print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
# clf = LinearSVC()
# clf.fit(X_train, y_train)
# y_pred = clf.predict(X_test)
# print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
# print("f1:",metrics.f1_score(y_test, y_pred))


print(src_X.shape, src_y.shape, tgt_X.shape, tgt_y.shape, src_y.mean(), tgt_y.mean())

if loss == 'mse':
    src_y = src_y * 2. - 1
    tgt_y = tgt_y * 2. - 1

if False:
    # Adding a constant feature to representation instead of bias unit
    src_X = add_bias_feature(src_X)
    tgt_X = add_bias_feature(tgt_X)
    
tgt_X, val_X, tgt_y, val_y = train_test_split(tgt_X, tgt_y, test_size=100, random_state=seed)


src_X = torch.from_numpy(src_X).float().to(device)
src_y = torch.from_numpy(src_y).float().to(device)
tgt_X = torch.from_numpy(tgt_X).float().to(device)
tgt_y = torch.from_numpy(tgt_y).float().to(device)
val_X = torch.from_numpy(val_X).float().to(device)
val_y = torch.from_numpy(val_y).float().to(device)


# src_X = torch.cat((src_X, torch.ones((src_X.shape[0], 1))), dim=1)
# tgt_X = torch.cat((tgt_X, torch.ones((tgt_X.shape[0], 1))), dim=1)

#################################################################

with torch.no_grad():
    print(src_X.shape, tgt_X.shape)

    cov = src_X.T @ src_X
    cov_tilde = tgt_X.T @ tgt_X
    src_rank = torch.linalg.matrix_rank(cov, hermitian=True)
    tgt_rank = torch.linalg.matrix_rank(cov_tilde, hermitian=True)
    print('ranks:', src_rank, tgt_rank)

    eigvals, eigvecs = torch.linalg.eigh(cov)
    idx = eigvals.argsort(descending=True)
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]
    s = torch.abs(eigvals) ** 0.5
    v = eigvecs
    vt = eigvecs.T

    eigvals, eigvecs = torch.linalg.eigh(cov_tilde)
    idx = eigvals.argsort(descending=True)
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]
    stilde = torch.abs(eigvals) ** 0.5
    vtilde = eigvecs
    vtildet = eigvecs.T


for config in configs:
    print(config)
    # if config['lamda'] > 0 and (config['k'] > src_rank or config['ktilde'] > tgt_rank):
    #     print(f"{config['k']}, {config['ktilde']} over the rank {src_rank}, {tgt_rank}. Skipping.")
    #     continue
    np.random.seed(seed)
    torch.manual_seed(seed)

    print('s0:', s[0])

    if loss == 'mse':
        config['lr'] = (1./(2*s[0])).item()
    else:
        config['lr'] = 1e-2

    model = nn.Sequential(
        nn.Linear(src_X.shape[1], 1, bias=True))
    with torch.no_grad():
        model[-1].weight *= 0.

    model = model.float().to(device)

    opt = optim.SGD(
                [{"params": model.parameters(), "lr_mult": 1.0}],
                lr=config['lr'], momentum=0.9, nesterov=True)

    # k = 20
    # ktilde = k
    k = config['k']
    ktilde = config['ktilde']
    if config['lamda'] > 0:
        bottom_dirs_tgt = vtildet[config['ktilde']:] * s[config['ktilde']:][:, None]
        bottom_dirs_src = vt[config['k']:] * s[config['k']:][:, None]   

    if loss == 'mse':
        loss_sup = torch.nn.MSELoss()
    else:
        loss = torch.nn.BCEWithLogitsLoss()
    train_curve = []
    test_curve = []
    val_curve = []
    train_f1_curve = []
    test_f1_curve = []
    val_f1_curve = []
    print('+ Starting training...')
    for epoch in range(config['n_epochs']):
        # print('epoch', epoch)
        
        model.train()

        # MSE Loss
        if loss == 'mse':
            weights = model[-1].weight.to(device)[0]
            task_loss = 0.0
            
            yhat = model(src_X)[:, 0]

            task_loss = ((yhat - src_y) ** 2).sum()

            if config['lamda'] > 0:
                weights = model[-1].weight[0]

                if config['remove_implicit']:
                    wv_bottom = bottom_dirs_src @ weights
                    task_loss -= (wv_bottom ** 2).sum()

                wv_bottom = bottom_dirs_tgt @ weights
                task_loss += config['lamda'] * (wv_bottom ** 2).sum()                    

            task_loss /= (src_y.shape[0])
        # CE Loss
        else:
            task_loss = loss(model(src_X), src_y.unsqueeze(dim=1))

        opt.zero_grad()
        task_loss.backward()
        opt.step()

        if epoch == config['n_epochs'] - 1:
            with torch.no_grad():
                model.eval()

                # mse
                # yhat = model(src_X)[:, 0]
                # loss = loss_sup(yhat, src_y)
                # train_curve.append(loss.detach().cpu().numpy())

                # yhat = model(tgt_X)[:, 0]
                # loss = loss_sup(yhat, tgt_y)
                # test_curve.append(loss.detach().cpu().numpy())

                # acc
                yhat = model(src_X)[:, 0] 
                if loss == 'ce':
                    yhat = yhat * 2 - 1.
                if torch.isnan(yhat).any():
                    acc = -torch.inf
                    f1 = [-np.inf for average in f1averages]
                else:
                    yhat = yhat > 0
                    acc = (1. * (yhat == (src_y > 0))).mean().item()
                    f1 = [f1_score(1*(src_y.cpu() > 0), yhat.detach().cpu().numpy(), 
                          average=average) for average in f1averages]
                train_curve.append(acc)
                train_f1_curve.append(f1)

                yhat = model(tgt_X)[:, 0]
                if loss == 'ce':
                    yhat = yhat * 2 - 1.
                if torch.isnan(yhat).any():
                    acc = -torch.inf
                    f1 = [-np.inf for average in f1averages]
                else:
                    yhat = yhat > 0
                    acc = (1. * (yhat == (tgt_y > 0))).mean().item()
                    f1 = [f1_score(1*(tgt_y.cpu() > 0), yhat.detach().cpu().numpy(), 
                          average=average) for average in f1averages]
                test_curve.append(acc)
                test_f1_curve.append(f1)

                yhat = model(val_X)[:, 0]
                if loss == 'ce':
                    yhat = yhat * 2 - 1.
                if torch.isnan(yhat).any():
                    acc = -torch.inf
                    f1 = [-np.inf for average in f1averages]
                else:
                    yhat = yhat > 0
                    acc = (1. * (yhat == (val_y > 0))).mean().item()
                    f1 = [f1_score(1*(val_y.cpu() > 0), yhat.detach().cpu().numpy(), 
                          average=average) for average in f1averages]
                val_curve.append(acc)
                val_f1_curve.append(f1)

                # if epoch == config['n_epochs'] - 1:
                #     print('---> mean of predictions:', (1.*yhat).mean())

                # print(f"epoch: {epoch}, train loss: {train_curve[-1]}, test loss: {test_curve[-1]}")

    plt.clf()
    plt.plot(train_curve, label='train')
    plt.plot(test_curve, label='test')
    plt.title(config)
    plt.legend()
    plt.savefig('acc.png')
    plt.clf()
    plt.plot([x[0] for x in train_f1_curve], label='train')
    plt.plot([x[0] for x in test_f1_curve], label='test')
    plt.title(config)
    plt.legend()
    plt.savefig('f1.png')
    print('train acc:', train_curve[-1])
    print('test acc:', test_curve[-1])
    print('val acc:', val_curve[-1])
    print('train f1:', train_f1_curve[-1])
    print('test f1:', test_f1_curve[-1])
    print('val f1:', val_f1_curve[-1])

    results['src_acc'][config_str(config)].append(train_curve[-1])
    results['tgt_acc'][config_str(config)].append(test_curve[-1])
    results['val_acc'][config_str(config)].append(val_curve[-1])
    results['src_f1'][config_str(config)].append(train_f1_curve[-1])
    results['tgt_f1'][config_str(config)].append(test_f1_curve[-1])
    results['val_f1'][config_str(config)].append(val_f1_curve[-1])


print(results)

# for key in results.keys():
#     print(key)
#     for config in configs:
#         print(config, np.array(results[key][config_str(config)]).mean())

print('saving results')
with open(os.path.join('results_new_alg', f"{src_lang}_{tgt_lang}_{seed}.json"), 'w') as f:
    json.dump(results, f)