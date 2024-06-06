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
    return f"{config['map_beta']}_{config['dis_smooth']}_{config['proc_iter']}"


device = torch.device(f'cuda:0') if torch.cuda.is_available() else torch.device('cpu')
loss = 'ce'


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
map_betas = [0.001, 0.01, 0.1]
dis_smooths = [0.1]
proc_iters = [0, 1, 2]

f1averages = ['binary', 'macro', 'weighted']

hyperparams = {
    'map_betas': map_betas,
    'dis_smooths': dis_smooths,
    'proc_iters': proc_iters,
}
print('saving hyperparams')
with open(os.path.join('results', f"hyperparams.json"), 'w') as f:
    json.dump(hyperparams, f)


configs = []


for map_beta, dis_smooth, proc_iter in product(map_betas, dis_smooths, proc_iters):
    config = {
        'map_beta': map_beta,
        'dis_smooth': dis_smooth,
        'proc_iter': proc_iter,
        'lamda': 0.0,
        'remove_implicit': False,
        'k': 0,
        'ktilde': 0,
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

with open(os.path.join('results', f"configs.json"), 'w') as f:
    json.dump(configs, f)


for config in configs:
    print(config)
    np.random.seed(seed)
    torch.manual_seed(seed)

    folder = f"/scratch/muse_dumped/{src_lang}_{tgt_lang}_{config['map_beta']}_{config['dis_smooth']}_{seed}/{src_lang}_{tgt_lang}"
    src_X = np.load(f"{folder}/vectors-{src_lang}_{config['proc_iter']}.npy")
    tgt_X = np.load(f"{folder}/vectors-{tgt_lang}_{config['proc_iter']}.npy")

    src_y = np.load(f'embeddings/bert_{src_lang}_y.npy')
    tgt_y = np.load(f'embeddings/bert_{tgt_lang}_y.npy')

    print(src_X.shape, src_y.shape, tgt_X.shape, tgt_y.shape, src_y.mean(), tgt_y.mean())

    if loss == 'mse':
        src_y = src_y * 2. - 1
        tgt_y = tgt_y * 2. - 1

    tgt_X, val_X, tgt_y, val_y = train_test_split(tgt_X, tgt_y, test_size=100, random_state=seed)

    src_X = torch.from_numpy(src_X).float().to(device)
    src_y = torch.from_numpy(src_y).float().to(device)
    tgt_X = torch.from_numpy(tgt_X).float().to(device)
    tgt_y = torch.from_numpy(tgt_y).float().to(device)
    val_X = torch.from_numpy(val_X).float().to(device)
    val_y = torch.from_numpy(val_y).float().to(device)

    with torch.no_grad():
        print(src_X.shape, tgt_X.shape)

        u, s, vt = torch.linalg.svd(src_X)
        v = vt.T
        utilde, stilde, vtildet = torch.linalg.svd(tgt_X)
        vtilde = vtildet.T

    print('s0:', s[0])

    config['lr'] = (1./(2*s[0])).item()

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
            
            if config['remove_implicit']:
                remove_imp_k = k
            else:
                remove_imp_k = src_X.shape[1]
            yu = u[:, :remove_imp_k].T @ src_y
            wv = vt[:remove_imp_k, :] @ weights
            wvs = wv * s[:remove_imp_k]
            task_loss += torch.norm(wvs - yu) ** 2

            wvtilde = vtildet[ktilde:stilde.detach().shape[0], :] @ weights
            wvstilde = wvtilde * stilde[ktilde:]
            task_loss += config['lamda'] * torch.norm(wvstilde) ** 2

            # normalizing the regularized loss by batch_size
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

    # plt.clf()
    # plt.plot(train_curve, label='train')
    # plt.plot(test_curve, label='test')
    # plt.title(config)
    # plt.legend()
    # plt.show()
    # plt.clf()
    # plt.plot([x[0] for x in train_f1_curve], label='train')
    # plt.plot([x[0] for x in test_f1_curve], label='test')
    # plt.title(config)
    # plt.legend()
    # plt.show()
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

for key in results.keys():
    print(key)
    for config in configs:
        print(config, np.array(results[key][config_str(config)]).mean())

print('saving results')
with open(os.path.join('results', f"{src_lang}_{tgt_lang}_{seed}.json"), 'w') as f:
    json.dump(results, f)