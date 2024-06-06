import numpy as np
import torch
import os
import sys
import matplotlib.pyplot as plt
from data_utils import load_data, add_bias_feature,\
 flatten_X
import torch.nn as nn
from itertools import product
from sklearn.model_selection import train_test_split
import json
from itertools import combinations


def config_str(config):
    return f"{config['lamda']}_{config['k']}_{config['ktilde']}"


device = torch.device(f'cuda:0') if torch.cuda.is_available() else torch.device('cpu')

plot_curves = False

# Experiment setting (should get from bash later)
meta_data = {
    'source_dataset': sys.argv[1],  # mnist or usps
    'target_dataset': sys.argv[2],  # mnist or usps
    'add_bias_feature': True,
    'num_runs': 1,
    'num_epoch': 500,
}

if len(sys.argv) < 4:
    meta_data['ratio'] = None
else:
    meta_data['ratio'] = float(sys.argv[3])

print(meta_data)


# Hyperparameter grid values
hyperparams = {
    'lamda': [0, 1e-1, 1e1],
    'k': [8, 16, 32, 64, 128, 256],
    'ktilde': [8, 16, 32, 64, 128, 256],
}

configs = []

# config = {
#     'lamda': 0,
#     'remove_implicit': False,
#     'k': 8,
#     'ktilde': 8,
# }
# configs.append(config)

for k, ktilde in product(hyperparams['k'], hyperparams['ktilde']):
    for lamda in hyperparams['lamda'][1:]:
        config = {
            'lamda': lamda,
            'remove_implicit': True,            
            'k': k,
            'ktilde': ktilde,
        }
        configs.append(config)

print(len(configs), 'configs')

# Results to save
result_keys = ['train_acc', 'test_acc', 'val_acc']
results = {}

# Running the experiment
# tasks = [(0, 1)]
tasks = list(combinations(range(10), 2))
# tasks = easy_tasks + hard_tasks
# tasks = list(optimal_k.keys())
for neg_class, pos_class in tasks:
    print('---------------------')
    task = (neg_class, pos_class)
    print(task)
    results[str(task)] = {
        key: {config_str(config): [] for config in configs}
        for key in result_keys}
    for run in range(meta_data['num_runs']):
        print('*********')
        print('run:', run)
        np.random.seed(run)
        torch.manual_seed(run)

        X_train_src, y_train_src, X_test_src, y_test_src = \
            load_data[meta_data['source_dataset']](task, ratio=meta_data['ratio'], seed=run)
        X_train_tgt, y_train_tgt, X_test_tgt, y_test_tgt = \
            load_data[meta_data['target_dataset']](task, ratio=None, seed=run)
        print(X_train_src.shape, y_train_src.shape, X_test_src.shape, y_test_src.shape)
        print(X_train_tgt.shape, y_train_tgt.shape, X_test_tgt.shape, y_test_tgt.shape)
        
        y_train_src = (y_train_src - 0.5) * 2
        y_test_src = (y_test_src - 0.5) * 2
        y_train_tgt = (y_train_tgt - 0.5) * 2
        y_test_tgt = (y_test_tgt - 0.5) * 2

        X_train_src = flatten_X(X_train_src)
        X_test_src = flatten_X(X_test_src)
        X_train_tgt = flatten_X(X_train_tgt)
        X_test_tgt = flatten_X(X_test_tgt)

        if meta_data['add_bias_feature']:
            # Adding a constant feature to representation instead of bias unit
            X_train_src = add_bias_feature(X_train_src)
            X_test_src = add_bias_feature(X_test_src)
            X_train_tgt = add_bias_feature(X_train_tgt)
            X_test_tgt = add_bias_feature(X_test_tgt)

        X_train_tgt, X_val, y_train_tgt, y_val = train_test_split(X_train_tgt, y_train_tgt, test_size=100, random_state=run)

        # Turning into torch tensors
        X_train_src_ten = torch.from_numpy(X_train_src).float().to(device)
        X_test_src_ten = torch.from_numpy(X_test_src).float().to(device)
        y_train_src_ten = torch.from_numpy(y_train_src).float().to(device)
        y_test_src_ten = torch.from_numpy(y_test_src).float().to(device)
        X_train_tgt_ten = torch.from_numpy(X_train_tgt).float().to(device)
        X_test_tgt_ten = torch.from_numpy(X_test_tgt).float().to(device)
        y_train_tgt_ten = torch.from_numpy(y_train_tgt).float().to(device)
        y_test_tgt_ten = torch.from_numpy(y_test_tgt).float().to(device)
        X_val_ten = torch.from_numpy(X_val).float().to(device)
        y_val_ten = torch.from_numpy(y_val).float().to(device)

        cov = X_train_src_ten.T @ X_train_src_ten
        cov_tilde = X_train_tgt_ten.T @ X_train_tgt_ten
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

        # print(v)
        # print(vtilde)
        for assumption_k in hyperparams['k']:
            # assumption = v[:, :assumption_k].T @ vtilde[:, :assumption_k]
            assumption = v.T @ vtilde
            assumption = assumption.cpu().detach().numpy()
            _, assumption_s, _ = np.linalg.svd(assumption)
            print('check:', task, assumption_k, assumption.shape, np.linalg.norm(assumption), assumption_s[-3:], np.linalg.slogdet(assumption))
            
        for config in configs:
            print('config:', config)

            if config['lamda'] > 0 and (config['k'] > src_rank or config['ktilde'] > tgt_rank):
                print(f"{config['k']}, {config['ktilde']} over the rank {src_rank}, {tgt_rank}. Skipping.")
                continue

            np.random.seed(run)
            torch.manual_seed(run)

            if config['lamda'] > 0:
                bottom_dirs_tgt = vtildet[config['ktilde']:] * s[config['ktilde']:][:, None]
                bottom_dirs_src = vt[config['k']:] * s[config['k']:][:, None]   

            model = nn.Sequential(
                nn.Linear(X_train_src_ten.shape[1], 1, bias=False)).to(device)
            with torch.no_grad():
                model[-1].weight *= 0.
            opt = torch.optim.SGD(model.parameters(),
                                  lr=(1./(100*s[0])).item())

            # loss_sup = torch.nn.MSELoss()

            train_curve = []
            test_curve = []
            val_curve = []
            for epoch in range(meta_data['num_epoch']):
                # print('epoch', epoch, end='\r', flush=True)

                model.train()

                yhat = model(X_train_src_ten)[:, 0]

                task_loss = ((yhat - y_train_src_ten) ** 2).sum()

                if config['lamda'] > 0:
                    weights = model[-1].weight[0]

                    if config['remove_implicit']:
                        wv_bottom = bottom_dirs_src @ weights
                        task_loss -= (wv_bottom ** 2).sum()

                    wv_bottom = bottom_dirs_tgt @ weights
                    task_loss += config['lamda'] * (wv_bottom ** 2).sum()                    

                task_loss /= (y_train_src_ten.shape[0])

                opt.zero_grad()
                task_loss.backward()
                opt.step()

                if plot_curves or (epoch == meta_data['num_epoch'] - 1):
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
                        yhat = model(X_train_src_ten)[:, 0] 
                        if torch.isnan(yhat).any():
                            acc = -torch.inf
                        else:
                            yhat = yhat > 0
                            acc = (1. * (yhat == (y_train_src_ten > 0))).mean().item()
                        train_curve.append(acc)

                        yhat = model(X_test_tgt_ten)[:, 0]
                        if torch.isnan(yhat).any():
                            acc = -torch.inf
                        else:
                            yhat = yhat > 0
                            acc = (1. * (yhat == (y_test_tgt_ten > 0))).mean().item()
                        test_curve.append(acc)

                        yhat = model(X_val_ten)[:, 0]
                        if torch.isnan(yhat).any():
                            acc = -torch.inf
                        else:
                            yhat = yhat > 0
                            acc = (1. * (yhat == (y_val_ten > 0))).mean().item()
                        val_curve.append(acc)

            results[str(task)]['train_acc'][config_str(config)].append(train_curve[-1])
            results[str(task)]['test_acc'][config_str(config)].append(test_curve[-1])
            results[str(task)]['val_acc'][config_str(config)].append(val_curve[-1])
            print(train_curve[-1], test_curve[-1], val_curve[-1])
            # print(train_curve)
            if plot_curves:
                plt.clf()
                plt.plot(train_curve, label='train')
                plt.plot(test_curve, label='test')
                plt.legend()
                plt.savefig('curves.png')

# print(results)
# print('saving')
# with open(os.path.join('results', f"{meta_data['source_dataset']}_{meta_data['target_dataset']}_{meta_data['ratio']}_results.json"), 'w') as f:
#     json.dump(results, f)

# with open(os.path.join('results', f"{meta_data['source_dataset']}_{meta_data['target_dataset']}_{meta_data['ratio']}_configs.json"), 'w') as f:
#     json.dump(configs, f)