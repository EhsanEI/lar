import numpy as np
import torch
import torch.nn as nn
from itertools import product
import os
import matplotlib.pyplot as plt
from data_utils import get_data_synthetic_clf, add_bias_feature
from sklearn.utils import shuffle
from sklearn.linear_model import LinearRegression, LogisticRegression
import json


def config_str(config):
    return f"{config['l2reg']}_{config['lamda']}_{config['k']}_{config['ktilde']}_{config['remove_implicit_reg']}"


plot_curves = False
plot_figs = True
save = False
device = torch.device(f'cuda:0') if torch.cuda.is_available() else torch.device('cpu')

# Experiment setting (should get from bash later)
meta_data = {
    'data_cnt': 1000,
    'val_cnt': 100,
    'num_runs': 1,#10,
    'num_epoch': 1,#20000,
    'ratio': None,  # subsampling first class
    'add_bias_feature': False,
}

# Hyperparameter grid values
hyperparams = {
    'remove_implicit_reg': [True],
    'l2reg': [0, 1e-0, 1e+1, 1e+2, 1e+3],
    'lamda': [0, 1e-0, 1e+1, 1e+2, 1e+3],
    'k': [1],
    'ktilde': [1],
}

# Making the grid

# Use this line if you want to compare all combinations of hyperparameters,
# including combination of l2 norm regularizer and our regularizer
# configs = list(dict(zip(hyperparams.keys(), x)) 
#                for x in product(*hyperparams.values()))

# Use this one if you want to compare our regularizer with the l2 norm regularizer
configs = []
for l2reg in hyperparams['l2reg']:
    config = {
        'l2reg': l2reg,
        'lamda': 0,
        'k': 1,  # Does not matter
        'ktilde': 1,  # Does not matter
        'remove_implicit_reg': False,
    }
    configs.append(config)
# for k, ktilde, lamda, remove_implicit_reg in product(
#         hyperparams['k'], hyperparams['ktilde'], hyperparams['lamda'][1:], hyperparams['remove_implicit_reg']):
#     config = {
#         'l2reg': 0,
#         'lamda': lamda,
#         'k': k,
#         'ktilde': ktilde,
#         'remove_implicit_reg': remove_implicit_reg
#     }
#     configs.append(config)

# print(configs)

# Results to save
result_keys = ['train_acc', 'val_acc', 'test_acc', 'weights', 'wstar', 'wdiff']
results = {key: {config_str(config): [] for config in configs}
           for key in result_keys}

# Running the experiment
for run in range(meta_data['num_runs']):
    print('***********')
    print('run:', run)
    np.random.seed(run)
    torch.manual_seed(run)

    X_src, y_src, X_tgt, y_tgt, X_val, y_val = get_data_synthetic_clf(meta_data, seed=run)
    print('src:', X_src.shape, y_src.shape)
    print('tgt:', X_val.shape, y_val.shape)
    print('val:', X_val.shape, y_tgt.shape)

    X_all = np.vstack((X_src, X_tgt))
    y_all = np.hstack((y_src, y_tgt))
    print(X_all.shape, y_all.shape, y_all[:5])
    logreg = LogisticRegression(penalty='none').fit(X_all, y_all)
    print('joint accuracy:', logreg.score(X_all, y_all))

    y_src = (y_src - 0.5) * 2
    y_tgt = (y_tgt - 0.5) * 2

    if meta_data['add_bias_feature']:
        # Adding a constant feature to representation instead of bias unit
        X_src = add_bias_feature(X_src)
        X_val = add_bias_feature(X_val)
        X_tgt = add_bias_feature(X_tgt)

    # Turning into torch tensors
    X_src_ten = torch.from_numpy(X_src).float().to(device)
    y_src_ten = torch.from_numpy(y_src).float().to(device)
    X_tgt_ten = torch.from_numpy(X_tgt).float().to(device)
    y_tgt_ten = torch.from_numpy(y_tgt).float().to(device)
    X_val_ten = torch.from_numpy(X_val).float().to(device)
    y_val_ten = torch.from_numpy(y_val).float().to(device)

    cov = X_src_ten.T @ X_src_ten
    cov_tilde = X_tgt_ten.T @ X_tgt_ten
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
    print('singular values:', s)

    eigvals, eigvecs = torch.linalg.eigh(cov_tilde)
    idx = eigvals.argsort(descending=True)
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]
    stilde = torch.abs(eigvals) ** 0.5
    vtilde = eigvecs
    vtildet = eigvecs.T

    for config in configs:
        print('config:', config)

        np.random.seed(run)
        torch.manual_seed(run)

        if config['lamda'] > 0:
            bottom_dirs_tgt = vtildet[config['ktilde']:] * s[config['ktilde']:][:, None]
            bottom_dirs_src = vt[config['k']:] * s[config['k']:][:, None]   

        model = nn.Sequential(
                nn.Linear(X_src_ten.shape[1], 1, bias=False)).to(device)
        with torch.no_grad():
            model[-1].weight *= 0.
        opt = torch.optim.SGD(model.parameters(),
                              lr=1e-4)#(1./(100*s[0])).item())

        # loss_sup = torch.nn.MSELoss()

        train_curve = []
        test_curve = []
        val_curve = []
        losses = []
        for epoch in range(meta_data['num_epoch']):
            # print('epoch', epoch, end='\r', flush=True)

            model.train()

            yhat = model(X_src_ten)[:, 0]

            task_loss = ((yhat - y_src_ten) ** 2).sum()

            if config['lamda'] > 0:
                weights = model[-1].weight[0]

                if config['remove_implicit_reg']:
                    wv_bottom = bottom_dirs_src @ weights
                    task_loss -= (wv_bottom ** 2).sum()

                wv_bottom = bottom_dirs_tgt @ weights
                task_loss += config['lamda'] * (wv_bottom ** 2).sum()   

            if config['l2reg'] > 0:
                weights = model[-1].weight[0]
                task_loss += config['l2reg'] * (weights ** 2).sum()

            task_loss /= (y_src_ten.shape[0])

            losses.append(task_loss.item())
    
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
                    yhat = model(X_src_ten)[:, 0] 
                    if torch.isnan(yhat).any():
                        acc = -torch.inf
                    else:
                        yhat = yhat > 0
                        acc = (1. * (yhat == (y_src_ten > 0))).mean().item()
                    train_curve.append(acc)

                    yhat = model(X_tgt_ten)[:, 0]
                    if torch.isnan(yhat).any():
                        acc = -torch.inf
                    else:
                        yhat = yhat > 0
                        acc = (1. * (yhat == (y_tgt_ten > 0))).mean().item()
                    test_curve.append(acc)

                    yhat = model(X_val_ten)[:, 0]
                    if torch.isnan(yhat).any():
                        acc = -torch.inf
                    else:
                        yhat = yhat > 0
                        acc = (1. * (yhat == (y_val_ten > 0))).mean().item()
                    val_curve.append(acc)

        # plt.clf()
        # plt.plot(losses)
        # plt.show()
        results['train_acc'][config_str(config)].append(train_curve[-1])
        results['test_acc'][config_str(config)].append(test_curve[-1])
        results['val_acc'][config_str(config)].append(val_curve[-1])
        print(train_curve[-1], test_curve[-1], val_curve[-1])

        # results['weights'][config_str(config)].append(
        #     model[0].weight.cpu().detach().numpy())

        w_np = model[0].weight.cpu().detach().numpy()
        linreg = LinearRegression(fit_intercept=False)
        linreg = linreg.fit(X_tgt, y_tgt)
        wstar = linreg.coef_

        # results['wstar'][config_str(config)].append(wstar)

        wdiff = ((w_np - wstar) ** 2).sum() ** 0.5
        results['wdiff'][config_str(config)].append(wdiff)
        print('wdiff:', wdiff)

        if plot_curves:
            plt.clf()
            plt.plot(train_curve, label='train')
            plt.plot(test_curve, label='test')
            plt.legend()
            plt.show()

        if plot_figs and run == 0:
            plt_dir = os.path.join('joint_risk')
            # if not os.path.exists(plt_dir):
            #     os.makedirs(plt_dir)

            for (X, y, title) in [(X_src, y_src, 'labeled'), 
                                  (X_tgt, y_tgt, 'test')]:
                cov_mat = X.T @ X
                eigvals, eigvecs = np.linalg.eigh(cov_mat)
                idx = eigvals.argsort()[::-1]
                eigvals = eigvals[idx]
                eigvecs = eigvecs[:, idx]
                plt.figure(figsize=(5, 5))
                # fig, ax = plt.subplots()
                plt.axvline(0, color='black')
                plt.axhline(0, color='black')

                plt.scatter(X[:, 0], X[:, 1], marker='o', c=y, cmap='coolwarm', alpha=.5, edgecolors='none')

                if config_str(config) == config_str(configs[0]):
                    plt.arrow(0, 0, eigvecs[0, 0], eigvecs[1, 0], color='black', width=.08)
                    plt.arrow(0, 0, eigvecs[0, 1], eigvecs[1, 1], color='black', width=.08)
                    no_adapt_w = model[0].weight.cpu().detach().numpy()
                else:
                    sep_x = [-4, 4]
                    sep_y = [-w_np[0, 0] / w_np[0, 1] * elem for elem in sep_x]
                    plt.plot(sep_x, sep_y, color='darkgreen', linewidth=3, label='Regularized')
                    sep_y = [-no_adapt_w[0, 0] / no_adapt_w[0, 1] * elem for elem in sep_x]
                    plt.plot(sep_x, sep_y, color='darkgreen', linestyle='dashed', linewidth=3, label='No Adaptation')
                    plt.legend()
                    # plt.arrow(0, 0, w_np[0, 0], w_np[0, 1], color='purple', width=.06)
                    # plt.arrow(0, 0, wstar[0], wstar[1], color='black', width=.06)

                plt.xlim(-4, 4)
                plt.ylim(-4, 4)

                plt.axis('off')
                plt.gca().set_aspect('equal')

                # plt.title(title + '\n' + config_str(config))

                plt.tight_layout(h_pad=0, w_pad=0)
                # plt.show()

                plt.savefig(os.path.join(plt_dir, 
                            f'{config_str(config)}_{title}.png'))
            # plt.close(fig)
                print(os.path.join(plt_dir, 
                            f'{config_str(config)}_{title}.png'))


if save:
    print('saving')
    with open(os.path.join('results', f"synthetic_{meta_data['ratio']}_results.json"), 'w') as f:
        json.dump(results, f)

    with open(os.path.join('results', f"synthetic_{meta_data['ratio']}_configs.json"), 'w') as f:
        json.dump(configs, f)
