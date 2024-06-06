import matplotlib.pyplot as plt
import json
import os
import numpy as np
import scipy.stats

plt.rcParams.update({'font.size': 22})
linewidth = 3
alpha = .3


def config_str(config):
    return f"{config['l2reg']}_{config['lamda']}_{config['k']}_{config['ktilde']}_{config['remove_implicit_reg']}"


folder = 'results_reg'
with open(os.path.join(folder, f"synthetic_None_results.json"), 'r') as f:
    results = json.load(f)

with open(os.path.join(folder, f"synthetic_None_configs.json"), 'r') as f:
    configs = json.load(f)


lamdas = []
ours_mean = []
ours_std = []
noimp_mean = []
noimp_std = []
l2regs = []
l2reg_mean = []
l2reg_std = []


for config in configs:
    this_result = [wdiff for wdiff in results['wdiff'][config_str(config)]]
    mean = np.mean(this_result)
    std = scipy.stats.sem(this_result)
    print(config, mean, std)
    if config['l2reg'] == 0 and config['remove_implicit_reg']:
        # lamdas.append(config['lamda'])
        ours_mean.append(mean)
        ours_std.append(std)
    if config['l2reg'] == 0 and not config['remove_implicit_reg']:
        lamdas.append(config['lamda'])
        noimp_mean.append(mean)
        noimp_std.append(std)
    if config['lamda'] == 0:
        l2regs.append(config['l2reg'])
        l2reg_mean.append(mean)
        l2reg_std.append(std)

ours_mean.insert(0, noimp_mean[0])
ours_std.insert(0, noimp_std[0])
print(lamdas)
print(ours_mean)
print(l2reg_mean)
print(noimp_mean)

plt.clf()
plt.plot(lamdas, ours_mean, linewidth=linewidth, color='darkgreen', marker='o', label='Label Alignment')
# plt.plot(lamdas, noimp_mean, linewidth=linewidth, color='indianred', marker='o', label='Ablation')
plt.plot(l2regs, l2reg_mean, linewidth=linewidth, color='darkorange', marker='o', label='$\ell_2$')
plt.fill_between(
    lamdas, np.array(ours_mean) - np.array(ours_std), np.array(ours_mean) + np.array(ours_std), color='darkgreen', alpha=alpha)
# plt.fill_between(
#     lamdas, np.array(noimp_mean) - np.array(noimp_std), np.array(noimp_mean) + np.array(noimp_std), color='indianred', alpha=alpha)
plt.fill_between(
    l2regs, np.array(l2reg_mean) - np.array(l2reg_std), np.array(l2reg_mean) + np.array(l2reg_std), color='darkorange', alpha=alpha)
plt.xscale('log')
plt.xlabel('Regularization Weight')
plt.ylabel('$\|\widehat{w^*} - w^*_\mathcal{T}\|$')
# plt.legend()
plt.tight_layout()
plt.savefig('wdiff_reg.png')