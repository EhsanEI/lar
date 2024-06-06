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


with open(os.path.join('results', f"synthetic_None_results.json"), 'r') as f:
    results = json.load(f)

with open(os.path.join('results', f"synthetic_None_configs.json"), 'r') as f:
    configs = json.load(f)

lamdas = []
ours_mean = []
ours_std = []
l2regs = []
l2reg_mean = []
l2reg_std = []


for config in configs:
    this_result = [100*acc for acc in results['test_acc'][config_str(config)]]
    mean = np.mean(this_result)
    std = scipy.stats.sem(this_result)
    print(config, mean, std)
    if config['l2reg'] == 0:
        lamdas.append(config['lamda'])
        ours_mean.append(mean)
        ours_std.append(std)
    if config['lamda'] == 0:
        l2regs.append(config['l2reg'])
        l2reg_mean.append(mean)
        l2reg_std.append(std)


print(ours_mean)
print(l2reg_mean)

plt.clf()
plt.plot(lamdas, ours_mean, linewidth=linewidth, color='limegreen', marker='o', label='2k epochs')
plt.plot(l2regs, l2reg_mean, linewidth=linewidth, color='darkorange', marker='o', label='$\ell_2$')
plt.fill_between(
    lamdas, np.array(ours_mean) - np.array(ours_std), np.array(ours_mean) + np.array(ours_std), color='limegreen', alpha=alpha)
plt.fill_between(
    l2regs, np.array(l2reg_mean) - np.array(l2reg_std), np.array(l2reg_mean) + np.array(l2reg_std), color='darkorange', alpha=alpha)
plt.xscale('log')
plt.xlabel('Regularization Weight')
plt.ylabel('Target Domain Accuracy')
plt.xlim(xmin=1, xmax=1e3)
plt.legend(prop={'size': 14})
plt.tight_layout()
plt.savefig('synthetic.png')