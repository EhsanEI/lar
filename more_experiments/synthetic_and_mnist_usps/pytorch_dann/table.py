import numpy as np
import json
import os
from tabulate import tabulate
from itertools import combinations


def config_str(config):
    return f"{config['hidden_dim']}"


hidden_dims = [128, 256, 512, 1024]
configs = [{'hidden_dim': hidden_dim} for hidden_dim in hidden_dims]
tasks = list(combinations(range(10), 2))
keys = ['source_acc', 'target_acc', 'valid_acc', 'domain_acc']

meta_datas = []

meta_datas.append({
    'source_dataset': 'usps',
    'target_dataset': 'mnist',
    'ratio': None
})

meta_datas.append({
    'source_dataset': 'mnist',
    'target_dataset': 'usps',
    'ratio': None
})

meta_datas.append({
    'source_dataset': 'mnist',
    'target_dataset': 'usps',
    'ratio': .3
})

meta_datas.append({
    'source_dataset': 'mnist',
    'target_dataset': 'usps',
    'ratio': .2
})

meta_datas.append({
    'source_dataset': 'mnist',
    'target_dataset': 'usps',
    'ratio': .1
})

headers = ['']
table = [['No Adaptation'], ['DANN']]

for meta_data in meta_datas:
    header = f"{meta_data['source_dataset']}"
    if meta_data['ratio'] is not None:
        header += f"({meta_data['ratio']})"
    header += f" -> {meta_data['target_dataset']}"
    headers.append(header)
    print(header)

    results = {}
    for task in tasks:
        results[str(task)] = {}
        for alg in ['erm', 'dann']:
            results[str(task)][alg] = {}
            for key in keys:
                results[str(task)][alg][key] = {}

    for hidden_dim in hidden_dims:
        config = {'hidden_dim': hidden_dim}

        with open(os.path.join('results', f"{meta_data['source_dataset']}_{meta_data['target_dataset']}_{meta_data['ratio']}_{config_str(config)}_results.json"), 'r') as f:
            config_results = json.load(f)
    
        for task in tasks:
            for alg in ['erm', 'dann']:
                for key in keys:
                    results[str(task)][alg][key][config_str(config)] = config_results[str(task)][alg][key][-1]           

    best_no_adaptation = 0
    best_dann = 0
    accs_no_adaptation = []  # list over tasks
    accs_dann = []  # list over tasks
    for task in results.keys():

        best_val_acc = -np.inf
        best_config = None
        for config in configs:
            val_acc = results[task]['erm']['valid_acc'][config_str(config)]
            if val_acc >= best_val_acc:
                best_val_acc = val_acc
                best_config = config

        acc_no_adaptation = results[task]['erm']['target_acc'][config_str(best_config)]
        accs_no_adaptation.append(acc_no_adaptation)

        best_val_acc = -np.inf
        best_config = None
        for config in configs:
            val_acc = results[task]['dann']['valid_acc'][config_str(config)]
            if val_acc >= best_val_acc:
                best_val_acc = val_acc
                best_config = config

        acc_dann = results[task]['dann']['target_acc'][config_str(best_config)]
        accs_dann.append(acc_dann)

        if acc_dann > acc_no_adaptation:
            best_dann += 1
        if acc_dann < acc_no_adaptation:
            best_no_adaptation += 1
        else:
            pass

        print(task, acc_no_adaptation, acc_dann)
    avg_dann = np.mean(accs_dann)
    avg_no_adaptation = np.mean(accs_no_adaptation)
    # print(best_no_adaptation, best_dann)
    # print(avg_no_adaptation, avg_dann)

    table[0].append(f"{avg_no_adaptation:.2f}")
    table[1].append(f"{avg_dann:.2f}")

print(tabulate(table, headers, tablefmt="simple"))