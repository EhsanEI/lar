import numpy as np
import json
import os
from tabulate import tabulate


def config_str(config):
    return f"{config['lamda']}_{config['k']}_{config['ktilde']}"


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
table = [['No Adaptation'], ['Ours']]

for meta_data in meta_datas:
    header = f"{meta_data['source_dataset']}"
    if meta_data['ratio'] is not None:
        header += f"({meta_data['ratio']})"
    header += f" -> {meta_data['target_dataset']}"
    headers.append(header)
    print(header)

    with open(os.path.join('results', f"{meta_data['source_dataset']}_{meta_data['target_dataset']}_{meta_data['ratio']}_results.json"), 'r') as f:
        results = json.load(f)

    with open(os.path.join('results', f"{meta_data['source_dataset']}_{meta_data['target_dataset']}_{meta_data['ratio']}_configs.json"), 'r') as f:
        configs = json.load(f)

    assert configs[0]['lamda'] == 0
    for config in configs[1:]:
        assert config['lamda'] > 0
    # print(results)
    # print(configs)

    best_no_adaptation = 0
    best_ours = 0
    accs_no_adaptation = []
    accs_ours = []
    for task in results.keys():
        # print(task)
        acc_no_adaptation = np.array(results[task]['test_acc'][config_str(configs[0])]).mean()
        accs_no_adaptation.append(acc_no_adaptation)

        best_val_acc = -np.inf
        best_config = None
        for config in configs[1:]:
            val_acc_runs = results[task]['val_acc'][config_str(config)]
            if len(val_acc_runs) == 0:
                # print('skipping', config)
                continue
            val_acc = np.mean(val_acc_runs)
            if val_acc >= best_val_acc:
                best_val_acc = val_acc
                best_config = config

        acc_ours = np.array(results[task]['test_acc'][config_str(best_config)]).mean()
        # print(best_config, acc_ours)
        accs_ours.append(acc_ours)

        if acc_ours > acc_no_adaptation:
            best_ours += 1
        if acc_ours < acc_no_adaptation:
            best_no_adaptation += 1
        else:
            pass

        print(task, acc_no_adaptation, acc_ours)
    avg_ours = np.mean(accs_ours)
    avg_no_adaptation = np.mean(accs_no_adaptation)
    # print(best_no_adaptation, best_ours)
    # print(avg_no_adaptation, avg_ours)

    table[0].append(f"{100*avg_no_adaptation:.2f}")
    table[1].append(f"{100*avg_ours:.2f}")

print(tabulate(table, headers, tablefmt="simple"))