import torch
from train import source_only, dann
import synthetic
import shallow_model_synthetic as model
from dannutils import get_free_gpu
import params
from itertools import combinations
import fire
import json
import numpy as np
import os


def main(ratio=None, hidden_dim=64):
    meta_data = {
            'data_cnt': 1000,
            'val_cnt': 100,
            'ratio': ratio,
    }
    experiment_name = f"synthetic_{meta_data['ratio']}_{hidden_dim}"
    print('******************************************************** experiment:', experiment_name)
    num_runs = 10

    keys = ['source_acc', 'target_acc', 'valid_acc', 'domain_acc']
    results = {}
    results = {}
    results['erm'] = {
        key: [] for key in keys
    }
    results['dann'] = {
        key: [] for key in keys
    }

    for run in range(num_runs):
        print('---------------------------------------- run:', run)
        
        source_train_loader, _, source_test_loader = synthetic.create_loaders(meta_data, target=False)
        target_train_loader, target_valid_loader, target_test_loader = synthetic.create_loaders(meta_data, target=True)
    
        if torch.cuda.is_available():
            # get_free_gpu()
            print('Running GPU : {}'.format(torch.cuda.current_device()))
            
            np.random.seed(0)
            torch.manual_seed(0)
            encoder = model.Extractor(hidden_dim=hidden_dim).cuda().double()
            classifier = model.Classifier(hidden_dim=hidden_dim).cuda().double()
            discriminator = model.Discriminator(hidden_dim=hidden_dim).cuda().double()
            source_only(encoder, classifier, source_train_loader, target_train_loader, 
                        source_test_loader, target_test_loader, target_valid_loader, 'erm', results['erm'])

            np.random.seed(0)
            torch.manual_seed(0)
            encoder = model.Extractor(hidden_dim=hidden_dim).cuda().double()
            classifier = model.Classifier(hidden_dim=hidden_dim).cuda().double()
            discriminator = model.Discriminator(hidden_dim=hidden_dim).cuda().double()
            dann(encoder, classifier, discriminator, source_train_loader, target_train_loader, 
                 source_test_loader, target_test_loader, target_valid_loader, 'dann', results['dann'])

        else:
            print("There is no GPU -_-!")
            break

    print(results)
    with open(os.path.join('results', f"{experiment_name}_results.json"), 'w') as f:
        json.dump(results, f)


if __name__ == "__main__":
    fire.Fire(main)
