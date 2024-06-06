import torch
from train import source_only, dann
import mnist
import usps
import shallow_model as model
from dannutils import get_free_gpu
import params
from itertools import combinations
import fire
import json
import numpy as np
import os
import pickle


def extract(encoder, loader):
    with torch.no_grad():
        rep = []
        y = []
        for batch_X, batch_y in loader:
            rep.append(encoder(batch_X.cuda()).detach().cpu().numpy())
            y.append(batch_y.numpy())
        rep = np.vstack(rep)
        y = np.hstack(y)
    return rep, y


def main(source_domain='mnist', target_domain='usps', ratio=None, hidden_dim=512):
    experiment_name = f"{source_domain}_{target_domain}_{ratio}_{hidden_dim}"
    print('******************************************************** experiment:', experiment_name)
    tasks = list(combinations(range(10), 2))
    # tasks = [(0, 1), (0, 2)]

    keys = ['source_acc', 'target_acc', 'valid_acc', 'domain_acc']
    results = {}

    source_reps = {}
    source_labels = {}
    target_reps = {}
    target_labels = {}

    for task in tasks:
        print('---------------------------------------- task:', task)
        
        results[str(task)] = {}
        results[str(task)]['erm'] = {
            key: [] for key in keys
        }
        results[str(task)]['dann'] = {
            key: [] for key in keys
        }

        source_reps[str(task)] = {}
        source_labels[str(task)] = {}
        target_reps[str(task)] = {}
        target_labels[str(task)] = {}

        if source_domain == 'usps':
            assert ratio is None
            source_train_loader, _, source_test_loader = usps.create_loaders(task)
        elif source_domain == 'mnist':
            source_train_loader, _, source_test_loader = mnist.create_loaders(task, ratio)
        else:
            raise NotImplementedError

        if target_domain == 'usps':
            target_train_loader, target_valid_loader, target_test_loader = usps.create_loaders(task, validation=True)
        elif target_domain == 'mnist':
            target_train_loader, target_valid_loader, target_test_loader = mnist.create_loaders(task, ratio=None, validation=True)
        else:
            raise NotImplementedError

        if torch.cuda.is_available():
            # get_free_gpu()
            print('Running GPU : {}'.format(torch.cuda.current_device()))
            
            np.random.seed(0)
            torch.manual_seed(0)
            encoder = model.Extractor(hidden_dim=hidden_dim).cuda()
            classifier = model.Classifier(hidden_dim=hidden_dim).cuda()
            discriminator = model.Discriminator(hidden_dim=hidden_dim).cuda()
            source_only(encoder, classifier, source_train_loader, target_train_loader, 
                        source_test_loader, target_test_loader, target_valid_loader, 'erm', results[str(task)]['erm'])

            source_reps[str(task)]['erm'], source_labels[str(task)]['erm'] = extract(encoder, source_train_loader)
            target_reps[str(task)]['erm'], target_labels[str(task)]['erm'] = extract(encoder, target_test_loader)

            np.random.seed(0)
            torch.manual_seed(0)
            encoder = model.Extractor(hidden_dim=hidden_dim).cuda()
            classifier = model.Classifier(hidden_dim=hidden_dim).cuda()
            discriminator = model.Discriminator(hidden_dim=hidden_dim).cuda()
            dann(encoder, classifier, discriminator, source_train_loader, target_train_loader, 
                 source_test_loader, target_test_loader, target_valid_loader, 'dann', results[str(task)]['dann'])

            source_reps[str(task)]['dann'], source_labels[str(task)]['dann'] = extract(encoder, source_train_loader)
            target_reps[str(task)]['dann'], target_labels[str(task)]['dann'] = extract(encoder, target_test_loader)

        else:
            print("There is no GPU -_-!")
            break

    print(results)
    with open(os.path.join('results', f"{experiment_name}_results.json"), 'w') as f:
        json.dump(results, f)

    with open(os.path.join('results', f"{experiment_name}_source_reps.pkl"), 'wb') as f:
        pickle.dump(source_reps, f)
    with open(os.path.join('results', f"{experiment_name}_source_labels.pkl"), 'wb') as f:
        pickle.dump(source_labels, f)
    with open(os.path.join('results', f"{experiment_name}_target_reps.pkl"), 'wb') as f:
        pickle.dump(target_reps, f)
    with open(os.path.join('results', f"{experiment_name}_target_labels.pkl"), 'wb') as f:
        pickle.dump(target_labels, f)


if __name__ == "__main__":
    fire.Fire(main)
