import torch
import numpy as np
from dannutils import set_model_mode
import params


def tester(encoder, classifier, discriminator, source_test_loader, target_test_loader, target_valid_loader, training_mode):
    print("Model test ...")

    encoder.cuda()
    classifier.cuda()
    set_model_mode('eval', [encoder, classifier])
    
    if training_mode == 'dann':
        discriminator.cuda()
        set_model_mode('eval', [discriminator])
        domain_correct = 0

    source_correct = 0
    target_correct = 0
    valid_correct = 0

    for batch_idx, source_data in enumerate(source_test_loader):
        # p = float(batch_idx) / len(source_test_loader)
        # alpha = 2. / (1. + np.exp(-10 * p)) - 1

        # 1. Source input -> Source Classification
        source_image, source_label = source_data
        source_image, source_label = source_image.cuda(), source_label.cuda()
        if params.target_domain == 'mnistm':
            source_image = torch.cat((source_image, source_image, source_image), 1)  # MNIST convert to 3 channel
        source_feature = encoder(source_image)
        source_output = classifier(source_feature)
        source_pred = source_output.data.max(1, keepdim=True)[1]
        source_correct += source_pred.eq(source_label.data.view_as(source_pred)).cpu().sum()

    for batch_idx, target_data in enumerate(target_test_loader):
        # p = float(batch_idx) / len(target_test_loader)
        # alpha = 2. / (1. + np.exp(-10 * p)) - 1

        # 2. Target input -> Target Classification
        target_image, target_label = target_data
        target_image, target_label = target_image.cuda(), target_label.cuda()
        target_feature = encoder(target_image)
        target_output = classifier(target_feature)
        target_pred = target_output.data.max(1, keepdim=True)[1]
        target_correct += target_pred.eq(target_label.data.view_as(target_pred)).cpu().sum()

    for batch_idx, target_data in enumerate(target_valid_loader):
        # p = float(batch_idx) / len(target_test_loader)
        # alpha = 2. / (1. + np.exp(-10 * p)) - 1

        # 2. Target input -> Target Classification
        target_image, target_label = target_data
        target_image, target_label = target_image.cuda(), target_label.cuda()
        target_feature = encoder(target_image)
        target_output = classifier(target_feature)
        target_pred = target_output.data.max(1, keepdim=True)[1]
        valid_correct += target_pred.eq(target_label.data.view_as(target_pred)).cpu().sum()
            
    if training_mode == 'dann':
        domain_data_count = 0
        target_test_iter = iter(target_test_loader)
        # 3. Combined input -> Domain Classificaion
        for batch_idx, source_data in enumerate(source_test_loader):
            try:
                target_data = next(target_test_iter)
            except StopIteration:
                target_test_iter = iter(target_test_loader)
                target_data = next(target_test_iter)
            p = float(batch_idx) / len(source_test_loader)
            alpha = 2. / (1. + np.exp(-10 * p)) - 1

            source_image, source_label = source_data
            source_image, source_label = source_image.cuda(), source_label.cuda()
            target_image, target_label = target_data
            target_image, target_label = target_image.cuda(), target_label.cuda()

            combined_image = torch.cat((source_image, target_image), 0)  # 64 = (S:32 + T:32)
            domain_source_labels = torch.zeros(source_label.shape[0]).type(torch.LongTensor)
            domain_target_labels = torch.ones(target_label.shape[0]).type(torch.LongTensor)
            domain_combined_label = torch.cat((domain_source_labels, domain_target_labels), 0).cuda()
            domain_feature = encoder(combined_image)
            domain_output = discriminator(domain_feature, alpha)
            domain_pred = domain_output.data.max(1, keepdim=True)[1]
            domain_correct += domain_pred.eq(domain_combined_label.data.view_as(domain_pred)).cpu().sum()
            # print(batch_idx, domain_combined_label.shape)
            domain_data_count += domain_combined_label.shape[0]

    if training_mode == 'dann':
        source_acc = 100. * source_correct.item() / len(source_test_loader.dataset)
        target_acc = 100. * target_correct.item() / len(target_test_loader.dataset)
        valid_acc = 100. * valid_correct.item() / len(target_valid_loader.dataset)
        domain_acc = 100. * domain_correct.item() / (domain_data_count)
        print("Test Results on DANN :")
        print('\nSource Accuracy: {}/{} ({:.2f}%)\n'
              'Target Accuracy: {}/{} ({:.2f}%)\n'
              'Valid Accuracy: {}/{} ({:.2f}%)\n'
              'Domain Accuracy: {}/{} ({:.2f}%)\n'.
            format(
            source_correct, len(source_test_loader.dataset), source_acc,
            target_correct, len(target_test_loader.dataset), target_acc,
            valid_correct, len(target_valid_loader.dataset), valid_acc,
            domain_correct, domain_data_count, domain_acc))


    else:
        source_acc = 100. * source_correct.item() / len(source_test_loader.dataset)
        target_acc = 100. * target_correct.item() / len(target_test_loader.dataset)
        valid_acc = 100. * valid_correct.item() / len(target_valid_loader.dataset)
        domain_acc = 0
        print("Test results on source_only :")
        print('\nSource Accuracy: {}/{} ({:.2f}%)\n'
              'Target Accuracy: {}/{} ({:.2f}%)\n'
              'Valid Accuracy: {}/{} ({:.2f}%)\n'.format(
            source_correct, len(source_test_loader.dataset), source_acc,
            target_correct, len(target_test_loader.dataset), target_acc,
            valid_correct, len(target_valid_loader.dataset), valid_acc))

    return source_acc, target_acc, valid_acc, domain_acc