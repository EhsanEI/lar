import torch
import numpy as np
import dannutils
import torch.optim as optim
import torch.nn as nn
import test
# import mnist
from dannutils import save_model
from dannutils import visualize
from dannutils import set_model_mode
import params

# Source : 0, Target :1
# source_test_loader = mnist.mnist_test_loader
# if params.target_domain == 'mnistm':
#     import mnistm
#     target_test_loader = mnistm.mnistm_test_loader
#     # target_test_loader = mnist.mnist_train_loader
# else:
#     import usps
#     target_test_loader = usps.usps_test_loader


def source_only(encoder, classifier, source_train_loader, target_train_loader, 
                source_test_loader, target_test_loader, target_valid_loader, save_name, results):
    print("Source-only training")
    # classifier_criterion = nn.CrossEntropyLoss().cuda() # wrooong!
    classifier_criterion = nn.NLLLoss().cuda()
    optimizer = optim.SGD(
        list(encoder.parameters()) +
        list(classifier.parameters()),
        lr=0.01, momentum=0.9)
    
    for epoch in range(params.epochs):
        print('Epoch : {}'.format(epoch))
        set_model_mode('train', [encoder, classifier])

        start_steps = epoch * len(source_train_loader)
        total_steps = params.epochs * len(target_train_loader)
        target_train_iter = iter(target_train_loader)
        
        for batch_idx, source_data in enumerate(source_train_loader):
            try:
                target_data = next(target_train_iter)
            except StopIteration:
                target_train_iter = iter(target_train_loader)
                target_data = next(target_train_iter)

            # print(source_data.shape, target_data.shape)
            source_image, source_label = source_data
            p = float(batch_idx + start_steps) / total_steps

            if params.target_domain == 'mnistm':
                source_image = torch.cat((source_image, source_image, source_image), 1)  # MNIST convert to 3 channel
            source_image, source_label = source_image.cuda(), source_label.cuda()  # 32
            # print(source_image.shape)

            optimizer = dannutils.optimizer_scheduler(optimizer=optimizer, p=p)
            optimizer.zero_grad()

            source_feature = encoder(source_image)

            # Classification loss
            class_pred = classifier(source_feature)
            # print(class_pred.shape, source_label.shape)
            class_loss = classifier_criterion(class_pred, source_label)

            class_loss.backward()
            optimizer.step()
            # if (batch_idx + 1) % 50 == 0:
            #     print('[{}/{} ({:.0f}%)]\tClass Loss: {:.6f}'.format(batch_idx * len(source_image), len(source_train_loader.dataset), 100. * batch_idx / len(source_train_loader), class_loss.item()))

        if (epoch + 1) % 1 == 0:
            source_acc, target_acc, valid_acc, domain_acc = test.tester(
                encoder, classifier, None, source_train_loader, target_test_loader, target_valid_loader, training_mode='source_only')
            results['source_acc'].append(source_acc)
            results['target_acc'].append(target_acc)
            results['valid_acc'].append(valid_acc)
            results['domain_acc'].append(domain_acc)

    # save_model(encoder, classifier, None, 'source', save_name)
    # visualize(encoder, 'source', save_name)


def dann(encoder, classifier, discriminator, source_train_loader, target_train_loader,
         source_test_loader, target_test_loader, target_valid_loader, save_name, results):
    print("DANN training")
    
    # classifier_criterion = nn.CrossEntropyLoss().cuda() # wrooong!
    classifier_criterion = nn.NLLLoss().cuda()
    # discriminator_criterion = nn.CrossEntropyLoss().cuda() # wrooong!
    discriminator_criterion = nn.NLLLoss().cuda()
    
    optimizer = optim.SGD(
    list(encoder.parameters()) +
    list(classifier.parameters()) +
    list(discriminator.parameters()),
    lr=0.01,
    momentum=0.9)
    
    for epoch in range(params.epochs):
        print('Epoch : {}'.format(epoch))
        set_model_mode('train', [encoder, classifier, discriminator])

        start_steps = epoch * len(source_train_loader)
        total_steps = params.epochs * len(target_train_loader)
        target_train_iter = iter(target_train_loader)
        
        for batch_idx, source_data in enumerate(source_train_loader):
            try:
                target_data = next(target_train_iter)
            except StopIteration:
                target_train_iter = iter(target_train_loader)
                target_data = next(target_train_iter)

            source_image, source_label = source_data
            target_image, target_label = target_data

            p = float(batch_idx + start_steps) / total_steps
            alpha = 2. / (1. + np.exp(-10 * p)) - 1

            if params.target_domain == 'mnistm':
                source_image = torch.cat((source_image, source_image, source_image), 1)

            source_image, source_label = source_image.cuda(), source_label.cuda()
            target_image, target_label = target_image.cuda(), target_label.cuda()
            combined_image = torch.cat((source_image, target_image), 0)

            optimizer = dannutils.optimizer_scheduler(optimizer=optimizer, p=p)
            optimizer.zero_grad()

            combined_feature = encoder(combined_image)
            source_feature = encoder(source_image)

            # 1.Classification loss
            class_pred = classifier(source_feature)
            class_loss = classifier_criterion(class_pred, source_label)

            # 2. Domain loss
            domain_pred = discriminator(combined_feature, alpha)

            domain_source_labels = torch.zeros(source_label.shape[0]).type(torch.LongTensor)
            domain_target_labels = torch.ones(target_label.shape[0]).type(torch.LongTensor)
            domain_combined_label = torch.cat((domain_source_labels, domain_target_labels), 0).cuda()
            domain_loss = discriminator_criterion(domain_pred, domain_combined_label)

            total_loss = class_loss + domain_loss
            total_loss.backward()
            optimizer.step()

            # if (batch_idx + 1) % 50 == 0:
            #     print('[{}/{} ({:.0f}%)]\tLoss: {:.6f}\tClass Loss: {:.6f}\tDomain Loss: {:.6f}'.format(
            #         batch_idx * len(source_image), len(source_train_loader.dataset), 100. * batch_idx / len(source_train_loader), total_loss.item(), class_loss.item(), domain_loss.item()))

        if (epoch + 1) % 1 == 0:
            source_acc, target_acc, valid_acc, domain_acc = test.tester(
                encoder, classifier, discriminator, source_test_loader, target_test_loader, target_valid_loader, training_mode='dann')
            results['source_acc'].append(source_acc)
            results['target_acc'].append(target_acc)
            results['valid_acc'].append(valid_acc)
            results['domain_acc'].append(domain_acc)

    # save_model(encoder, classifier, discriminator, 'source', save_name)
    # visualize(encoder, 'source', save_name)
