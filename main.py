import os

import numpy as np
from tensorboardX import SummaryWriter
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import yaml

from augm_data import get_next_batch, supervised_batch, unsupervised_batch, update_eta
from model import Model


if torch.cuda.is_available():
    torch.set_default_tensor_type(torch.cuda.FloatTensor)
    device = torch.device("cuda")
else:
    torch.set_default_tensor_type(torch.FloatTensor)
    device = torch.device("cpu")

torch.manual_seed(137)
writer = SummaryWriter(os.getcwd())


def extract_classes(dataset, samples_per_class):
    class_indices = {} 
    count = 0
    nr_of_classes = 10
    for i in range(len(dataset)):
        _img_info, class_id = dataset.__getitem__(i)
        if class_id not in class_indices:    
            class_indices[class_id] = [i] 
            count += 1
        elif len(class_indices[class_id]) <  samples_per_class:
            class_indices[class_id].append(i)
            count += 1
        if count >= samples_per_class*nr_of_classes:
            break

    concat_indices = []  
    for (_, index) in class_indices.items():
        concat_indices += index 
    return concat_indices  


def get_data_loaders(config):
    train_transform = transforms.Compose(
        [transforms.RandomCrop(28),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))])

    trainset = torchvision.datasets.MNIST(root='./data', train=True,
                                            download=True, transform=train_transform)
    extract_class_idx = extract_classes(trainset, 1) 
    subsampler_train = torch.utils.data.SubsetRandomSampler(extract_class_idx)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=config['supervised_batch_size'],
                                            sampler=subsampler_train)

    unlabeled_idx = np.setdiff1d(np.arange(len(trainset)), extract_class_idx) 
    subsampler_train_unlabeled = torch.utils.data.SubsetRandomSampler(unlabeled_idx)
    trainloader_unlabeled = torch.utils.data.DataLoader(trainset, batch_size=config['unsupervised_batch_size'],
                                            sampler=subsampler_train_unlabeled, drop_last=True)

    test_transform = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))])

    testset = torchvision.datasets.MNIST(root='./data', train=False,
                                        download=True, transform=test_transform)
    extract_class_idx = extract_classes(testset, 100) 
    subsampler_test = torch.utils.data.SubsetRandomSampler(extract_class_idx)
    testloader = torch.utils.data.DataLoader(testset, batch_size=10,
                                            sampler=subsampler_test)

    return trainloader, trainloader_unlabeled, testloader


def train_epoch(train_loader, unlabeled_iter, unsup_batch_iterator, optimizer, model, epoch, eta, lambd):
    for _, data in enumerate(train_loader, 0):
        x_unlab, unlabeled_iter = get_next_batch(unlabeled_iter, unsup_batch_iterator)
        inputs, labels = data
        data = inputs.to(device), labels.to(device)
        optimizer.zero_grad()

        supervised_loss = supervised_batch(model, data, eta)
        writer.add_scalar("loss/supervised", supervised_loss.detach(), epoch)

        unsupervised_loss = unsupervised_batch(model, x_unlab)
        writer.add_scalar("loss/unsupervised", unsupervised_loss.detach(), epoch)

        total_loss = supervised_loss + lambd*unsupervised_loss

        total_loss.backward()
        optimizer.step()
        writer.add_scalar("loss/total", total_loss.detach(), epoch)
        print(f'[Epoch: {epoch}] loss: {total_loss.detach()}')

    return unlabeled_iter


def evaluate_model(test_loader, model, config):
    nr_classes = config['classes']
    class_correct = list(0. for i in range(nr_classes))
    class_total = list(0. for i in range(nr_classes))
    classes = config['class_names']
    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            outputs = model(images.to(device))
            _, predicted = torch.max(outputs, 1)
            c = (predicted == labels).squeeze()
            for i in range(images.size(0)):
                label = labels[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1
    mean = 0
    for i in range(nr_classes):
        acc = 100 * class_correct[i]/class_total[i]
        mean += acc
        print(f'Accuracy of {classes[i]} : {acc} %') 
    print(f'Mean Accuracy: {mean/nr_classes}')


def main():
    with open('config.yaml') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    train_loader, train_loader_unlabeled, test_loader = get_data_loaders(config)
    unlabeled_iter = iter(train_loader_unlabeled)
    model = Model()    
    model.to(device)

    optimizer = optim.SGD(model.parameters(), lr=config['lr'], momentum=0.9)     
    epochs = config['epochs']
    classes = config['classes']
    lambd = config['lambda']
    tsa_schedule = config['tsa_schedule']
    for epoch in range(epochs):
        eta = update_eta(tsa_schedule, epochs, classes, epoch)
        writer.add_scalar("eta", eta, epoch)
        unlabeled_iter = train_epoch(train_loader, unlabeled_iter, train_loader_unlabeled, optimizer, model, epoch, eta, lambd)
    
    print('Finished Training')
    evaluate_model(test_loader, model, config)
    writer.close()


if __name__ == "__main__":
    main()
                