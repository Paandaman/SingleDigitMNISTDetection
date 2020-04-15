from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

from model import Model

if torch.cuda.is_available():
    torch.set_default_tensor_type(torch.cuda.FloatTensor)
    device = torch.device("cuda")
else:
    torch.set_default_tensor_type(torch.FloatTensor)
    device = torch.device("cpu")


def extract_classes(dataset, samples_per_class) -> List[int]:
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


def get_data_loader(): 
    transform = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))])

    trainset = torchvision.datasets.MNIST(root='./data', train=True,
                                            download=True, transform=transform)
    extract_class_idx = extract_classes(trainset, 1) 
    subsampler_train = torch.utils.data.SubsetRandomSampler(extract_class_idx)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=10,
                                            sampler=subsampler_train)


    testset = torchvision.datasets.MNIST(root='./data', train=False,
                                        download=True, transform=transform)
    extract_class_idx = extract_classes(testset, 100) 
    subsampler_test = torch.utils.data.SubsetRandomSampler(extract_class_idx)
    testloader = torch.utils.data.DataLoader(testset, batch_size=10,
                                            sampler=subsampler_test)

    return trainloader, testloader


def train_epoch(train_loader, optimizer, model, criterion, epoch):
    for i, data in enumerate(train_loader, 0):
        inputs, labels = data

        optimizer.zero_grad()

        outputs = model(inputs.to(device))
        loss = criterion(outputs, labels.to(device))
        loss.backward()
        optimizer.step()

        print(f'[Epoch: {epoch}] loss: {round(loss.item(), 40)}')


def evaluate_model(test_loader, model):
    class_correct = list(0. for i in range(10))
    class_total = list(0. for i in range(10))
    classes = ('0', '1', '2', '3',
            '4', '5', '6', '7', '8', '9')
    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            outputs = model(images.to(device))
            _, predicted = torch.max(outputs, 1)
            c = (predicted == labels).squeeze()
            for i in range(10):
                label = labels[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1
    mean = 0
    for i in range(10):
        acc = 100 * class_correct[i]/class_total[i]
        mean += acc
        print(f'Accuracy of {classes[i]} : {acc} %') 
    print(f'Mean Accuracy: {mean/10}')

def main():
    train_loader, test_loader = get_data_loader()
    model = Model()    
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)    
    # What about validation data?
    for epoch in range(1000):
        train_epoch(train_loader, optimizer, model, criterion, epoch)

    print('Finished Training')
    evaluate_model(test_loader, model)


if __name__ == "__main__":
    # put classes etc in config file?
    main()
                