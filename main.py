""" 
    use pnasNet and CIFAR-10
    
    use new pytorch 0.4.0 code style
"""

import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms

from nasnet_pytorch import *

import time

from tqdm import tqdm     # progress bar


"""
    prepare train set and test set
"""
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])
     
cifar10_path = '../CIFAR-10'
     
trainset = torchvision.datasets.CIFAR10(root=cifar10_path, train=True,
                                        download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, 
                                          shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root=cifar10_path, train=False,
                                       download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=100, 
                                         shuffle=False, num_workers=2)

           
"""
    build the network
"""
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

net = NASnet().to(device)
net = nn.DataParallel(net)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[70,140,250], gamma=0.1)


"""
    training function
"""
def train(epoch):
    running_loss = 0.0
    
    start_time = time.time()
    
    scheduler.step()
    for param_group in optimizer.param_groups:
        print("Current learning rate:", param_group['lr'])
    
    net.train()
    
    for inputs, labels in tqdm(trainloader):
        
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()

        epoch = torch.Tensor([epoch])
        y, aux_head = net(inputs, epoch)

        loss_y = criterion(outputs, labels)
        loss_aux_head = criterion(aux_head, labels)
        
        loss = loss_y + 0.4 * loss_aux_head
        
        loss.backward()
        
        optimizer.step()
        
        running_loss += loss.item() # get python number from tensor
    
    print("Epoch %d: Loss: %f" %(epoch, running_loss))
    print("duration:", time.time()-start_time)
    


"""
    test function
"""
def test(epoch):
    global best_acc
    test_loss = 0
    correct = 0
    total = 0

    net.eval()
    
    with torch.no_grad():      # don't need grad when testing
        for images, labels in testloader:
            images, labels = images.to(device), labels.to(device)
        
            outputs = net(images)
            loss = criterion(outputs, labels)
        
            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    acc = 100.*correct/total
    print('Acc in epoch %d: %.3f%% (%d/%d)' %(epoch, acc, correct, total))
    
    if acc > best_acc:
        best_acc = acc
        

# the best accuracy    
best_acc = 0

if __name__ == "__main__":
    for epoch in range(500):
        train(epoch)
        test(epoch)
    print("THE best accuracy:", best_acc)
    
