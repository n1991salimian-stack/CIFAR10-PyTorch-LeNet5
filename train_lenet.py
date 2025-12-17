"""
Project: CIFAR-10 Image Classification using LeNet-5
Framework: PyTorch
Author: Seyede Najme Salimian
Date: December 2025
Description: Implementation of a classic CNN architecture for object recognition.
"""
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

print(torch.cuda.is_available())

class LeNet5(nn.Module):
    def __init__(self):
        super(LeNet5, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, kernel_size=5)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
        self.tanh = nn.Tanh()  # Tanh activation function

    def forward(self, x):
        x = self.tanh(self.conv1(x))
        x = torch.nn.functional.avg_pool2d(x, 2)
        x = self.tanh(self.conv2(x))
        x = torch.nn.functional.avg_pool2d(x, 2)
        x = x.view(-1, self.num_flat_features(x))
        x = self.tanh(self.fc1(x))
        x = self.tanh(self.fc2(x))
        x = self.fc3(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

# Load dataset
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
trainset, valset = torch.utils.data.random_split(trainset, [45000, 5000])

trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                          shuffle=True, num_workers=2)
valloader = torch.utils.data.DataLoader(valset, batch_size=4,
                                        shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

class Model:
    def __init__(self, model, learning_rate, device):
        self.model = model
        self.lr = learning_rate
        self.loss = nn.CrossEntropyLoss()
        self.opt = optim.Adam(self.model.parameters(), lr=self.lr)
        self.train_loss = []
        self.val_loss = []
        self.train_acc = []
        self.val_acc = []
        self.device = device

    def batch_accuracy(self, output, target):
        _, predicted = torch.max(output, 1)
        correct = (predicted == target).sum().item()
        return correct / target.size(0) * 100

    def train_step(self, train_loader):
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        for data in train_loader:
            inputs, labels = data[0].to(self.device), data[1].to(self.device)
            self.opt.zero_grad()
            outputs = self.model(inputs)
            loss = self.loss(outputs, labels)
            loss.backward()
            self.opt.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        train_loss = running_loss / len(train_loader)
        train_acc = 100 * correct / total
        self.train_loss.append(train_loss)
        self.train_acc.append(train_acc)

    def validation_step(self, val_loader):
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for data in val_loader:
                inputs, labels = data[0].to(self.device), data[1].to(self.device)
                outputs = self.model(inputs)
                loss = self.loss(outputs, labels)

                running_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        val_loss = running_loss / len(val_loader)
        val_acc = 100 * correct / total
        self.val_loss.append(val_loss)
        self.val_acc.append(val_acc)


# Initialize the model, criterion, and optimizer
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = LeNet5().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training
epochs = 20
model_instance = Model(model, 0.001, device)
for epoch in tqdm(range(epochs), desc='Epoch'):
    model_instance.train_step(trainloader)
    model_instance.validation_step(valloader)
 #plot
plt.figure(dpi=100)
plt.grid()
plt.plot(model_instance.train_acc)
plt.plot(model_instance.val_acc)
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.show()
