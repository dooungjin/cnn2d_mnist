import numpy as np
import torch
import torchvision
import torch.nn.functional as F
from torch.autograd import Variable
from time import time
from torchvision import datasets, transforms
from torch import nn, optim

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

trainset = datasets.MNIST('./data/', download=True, train=True, transform=transform)
valset = datasets.MNIST('./data/', download=True, train=False, transform=transform)
train_loader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)
val_loader = torch.utils.data.DataLoader(valset, batch_size=64, shuffle=True)

epochs = 20
learning_rate = 0.001
drop_rate = 0.5
conv_dims = [1, 16, 32]
dense_dims = [512, 128, 10]

class Net(nn.Module):
    def __init__(self, conv_dims, dense_dims, drop_rate):
        super(Net, self).__init__()

        self.conv_dims = conv_dims
        self.dense_dims = dense_dims
        self.drop_rate = drop_rate

        for i in range(1, len(self.conv_dims)):
            setattr(self, 'conv_' + str(i), nn.Conv2d(self.conv_dims[i-1], self.conv_dims[i], kernel_size = 5))

        for i in range(1, len(self.dense_dims)):
            setattr(self, 'dense_' + str(i), nn.Linear(self.dense_dims[i-1], self.dense_dims[i]))

        self.mp = nn.MaxPool2d(2)
        self.dropout = nn.Dropout(self.drop_rate)

    def forward(self, x):
        for i in range(1, len(self.conv_dims)):
            x = F.relu(self.mp(self.dropout(getattr(self, 'conv_' + str(i))(x))))

        x = x.view(-1, self.dense_dims[0])
        for i in range(1, len(self.dense_dims)):
            if i == (len(self.dense_dims) - 1):
                x = F.relu(getattr(self, 'dense_' + str(i))(x))
            else:
                x = self.dropout(F.relu(getattr(self, 'dense_' + str(i))(x)))

        return x

model = Net(conv_dims, dense_dims, drop_rate)
model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)

for epoch in range(0, epochs):
    train_loss = 0

    model.train()
    for i, data in enumerate(train_loader, 0):
        inputs, labels = data
        inputs, labels = Variable(inputs).to(device), Variable(labels).to(device)
        y_pred = model(inputs).to(device)
        loss = criterion(y_pred, labels.squeeze()).to(device)
        train_loss = train_loss + loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print("Train Loss:", "{:.6f}".format(train_loss))
