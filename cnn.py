import pickle

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import matplotlib.pyplot as plt
import cv2

from sklearn.model_selection import train_test_split
from imageio import imread

CLASSES = ('a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm',
           'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z')

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def one_hot_encode(y, classes):
    result = np.zeros((len(y), classes))
    result[np.arange(len(y)), y] = 1
    return result


def show_images(x, y):
    f, axes1 = plt.subplots(len(x) // 5, 5)
    axes = np.reshape(axes1, -1)
    for i, im in enumerate(x):
        axes[i].imshow(im, cmap='gray', vmin=0, vmax=1)
        axes[i].set_title(CLASSES[y[i]])

    plt.show()


def show_random_images(x, y, n):
    choice = np.random.choice(len(x), n)
    show_images(x[choice], y[choice])


def pretrain(network, steps):
    network.to(device)

    x = []
    x.append(np.zeros((20, 20)))
    x.append(np.ones((20, 20)))

    for i in range(100):
        img = np.ones((20, 20))
        img = cv2.randn(img, 0.5, 0.5)

        x.append(img)

    x = np.array(x)
    y = np.array([[1] * 26] * x.shape[0])

    x, y = torch.from_numpy(x).float(), torch.from_numpy(y).float()
    inputs, labels = Variable(x).to(device), Variable(y).to(device)
    loss_function = nn.MSELoss()
    optimizer = torch.optim.Adam(network.parameters(), lr=0.1)

    for i in range(steps):
        outputs = network(inputs)
        loss = loss_function(outputs, labels)
        loss.backward()
        optimizer.step()

        if i % 50 == 0:
            print(f'Pretraining... Step {i}/{steps}')

        optimizer.zero_grad()


def train(network, train_x, train_y, steps=5000, minibatch_size=200, lr=0.01):
    loss_function = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(network.parameters(), lr=lr)

    running_loss = 0

    train_x, train_y = torch.from_numpy(train_x).float(), torch.from_numpy(train_y).long()

    network.to(device)
    network.train()

    for i in range(steps):
        c = np.random.choice(len(train_x), minibatch_size)
        x = train_x[c]
        y = train_y[c]

        inputs, labels = Variable(x), Variable(y)

        inputs = inputs.to(device)
        labels = labels.to(device)
        outputs = network(inputs)
        loss = loss_function(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.data.item()

        if i % 50 == 0:
            print(f'[Step {i}] Loss: {running_loss / 50:.3e}')
            running_loss = 0

        optimizer.zero_grad()


def test(network, test_x, test_y):
    network.to(device)
    network.eval()

    test_x, test_y = torch.from_numpy(test_x).float().to(device), torch.from_numpy(test_y).long().to(device)
    outputs = network(test_x)
    pred = torch.argmax(outputs, dim=1)

    correct = (pred == test_y).sum().item()
    total = test_x.size(0)

    loss_function = nn.CrossEntropyLoss()
    loss = loss_function(outputs, test_y)

    print()
    print(f'[Test set] Loss: {loss.data.item():.3f}\tAccuracy: {correct / total:.2%}')


def evaluate(model, x):
    x = torch.Tensor(x).to(device)
    return model(x)


def save(model, path):
    torch.save(model, path)


def load(path):
    return torch.load(path)


class Network(nn.Module):

    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        self.dropout1 = nn.Dropout2d(0.25)
        self.dropout2 = nn.Dropout()

        self.fc1 = nn.Linear(128 * 10 * 10, 1024)
        self.fc2 = nn.Linear(1024, 26)

    def forward(self, x):
        # Input: (20, 20)
        x = x.view(-1, 1, 20, 20)

        # -> (32, 20, 20)
        x = F.relu(self.conv1(x))

        # -> (32, 10, 10)
        #x = self.pool(x)

        # -> (64, 10, 10)
        x = F.relu(self.conv2(x))

        # -> (64, 5, 5)
        #x = self.pool(x)

        # -> (128, 20, 20)
        x = F.relu(self.conv3(x))

        # -> (128, 10, 10)
        x = self.pool(x)

        x = self.dropout1(x)

        # Flatten -> (128 * 10 * 10)
        x = x.view(-1, 128 * 10 * 10)

        # -> 1024
        x = F.relu(self.fc1(x))

        x = self.dropout2(x)

        # -> 26
        x = self.fc2(x)

        if not self.training:
            x = F.softmax(x, dim=1)

        return x


if __name__ == '__main__':
    data = np.load('data/raw.npz')
    x, y = data['x'], data['y']
    show_random_images(x, y, 10)

    train_x, test_x, train_y, test_y = train_test_split(x, y, random_state=42)

    network = Network()

    # pretrain(network, 1000)
    train(network, train_x, train_y, steps=5000)
    test(network, test_x, test_y)
    save(network, 'model/model1')
