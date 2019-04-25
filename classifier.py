import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from imageio import imread

CLASSES = ('a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm',
           'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z')


def one_hot_encode(y, classes):
    result = np.zeros((len(y), classes))
    result[np.arange(len(y)), y] = 1
    return result


def load_data(path, n_images):
    x = np.empty((n_images, 1, 20, 20), dtype=np.uint8)
    y = np.empty(n_images, dtype=np.uint8)

    classes = [name for name in os.listdir(path) if os.path.isdir(os.path.join(path, name))]
    j = 0
    for i, c in enumerate(classes):
        p = os.path.join(path, c)

        for im in os.listdir(p):
            im_p = os.path.join(p, im)
            x[j, ...] = imread(im_p)
            y[j, ...] = i

            j += 1

    return x, y


def show_images(x, y):
    f, axes1 = plt.subplots(len(x) // 5, 5)
    axes = np.reshape(axes1, -1)
    for i, im in enumerate(x):
        axes[i].imshow(im[0], cmap='gray', vmin=0, vmax=255)
        axes[i].set_title(CLASSES[y[i]])

    plt.show()


def show_random_images(x, y, n):
    choice = np.random.choice(len(x), n)
    show_images(x[choice], y[choice])


def train(network, train_x, train_y, steps=5000, minibatch_size=200, lr=0.01):
    loss_function = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(network.parameters(), lr=lr)

    running_loss = 0

    train_x, train_y = torch.from_numpy(train_x).float(), torch.from_numpy(train_y).long()

    for i in range(steps):
        c = np.random.choice(len(train_x), minibatch_size)
        x = train_x[c]
        y = train_y[c]

        inputs, labels = Variable(x), Variable(y)

        optimizer.zero_grad()
        outputs = network(inputs)
        loss = loss_function(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.data[0]

        if i % 50 == 0:
            print(running_loss / 50)
            running_loss = 0


class Network(nn.Module):

    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=5, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        self.fc1 = nn.Linear(5 * 10 * 10, 64)
        self.fc2 = nn.Linear(64, 26)

    def forward(self, x):
        # Input: (1, 20, 20)

        # -> (5, 20, 20)
        x = F.relu(self.conv1(x))

        # -> (5, 10, 10)
        x = self.pool(x)

        # Flatten -> (5 * 10 * 10)
        x = x.view(-1, 5 * 10 * 10)

        # -> 64
        x = F.relu(self.fc1(x))

        # -> 26
        x = self.fc2(x)

        return x


if __name__ == '__main__':
    x, y = load_data('data/chars74k-lite', 7112)
    show_random_images(x, y, 10)

    x = x / 255
    y = one_hot_encode(y, 26)

    train_x, test_y, train_y, test_y = train_test_split(x, y, random_state=42)

    network = Network()

    train(network, train_x, train_y)
