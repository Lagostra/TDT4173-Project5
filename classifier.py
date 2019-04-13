import numpy as np
import os
import matplotlib.pyplot as plt

from imageio import imread

CLASSES = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm',
           'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']


def load_data(path, n_images):
    x = np.empty((n_images, 20, 20), dtype=np.uint8)
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


def plot_images(x, y):
    f, axes1 = plt.subplots(len(x) // 5, 5)
    axes = np.reshape(axes1, -1)
    for i, im in enumerate(x):
        axes[i].imshow(im, cmap='gray', vmin=0, vmax=255)
        axes[i].set_title(CLASSES[y[i]])

    plt.show()


def plot_random(x, y, n):
    choice = np.random.choice(len(x), n)
    plot_images(x[choice], y[choice])


if __name__ == '__main__':
    x, y = load_data('data/chars74k-lite', 7112)
    plot_random(x, y, 10)
