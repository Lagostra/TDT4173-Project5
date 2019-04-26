import os
import numpy as np
from imageio import imread


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


def black_characters(x):
    if np.mean(x) > 0.5:
        return 1.0 - x


def preprocess(x):
    x = x / 255

    # Flip so that all characters are black.
    # x = np.apply_along_axis(black_characters, 1, x)

    return x


if __name__ == '__main__':
    x, y = load_data('data/chars74k-lite', 7112)

    x = preprocess(x)

    np.savez('data/preprocessed.npz', x=x, y=y)
