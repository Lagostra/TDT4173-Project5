import os
import numpy as np
import cv2
import matplotlib.pyplot as plt


def load_data(path, n_images):
    x = np.empty((n_images, 20, 20), dtype=np.uint8)
    y = np.empty(n_images, dtype=np.uint8)

    classes = [name for name in os.listdir(path) if os.path.isdir(os.path.join(path, name))]
    j = 0
    for i, c in enumerate(classes):
        p = os.path.join(path, c)

        for im in os.listdir(p):
            im_p = os.path.join(p, im)
            im = cv2.imread(im_p, 0)
            x[j, ...] = im
            y[j, ...] = i

            j += 1

    return x, y


def threshold(x):
    blur = cv2.medianBlur(x, 3)

    th, dst = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)

    return dst


def preprocess(x):

    for i in range(x.shape[0]):
        x[i] = threshold(x[i])

    # x = np.apply_along_axis(threshold, 0, x)

    x = x / 255

    # Flip so that all characters are black.
    # x = np.apply_along_axis(black_characters, 1, x)

    return x


def imshow(im):
    plt.imshow(im, cmap='gray', vmin=0, vmax=255)
    plt.show()


if __name__ == '__main__':
    x, y = load_data('data/chars74k-lite', 7112)

    x = preprocess(x)

    np.savez('data/preprocessed.npz', x=x, y=y)
