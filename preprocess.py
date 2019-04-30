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


def threshold(x, scaled=False):
    if scaled:
        x = (x * 255).astype(np.uint8)

    blur = cv2.medianBlur(x, 3)

    th, dst = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)

    dst = dst / 255

    return dst


def rotate(x, angle=90):
    M = cv2.getRotationMatrix2D((9.5, 9.5), angle, 1.0)

    return cv2.warpAffine(x, M, (20, 20))


def flip(x):
    return [
                cv2.flip(x, 0),
                cv2.flip(x, 1),
                cv2.flip(x, -1)
            ]


def preprocess(x, y, do_threshold=False, flip=True, rotate=True):

    if rotate:

        for i in range(x.shape[0]):
            rotated = rotate(x)
            x = np.append(x, np.reshape(rotated, (-1, 20, 20)), axis=0)
            y = np.append(y, y[i])

    if flip:
        for i in range(x.shape[0]):
            x = np.append(x, flip(x[i]), axis=0)
            y = np.append(y, [y[i]] * 3)

    if do_threshold:
        for i in range(x.shape[0]):
            x[i] = threshold(x[i])

    x = x / 255

    return x, y


def imshow(im):
    plt.imshow(im, cmap='gray', vmin=0, vmax=255)
    plt.show()


if __name__ == '__main__':
    x, y = load_data('data/chars74k-lite', 7112)

    x1, y1 = preprocess(x, y, do_threshold=False, flip=False, rotate=False)
    np.savez('data/raw.npz', x=x, y=y)

    x1, y1 = preprocess(x, y, do_threshold=False, flip=True, rotate=True)
    np.savez('data/enhanced.npz', x=x, y=y)

    x1, y1 = preprocess(x, y, do_threshold=True, flip=False, rotate=False)
    np.savez('data/thresholded.npz', x=x, y=y)

    x1, y1 = preprocess(x, y, do_threshold=True, flip=True, rotate=True)
    np.savez('data/enhanced-thresholded.npz', x=x, y=y)
