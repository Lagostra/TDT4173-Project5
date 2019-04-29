import cv2
import numpy as np
import cnn
from cnn import Network
import randomforest
import matplotlib.pyplot as plt
from matplotlib import patches

from preprocess import threshold


def load(path):
    return cv2.imread(path, 0)


def preprocess(image):
    return image / 255


def overlap(l1, r1, l2, r2):
    if l1[0] > r2[0] or l2[0] > r1[0]:
        return False

    if l1[1] > r2[1] or l2[1] > r1[1]:
        return False

    return True


def detect_characters(model, image, model_type='randomforest', filter_overlaps=True, rotate=False):
    coordinates = []
    confidence = []
    sections = []
    for y in range(0, image.shape[0] - 20, 1):
        for x in range(0, image.shape[1] - 20, 1):
            s = image[y:y+20, x:x+20]

            if s.mean() > 0.70:
                continue

            sections.append(s)
            coordinates.append((x, y))

            if rotate:
                M = cv2.getRotationMatrix2D((9.5, 9.5), 90, 1.0)
                rotated = cv2.warpAffine(s, M, (20, 20))
                sections.append(rotated)
                M = cv2.getRotationMatrix2D((9.5, 9.5), 180, 1.0)
                sections.append(cv2.warpAffine(s, M, (20, 20)))
                M = cv2.getRotationMatrix2D((9.5, 9.5), 270, 1.0)
                sections.append(cv2.warpAffine(s, M, (20, 20)))

                coordinates.extend([(x, y)] * 3)

    sections = np.array(sections)
    coordinates = np.array(coordinates)

    if 'threshold' in model_type:
        for i in range(sections.shape[0]):
            sections[i] = threshold(sections[i])


    if model_type == 'randomforest':
        pred = randomforest.evaluate(model, sections)
        conf = pred.max(1)
        confidence = conf
    elif model_type == 'cnn':
        for i in range(0, sections.shape[0], 1000):
            pred = cnn.evaluate(model, sections[i:i + 1000])
            conf = pred.max(1)[0]
            confidence.extend(conf.tolist())
    confidence = np.array(confidence)

    char_detected = np.where(confidence > 0.5)

    selected_coords = char_detected[0].tolist()

    if filter_overlaps:
        for i in char_detected[0]:
            for j in char_detected[0]:
                if i == j:
                    continue

                if i not in selected_coords or j not in selected_coords:
                    continue

                c1 = coordinates[i]
                c2 = coordinates[j]

                if overlap(c1, (c1[0] + 20, c1[1] + 20), c2, (c2[0] + 20, c2[1] + 20)):
                    if confidence[i] > confidence[j]:
                        selected_coords.remove(j)
                    else:
                        selected_coords.remove(i)

    return coordinates[selected_coords]


def plot_result(image, coords):
    fig, ax = plt.subplots(1)
    ax.imshow(image, cmap='gray', vmin=0, vmax=1)

    for c in coords:
        rect = patches.Rectangle((c[0], c[1]), 20, 20, linewidth=1, edgecolor='r', fill=False)
        ax.add_patch(rect)

    plt.show()


if __name__ == '__main__':
    image = load('data/detection-images/detection-2.jpg')
    image = preprocess(image)

    model_type = ('cnn', 'randomforest', 'randomforest-thresholded')[2]

    if model_type.startswith('randomforest'):
        model = randomforest.load(f'model/{model_type}')
    elif model_type.startswith('cnn'):
        model = cnn.load(f'model/{model_type}')

    detected_coords = detect_characters(model, image, rotate=False, model_type=model_type)

    plot_result(image, detected_coords)
