import cv2
import numpy as np
import cnn
from cnn import Network


def load(path):
    return cv2.imread(path, 0)


def preprocess(image):
    return image / 255


def section(image):
    sections = []
    coordinates = []
    for y in range(image.shape[0] - 20):
        for x in range(image.shape[1] - 20):
            sections.append(image[y:y+20, x:x+20])
            coordinates.append((x, y))

    return np.array(sections), np.array(coordinates)


def detect_characters(model, image):

    coordinates = []
    confidences = []
    for y in range(image.shape[0] - 20):
        row = []
        for x in range(image.shape[1] - 20):
            s = image[y:y+20, x:x+20]
            row.append(s)
            coordinates.append((x, y))

        pred = cnn.evaluate(model, row)
        pred_numpy = pred.data.cpu().numpy()
        confidence = pred.max(dim=1)

        confidences.extend(confidence)

    pass


if __name__ == '__main__':
    image = load('data/detection-images/detection-1.jpg')
    image = preprocess(image)

    sections, coords = section(image)

    #model = cnn.load('model/model1')

    detect_characters(model, image)

    pass