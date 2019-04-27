import numpy as np
import pickle

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


def save(model, path):
    with open(path, 'wb') as f:
        pickle.dump(model, f)


def load(path):
    with open(path, 'rb') as f:
        return pickle.load(f)


if __name__ == '__main__':
    data = np.load('data/raw.npz')
    x, y = data['x'], data['y']

    x = np.reshape(x, (-1, 20*20))

    train_x, test_x, train_y, test_y = train_test_split(x, y, random_state=42)

    model = RandomForestClassifier(n_estimators=2000, max_features=50, n_jobs=4, verbose=True)
    model.fit(train_x, train_y)

    save(model, 'model/randomforest')

    pred_test = model.predict(test_x)

    print(f'Accuracy: {accuracy_score(test_y, pred_test)}')
