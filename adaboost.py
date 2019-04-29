import numpy as np
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import train_test_split

if __name__ == '__main__':
    data = np.load('data/raw.npz')
    x, y = data['x'], data['y']

    x = np.reshape(x, (-1, 20*20))

    train_x, test_x, train_y, test_y = train_test_split(x, y, random_state=42)

    model = AdaBoostClassifier(n_estimators=1000)
    model.fit(train_x, train_y)

    print(f'Accuracy: {model.score(test_x, test_y)}')