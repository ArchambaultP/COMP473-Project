import numpy as np
from MultiLayerNetwork import MLN
from pathlib import Path
import csv
from functions import softmax
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split

"""
    Loads data from the datasets then analyzes the neural framework.
    Datasets for the projects are: MNIST, CIFAR-10, IRIS and HASYv2
"""


def main():
    iris_path = Path('datasets/IRIS')
    with open(iris_path / "IRIS_train_test.csv", 'r') as file:
        X_data = []
        Y_data = []
        for img in csv.reader(file):
            Y_data.append(img[5])
            X_data.append(img[1:5])

    X_train_test = np.array(X_data, dtype=float)
    Y_train_test = np.array(Y_data, dtype=str)
    Y_train_test = Y_train_test.reshape((Y_train_test.shape[0], 1))
    X_train, X_test, Y_train, Y_test = train_test_split(X_train_test, Y_train_test, test_size=0.2, random_state=1)

    train_x, train_y, test_x, test_y = pre_process_data(X_train, Y_train, X_test, Y_test)

    layers_dims = [64, 32, 3]  # Layer dimensions
    dropout_probs = [0.5, 0.5]  # Chance to drop each node at Layer l. We exclude output layer from dropout.

    mln = MLN(layers_dims, dropout_probs)
    mln.fit(train_x, train_y, learning_rate=0.1, n_iterations=10000)
    print("Train Accuracy:", mln.predict(train_x, train_y))
    print("Test Accuracy:", mln.predict(test_x, test_y))
    mln.plot_cost()


def pre_process_data(train_x, train_y, test_x, test_y):
    """
    Pre process MNIST data, may need to process other datasets differently
    """
    train_x = train_x
    test_x = test_x
    enc = OneHotEncoder(sparse=False, categories='auto')
    train_y = enc.fit_transform(train_y.reshape(len(train_y), -1))

    test_y = enc.transform(test_y.reshape(len(test_y), -1))

    return train_x, train_y, test_x, test_y


if __name__ == "__main__":
    main()