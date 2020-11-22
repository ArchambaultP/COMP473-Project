import numpy as np
import matplotlib.pylab as plt
from pathlib import Path
import csv
from functions import sigmoid, softmax, sigmoid_derivative

"""
    Multilayered neural network class.

    Note: only the sigmoid and softmax activation functions have been implemented.
"""

class MLN:
    def __init__(self, layers_size, dropout_probs):
        self.layers_size = layers_size
        self.dropout_probs = dropout_probs
        self.parameters = {}
        self.L = len(self.layers_size)
        self.n = 0
        self.costs = []
 
 
    def initialize_parameters(self):
        """
        Initializes the network's parameters
        """ 
        for l in range(1, len(self.layers_size)):
            self.parameters[f"W{l}"] = np.random.randn(self.layers_size[l], self.layers_size[l - 1]) / np.sqrt(
                self.layers_size[l - 1])
            self.parameters[f"b{l}"] = np.zeros((self.layers_size[l], 1))
 
    def forward_propagation(self, X):
        """
        Compute forward propagation for (L-1)* Sigmoid -> Softmax
        """
        store = {}
 
        A = X.T
        for l in range(self.L - 1):
            Z = self.parameters[f"W{l+1}"].dot(A) + self.parameters[f"b{l+1}"]
            A = sigmoid(Z)
            drop_prob = self.dropout_probs[l]
            D = np.random.rand(A.shape[0], A.shape[1])
            D = D >= drop_prob
            A = (A * D)/(1 - drop_prob)
            store[f"A{l+1}"] = A
            store[f"W{l+1}"] = self.parameters[f"W{l+1}"]
            store[f"Z{l+1}"] = Z
            store[f"D{l+1}"] = D
 
        Z = self.parameters[f"W{self.L}"].dot(A) + self.parameters[f"b{self.L}"]
        A = softmax(Z)
        store[f"A{self.L}"] = A
        store[f"W{self.L}"] = self.parameters[f"W{self.L}"]
        store[f"Z{self.L}"] = Z
 
        return A, store
 
    def back_propagation(self, X, Y, store):
        """
        Compute Back propagation for Softmax -> (L-1) * Sigmoid
        """
        derivatives = {}
 
        store["A0"] = X.T
 
        A = store[f"A{self.L}"]
        dZ = A - Y.T
 
        dW = dZ.dot(store[f"A{self.L-1}"].T) / self.n
        db = np.sum(dZ, axis=1, keepdims=True) / self.n
        dAPrev = store[f"W{self.L}"].T.dot(dZ)
 
        derivatives[f"dW{self.L}"] = dW
        derivatives[f"db{self.L}"] = db
 
        for l in range(self.L - 1, 0, -1):
            D = store[f"D{l}"]
            dAPrev = (dAPrev*D)/(1 - self.dropout_probs[l-1])
            dZ = dAPrev * sigmoid_derivative(store[f"Z{l}"])
            dW = dZ.dot(store["A" + str(l - 1)].T) / self.n
            db = np.sum(dZ, axis=1, keepdims=True) / self.n
            if l > 1:
                dAPrev = store[f"W{l}"].T.dot(dZ)
 
            derivatives[f"dW{l}"] = dW
            derivatives[f"db{l}"] = db
 
        return derivatives
 
    def fit(self, X, Y, learning_rate=0.01, n_iterations=2500):
        """
        Fit our model to our training data X and true labels Y. Y should be one hot encoded for multiclass prediction
        """
 
        self.n = X.shape[0]
 
        self.layers_size.insert(0, X.shape[1])
 
        self.initialize_parameters()
        for loop in range(n_iterations):
            A, store = self.forward_propagation(X)
            cost = -np.mean(Y * np.log(A.T+ 1e-8)) # add a small number to avoid log(0) -> cost will never reach 0
            derivatives = self.back_propagation(X, Y, store)
 
            for l in range(1, self.L + 1):
                self.parameters[f"W{l}"] = self.parameters[f"W{l}"] - learning_rate * derivatives[f"dW{l}"]
                self.parameters[f"b{l}"] = self.parameters[f"b{l}"] - learning_rate * derivatives[f"db{l}"]
 
            if loop % 100 == 0:
                print("Cost: ", cost, "Train Accuracy:", self.predict(X, Y))
 
            if loop % 10 == 0:
                self.costs.append(cost)
 
    def predict(self, X, Y):
        """
        Computes the accuracy of predictions applied to a batch of samples.
        """
        A, cache = self.forward_propagation(X)
        y_hat = np.argmax(A, axis=0)
        Y = np.argmax(Y, axis=1)
        accuracy = (y_hat == Y).mean()
        return accuracy * 100
 
    def plot_cost(self):
        plt.figure()
        plt.plot(np.arange(len(self.costs)), self.costs)
        plt.xlabel("epochs")
        plt.ylabel("cost")
        plt.show()

 
 
if __name__ == '__main__':
    main()

