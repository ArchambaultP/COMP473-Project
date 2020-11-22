import numpy as np

def sigmoid(Z):
    return 1 / (1 + np.exp(-Z))
 
def softmax(Z):
    expZ = np.exp(Z - np.max(Z))
    return expZ / expZ.sum(axis=0, keepdims=True)

def sigmoid_derivative(Z):
    s = 1 / (1 + np.exp(-Z))
    return s * (1 - s)