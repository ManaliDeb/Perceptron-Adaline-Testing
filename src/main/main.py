import numpy as np
import matplotlib.pyplot as plt


# create perceptron class
class Perceptron:
    def __init__(self, learning_rate=0.01, iters=1000):
        self.lr = learning_rate
        self.iters = iters
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        # initialize weight and bias
        samples, features = X.shape
        self.weights = np.zeros(features)
        self.bias = 0

        # learning algo
        for _ in range(self.iters):
            for idx, x_i in enumerate(X):
                linear_output = np.dot(x_i, self.weights) + self.bias
                y_predicted = self._unit_step_function(linear_output)

                # weight update rule
                update = self.lr * (y[idx] - y_predicted)
                self.weights += update * x_i
                self.bias += update
