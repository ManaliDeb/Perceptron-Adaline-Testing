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

    def predict(self, X):
        linear_output = np.dot(X, self.weights) + self.bias
        return self._unit_step_function(linear_output)

    def _unit_step_function(self, x):
        return np.where(x >= 0, 1, 0)

# linearly separable dataset
X = np.array([
    # positive
    [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8],
    # negative
    [1, 1], [0, 1], [1, 0,], [0, 0], [2, 1], [3, 0]
])
# 1 for pos, 0 for neg
y = np.array([1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0])

model = Perceptron(learning_rate=0.1, iters=12)
model.fit(X, y)

# decision boundary