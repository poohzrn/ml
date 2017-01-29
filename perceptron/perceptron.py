import numpy as np


class Perceptron:

    """A simple perceptron classifier
        Parameters:
            learning_rate: float
                learning rate for the classifier.
                Must the value must be between 0.0 and 1.0
            iterations: int
                Passes over the training data
        Attributes:
            weights: 1-d array
                weights after fitting.
            errors: number of misclassifications in every epoch
    """

    def __init__(self, learning_rate=0.01, iterations=10):
        """ """
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.errors = []
        self.weights = None

    def fit(self, X, y):
        """Fit training data.

        :X: Matrix, shape = [n_samples, n_features]
            Training vectors, where n_samples denotes number of samples
            and n_features is the number of features
        :y: Array-like, shape = [n_samples]
            target values
        :returns:
            self : object
        """

        self.weights = np.zeros(1 + X.shape[1])
        for _ in range(self.iterations):
            errors = 0
            for xi, target in zip(X, y):
                update = self.learning_rate * (target - self.predict(xi))
                self.weights[1:] += update * xi
                self.weights[0] += update
                errors += int(update != 0.0)
            self.errors.append(errors)
        return self

    def net_input(self, X):
        "Calculate net input"
        return np.dot(X, self.weights[1:]) + self.weights[0]

    def predict(self, X):
        "Return class label after unit step"
        return np.where(self.net_input(X) >= 0.0, 1, -1)
