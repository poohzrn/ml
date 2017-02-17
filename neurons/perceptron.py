#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from utils.datahelper import load_iris


class Perceptron:

    """A simple perceptron classifier
        Parameters:
            learning_rate: learning rate for the classifier.
            iterations: Passes over the training data
        Attributes:
            weights: 1-d array:  weights after fitting.
            errors: number of misclassifications in every epoch
    """

    def __init__(self, learning_rate=0.01, iterations=10):
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
        # Calculate net input
        return np.dot(X, self.weights[1:]) + self.weights[0]

    def predict(self, X):
        # Return class label after unit step
        return np.where(self.net_input(X) >= 0.0, 1, -1)


def PerceptronExample(visualize=False):
    "Example of classification on Iris using a perceptron"
    # Load Data
    data = load_iris()
    y = data.iloc[0:100, 4].values
    y = np.where(y == 'Iris-setosa', -1, 1)
    X = data.iloc[0:100, [0, 2]].values
    # scatter_plot(X,y)
    # Train our perceptron
    perceptron = Perceptron(learning_rate=0.2, iterations=10)
    perceptron.fit(X, y)
    if visualize:
        plt.plot(range(1, len(perceptron.errors) + 1),
                 perceptron.errors, marker='o')
        plt.xlabel('Epocs')
        plt.ylabel('Number of misclassifications')
        plt.show()


def scatter_plot(X, y):
    "Plot a 2d graph"
    plt.scatter(X[:50, 0], X[:50, 1], color='red',
                marker='o', label='setosa')
    plt.scatter(X[50:100, 0], X[50:100, 1], color='blue',
                marker='x', label='versicolor')
    plt.xlabel('sepal length')
    plt.ylabel('petal length')
    plt.legend(loc='upper left')
    plt.show()
