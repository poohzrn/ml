#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from .perceptron import Perceptron
from utils.plotting import plot_decision_regions
from utils.datahelper import load_iris


class ADALine(Perceptron):

    def __init__(self, learning_rate=0.01, iterations=10):
        self.cost = []
        Perceptron.__init__(self, learning_rate, iterations)

    def fit(self, X, y):
        self.weights = np.zeros(1 + X.shape[1])
        for iteration in range(self.iterations):
            output = self.net_input(X)
            errors = (y - output)
            self.weights[1:] += self.learning_rate * X.T.dot(errors)
            self.weights[0] += self.learning_rate * errors.sum()
            self.cost.append((errors ** 2).sum() / 2.0)
        return self

    def activation(self, X):
        "Linear Activation"
        return self.net_input(X)

    def predict(self, X):
        return np.where(self.activation(X) >= 0.0, 1, -1)


def ADALineExample(visualize=False):
    adaline = ADALine(learning_rate=0.001, iterations=25)
    data = load_iris()
    y = data.iloc[0:100, 4].values
    y = np.where(y == 'Iris-setosa', -1, 1)
    X = data.iloc[0:100, [0, 2]].values
    # Standardize the data:
    X_std = np.copy(X)
    X_std[:, 0] = (X[:, 0] - X[:, 0].mean()) / X[:, 0].std()
    X_std[:, 1] = (X[:, 1] - X[:, 1].mean()) / X[:, 1].std()

    adaline.fit(X_std, y)
    if not visualize:
        plot_decision_regions(X_std, y, classifier=adaline)
        plt.show()
        plt.plot(range(1, len(adaline.cost) + 1), adaline.cost)
        plt.xlabel('Epocs')
        plt.ylabel('SSE')
        plt.show()
