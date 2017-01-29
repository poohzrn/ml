import matplotlib.pyplot as plt
import numpy as np

from perceptron.perceptron import Perceptron
from utils.datahelper import DataHandler


class PerceptronExample:

    @classmethod
    def Run(cls):
        """Example of classification on Iris using a perceptron"""
        # Load Data
        data = DataHandler.load_iris()
        y = data.iloc[0:100, 4].values
        y = np.where(y == 'Iris-setosa', -1, 1)
        X = data.iloc[0:100, [0, 2]].values
        # cls.scatter_plot(X,y)

        # Train and classify
        cls.train(X, y)

    @classmethod
    def scatter_plot(cls, X, y):
        "Plot a 2d graph"
        plt.scatter(X[:50, 0],
                    X[:50, 1],
                    color='red',
                    marker='o',
                    label='setosa')
        plt.scatter(X[50:100, 0],
                    X[50:100, 1],
                    color='blue',
                    marker='x',
                    label='versicolor')
        plt.xlabel('sepal length')
        plt.ylabel('petal length')
        plt.legend(loc='upper left')
        plt.show()

    @classmethod
    def train(cls, X, y, visualize=False):
        """Train our perceptron """
        perceptron = Perceptron(learning_rate=0.2, iterations=10)
        perceptron.fit(X, y)
        if visualize:
            plt.plot(range(1, len(perceptron.errors) + 1),
                     perceptron.errors, marker='o')
            plt.xlabel('Epocs')
            plt.ylabel('Number of misclassifications')
            plt.show()
