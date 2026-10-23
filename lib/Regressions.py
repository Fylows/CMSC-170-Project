import numpy as np


class LinearRegression(object):
    pass

class PolyNomialRegression(object):
    pass

# Referenced from
# https://www.geeksforgeeks.org/machine-learning/implementation-of-ridge-regression-from-scratch-using-python/
# https://www.geeksforgeeks.org/machine-learning/what-is-ridge-regression/
class RidgeRegression(object):
    def __init__(self, iterations=1000, learning_rate=0.01, alpha=1.0):
        self.alpha = alpha
        self.iterations = iterations
        self.learning_rate = learning_rate
    
    # training the model using gradient descent
    def fit(self, X, y):
        self.m, self.n = X.shape
        
        for _ in range(self.iterations):
            Y_pred = self.predict(X)
            
            # gradient calculation
            dw = (-2 * (X.T).dot(y - Y_pred) + self.alpha * np.sign(self.weights)) / self.m
            db = -2 * np.sum(y - Y_pred) / self.m

            # weight update
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

    def predict(self, X):
        return X.dot(self.weights) + self.bias

class ARIMARegression(object):
    pass