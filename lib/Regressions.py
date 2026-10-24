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
        n, n_features = X.shape
        n_targets = y.shape[1] # number of target variables 
        
        # initialize weights and biases
        self.weights = np.zeros((n_features, n_targets), dtype=float) # weights should match dimensions of features x targets
        self.bias = np.zeros(n_targets, dtype=float) # bias for each target variable

        for _ in range(self.iterations):
            Y_pred = self.predict(X)
            diff = y - Y_pred
            
            # gradients of weights and bias
            dw = (-2 * X.T.dot(diff) + 2 * self.alpha * self.weights) / n
            db = -2 * np.sum(diff, axis=0) / n

            # update weights and bias
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db
        return self

    def predict(self, X):
        return X.dot(self.weights) + self.bias

class ARIMARegression(object):
    pass