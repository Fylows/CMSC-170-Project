import numpy as np

# Linear Regression: ________________________________________________

# Referenced from:
#    https://www.geeksforgeeks.org/machine-learning/ml-linear-regression/
class LinearRegression(object):
    def __init__(self, iterations = 1000, learning_rate = 0.01):
        self.iterations = iterations
        self.learning_rate = learning_rate

    def fit(self, X, y):
        n, n_features = X.shape  # Get no. of years (n) and columns (n_features)
        n_targets = y.shape[1]   # Get no. of fuel sources.

        # Set up starting line.
        self.weights = np.zeros((n_features, n_targets), dtype = float)
        self.bias = np.zeros(n_targets, dtype = float)

        # Loop for self.iterations times:
        for _ in range(self.iterations):
            Y_pred = self.predict(X)                 # Predict w/ current line.      
            diff = y - Y_pred                        # Get error.

            dw = (X.T.dot(diff) / n) * -2            # Get weight and bias gradients.
            db = (np.sum(diff, axis = 0) / n) * -2

            self.weights -= self.learning_rate * dw  # Update line.
            self.bias -= self.learning_rate * db

        return self

    def predict(self, X):
        return X.dot(self.weights) + self.bias

# Polynomial Regression: ____________________________________________

# Referenced from
# https://www.geeksforgeeks.org/machine-learning/python-implementation-of-polynomial-regression/ 
class PolynomialRegression(object):
    def __init__ (self, iterations=1000, learning_rate=0.01, degree=4):
        self.iterations = iterations            
        self.learning_rate = learning_rate
        self.degree = degree               

    def fit(self, X, y):
        n, n_features = X.shape                 # get no. of years
        n_targets = y.shape[1]                  # get no. of fuel sources

        X_poly = np.ones((n, self.degree), dtype = float)       # create array for polynomial feat.

        for i in range(self.degree):
            X_poly[:, i] = X[:, 0] ** (i + 1)         # calculate x, x^2, x^3, ^4

        n_features = X_poly.shape[1]            # update no. of feat. after expansion

        self.weights = np.zeros((n_features, n_targets), dtype = float)
        self.bias = np.zeros(n_targets, dtype = float)              # initialize bias

        # loop for n iterations
        for _ in range(self.iterations):
            Y_pred = X_poly.dot(self.weights) + self.bias       # predict w current weights and bias
            diff = y - Y_pred                                   # get error

            # weight and bias
            dw = (X_poly.T.dot(diff) / n) * -2
            db = (np.sum(diff, axis = 0)/ n) * -2

            self.weights -= self.learning_rate * dw             # update weight
            self.bias -= self.learning_rate * db                # update bias
        
        return self

    def predict(self, X):
        X_poly = np.ones((X.shape[0], self.degree), dtype = float)

        for i in range(self.degree):
            X_poly[:, i] = X[:, 0] ** (i + 1)
        
        
        return X_poly.dot(self.weights) + self.bias


# Ridge Regression: _________________________________________________

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

# ARIMA Regression: _________________________________________________

class ARIMARegression(object):
    pass