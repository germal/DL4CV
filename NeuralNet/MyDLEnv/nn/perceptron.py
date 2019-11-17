import numpy as np

class Perceptron:
    def __init__(self, N, alpha=0.1):
        #intialize and store learning rate
        #N number of input values
        self.W = np.random.randn(N + 1) / np.sqrt(N)
        self.alpha = alpha

    def step(self, x):
        # apply step function activation
        return 1 if x > 0 else 0

    def fit(self, X, y, epochs=10):
        # insert column of 1's for bias trick
        X = np.c_[X, np.ones((X.shape[0]))]
        # loop over epochs
        for epoch in np.arange(0, epochs):
            for (x, target) in zip(X,y):
                # take dot prod. between input and weight matrix
                # then pass through step function for prediction
                p = self. step(np.dot(x, self.W))

                # perform weight update if incorrect
                if p != target:
                    error = p - target
                    self.W += -self.alpha * error * x

    def predict(self, X, addBias=True):
        X = np.atleast_2d(X) # ensure input is matrix

        # check to see about bias column
        if addBias:
            X = np.c_[X, np.ones((X.shape[0]))]

        return self.step(np.dot(X, self.W))
