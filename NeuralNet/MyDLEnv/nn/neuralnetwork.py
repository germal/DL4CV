import numpy as np

class NeuralNetwork:
    def __init__(self, layers, alpha=0.1):
        self.W = []
        self.layers = layers
        self.alpha = alpha

        # initialize weights for layers but not last 2
        for i in np.arange(0, len(layers) - 2):
            # add extra node for bias term
            # weight matrix of MxN size for each layer (M) and layer + 1 (N)
            w = np.random.randn(layers[i] + 1, layers[i + 1] + 1)
            self.W.append(w / np.sqrt(layers[i])) # normalized to sum to 1

        # last two layers, needs bias and doesn't need bias
        w = np.random.randn(layers[-2] + 1, layers[-1])
        self.W.append(w / np.sqrt(layers[-2]))

    def __repr__(self):
        # construct string to represent architecture
        return "NeuralNetwork: {}".format(
            "-".join(str(l) for l in self.layers))

    def sigmoid(self, x):
        return 1.0 / (1 + np.exp(-x))

    def sigmoid_deriv(self, x):
        # assumes x has already been passed through sigmoid
        return x * (1 - x)

    def fit(self, X, y, epochs=1000, displayUpdate=100):
        # insert bias column
        X = np.c_[X, np.ones((X.shape[0]))]

        # loop over epochs
        for epoch in np.arange(0, epochs):
            # loop over data
            for (x, target) in zip(X, y):
                self.fit_partial(x, target)
            # check to see if update is displayed
            if epoch == 0 or (epoch + 1) % displayUpdate == 0:
                loss = self.calculate_loss(X,y)
                print("[INFO] epoch={}, loss={:.7f}".format(
                    epoch + 1, loss
                ))

    def fit_partial(self, x, y):
        # construct list of output activations for each layer
        # first activation is special case - just the input feature vector
        A = [np.atleast_2d(x)] # initialize
        # FEEDFORWARD
        for layer in np.arange(0, len(self.W)):
            net = A[layer].dot(self.W[layer])
            out = self.sigmoid(net)
            A.append(out)

        # BACKPROPAGATION
        error = A[-1] - y # difference between prediction and target
        D = [error * self.sigmoid_deriv(A[-1])] #delta for final layer

        for layer in np.arange(len(A) - 2, 0, -1):
            # the delta for the current layer is equal to the delta
            # of the previous layer dotted with the weight matrix
            # of the current layer, followed by multiplying the delta
            # by the derivative of nonlinear activation function
            # for the activation of the current layer
            delta = D[-1].dot(self.W[layer].T)
            delta = delta * self.sigmoid_deriv(A[layer])
            D.append(delta)

        # Need to reverse deltas since they are in reverse order
        D = D[::-1]

        # WEIGHT UPDATE PHASE
        for layer in np.arange(0, len(self.W)):
            # update by taking dot product of layer
            # activations with their respective deltas, then
            # multiplying this value by small learning rate
            # adding to the weight matrix -- actual learning
            self.W[layer] += -self.alpha * A[layer].T.dot(D[layer])

    def predict(self, X, addBias=True):
        p = np.atleast_2d(X)

        if addBias:
            p = np.c_[p, np.ones((p.shape[0]))]

        for layer in np.arange(0, len(self.W)):
            p = self.sigmoid(np.dot(p, self.W[layer]))

        return p

    def calculate_loss(self, X, targets):
        targets = np.atleast_2d(targets)
        predictions = self.predict(X, addBias=False)
        loss = 0.5 * np.sum((predictions - targets) ** 2)

        return loss




