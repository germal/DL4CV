from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
import numpy as np
import argparse

def sigmoid_activation(x):
    # compute the sigmoid activation
    return 1.0 / (1 + np.exp(-x))


def sigmoid_deriv(x):
    # compute the derivative assuming x has already been passed
    # through the sigmoid_activation
    # that is x = sigmoid_activation(x)
    return x * (1 - x)


def predict(X, W):
    preds = sigmoid_activation(X.dot(W))

    # apply step function to threshold the outputs to binary
    preds[preds <= 0.5] = 0
    preds[preds > 0] = 1

    return preds


def next_batch(X, y, batchSize):
    # loop over data X in mini-batches
    # tuple of current batched data and labels
    for i in np.arange(0, X.shape[0], batchSize):
        yield (X[i: i + batchSize], y[i: i + batchSize])


ap = argparse.ArgumentParser()
ap.add_argument("-e", "--epochs", type=float, default=100, help="# of epochs")
ap.add_argument("-a", "--alpha", type=float, default=0.01, help="learning rate")
ap.add_argument("-b", "--batchsize", type=int, default=32,
                help="size of SGD mini-batches")
args = vars(ap.parse_args())

# generate 2-class classification problem
(X, y) = make_blobs(n_samples=1000, n_features=2, centers=2, cluster_std=1.5, random_state=1)
y = y.reshape((y.shape[0], 1))
# insert a column of 1's as the last entry in the feature
# bias trick
X = np.c_[X, np.ones((X.shape[0]))]

# partition into train and test
(trainX, testX, trainY, testY) = train_test_split(X, y, test_size=0.5, random_state=42)

# Initialize weight matrix random
print("[INFO] training...")
W = np.random.randn(X.shape[1], 1)
losses = []

for epoch in np.arange(0, args["epochs"]):
    # total loss for each epoch (full dataset)
    epochLoss = []

    # loop over data in batches
    for (batchX, batchY) in next_batch(trainX, trainY, args["batchsize"]):
        # dot product of current batch of features and weight matrix
        # then pass through activation
        preds = sigmoid_activation(batchX.dot(W))

        # determine error, which is difference
        error = preds - batchY
        epochLoss.append(np.sum(error ** 2))

        # update using gradient descent
        d = error * sigmoid_deriv(preds)
        gradient = batchX.T.dot(d)

        # inside batch loop - multiple updates per epoch
        W += -args["alpha"] * gradient

    loss = np.average(epochLoss)
    losses.append(loss)

    if epoch == 0 or (epoch + 1) % 5 == 0:
        print(" epoch={}, loss={:.7f}".format(int(epoch + 1),
                                              loss))

# evaluate
print("[INFO] evaluating...")
preds = predict(testX, W)
print(classification_report(testY, preds))

# plot the (testing) classification data
plt.style.use("ggplot")
plt.figure()
plt.title("Data")
plt.scatter(testX[:, 0], testX[:, 1], marker="o", c=testY[:, 0], s=30)

plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, args["epochs"]), losses)
plt.title("Training Loss")
plt.xlabel("Epoch #")
plt.ylabel("Loss")
plt.show()