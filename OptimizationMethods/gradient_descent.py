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


ap = argparse.ArgumentParser()
ap.add_argument("-e", "--epochs", type=float, default=100, help="# of epochs")
ap.add_argument("-a", "--alpha", type=float, default=0.01, help="learning rate")
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
    preds = sigmoid_activation(trainX.dot(W))

    error = preds - trainY
    loss = np.sum(error ** 2)
    losses.append(loss)

    # gradient descent update
    d = error * sigmoid_deriv(preds)
    gradient = trainX.T.dot(d)

    W += -args["alpha"] * gradient

    if epoch == 0 or (epoch + 1) % 5 == 0:
        print("[INFO] epochs={}, loss{:.7f}".format(int(epoch + 1),
                                                   loss))

# evaluate model
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
