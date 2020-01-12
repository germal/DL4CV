from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.datasets import cifar10
from tensorflow.keras import backend as K
import matplotlib.pyplot as plt
import numpy as np
import argparse

ap = argparse.ArgumentParser()
ap.add_argument("-o", "--output", required=True, help="path to output loss/accuarcy plot")
args = vars(ap.parse_args())

#grab MNIST dataset (11MB)
print("[INFO] accessing CIFAR-10 data..")
((trainX, trainY),(testX,testY)) = cifar10.load_data()

# each image is 28x28x1
# need to flatten image to list of 28x28 (single vector)
trainX = trainX.reshape((trainX.shape[0], 3072)) #32 x 32 x 3
testX = testX.reshape((testX.shape[0], 3072))

# scale data to range [0,1]
trainX = trainX.astype("float32") / 255.0
testX = testX.astype("float32") / 255.0

# labels to vectors one-hot
lb = LabelBinarizer()
trainY = lb.fit_transform(trainY)
testY = lb.transform(testY)

# label names
labelNames = ["airplane", "automobile", "bird", "cat", "deer",
              "dog", "frog", "horse", "ship", "truck"]

#define 3072-1024-512-256-128-64-10 architecture
model = Sequential()
model.add(Dense(1024, input_shape=(3072,), activation='relu'))
model.add(Dense(512, activation='relu'))
model.add(Dense(256, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(10, activation='softmax'))

print("[INFO] training network...")
sgd = SGD(0.01)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
H = model.fit(trainX, trainY, validation_data = (testX, testY), epochs=100, batch_size=32)

print("[INFO] evaluating network...")
predictions = model.predict(testX, batch_size=32)
print(classification_report(testY.argmax(axis=1),
                            predictions.argmax(axis=1),
                            target_names=labelNames))

plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, 100), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, 100), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, 100), H.history["accuracy"], label="train_acc")
plt.plot(np.arange(0, 100), H.history["val_accuracy"], label="val_acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend()
plt.savefig(args["output"])
