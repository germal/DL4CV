import matplotlib
matplotlib.use("Agg")

from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report
from MyDLEnv.nn.conv import MiniVGGNet
from MyDLEnv.callbacks import TrainingMonitor
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.datasets import cifar10
import argparse
import os

#input:
#output: D:\Github\DL4CV\MiniVGGNet\output

ap = argparse.ArgumentParser()
ap.add_argument("-o", "--output", required=True,
                help="path to the output directory")
args = vars(ap.parse_args())

print("[INFO] process ID: {}".format(os.getpid()))

print("[INFO] loading CIFAR-10 data...")
((trainX, trainY), (testX, testY)) = cifar10.load_data()

#load and scale to [0, 1]
trainX = trainX.astype('float') / 255.0
testX = testX.astype('float') / 255.0

le = LabelBinarizer()
trainY = le.fit_transform(trainY)
testY = le.transform(testY)

# initialize the label names for the CIFAR-10 dataset
labelNames = ["airplane", "automobile", "bird", "cat", "deer",
              "dog", "frog", "horse", "ship", "truck"]

print("[INFO] compiling model...")
opt = SGD(lr=0.01, momentum=0.9, nesterov=True)
model = MiniVGGNet.build(width=32, height=32, depth=3, classes=10)
model.compile(loss='categorical_crossentropy', optimizer=opt,
              metrics=["accuracy"])

figPath = os.path.sep.join([args["output"], "{}.png".format(os.getpid())])
jsonPath = os.path.sep.join([args["output"], "{}.json".format(os.getpid())])
callbacks = [TrainingMonitor(figPath, jsonPath=jsonPath)]


print("[INFO] training network...")
H = model.fit(trainX, trainY, validation_data=(testX, testY),
              batch_size=64, epochs=100, callbacks=callbacks, verbose=1)

print("[INFO] evaluating network...")
predictions = model.predict(testX, batch_size=64)
print(classification_report(testY.argmax(axis=1),
                            predictions.argmax(axis=1),
                            target_names=labelNames))


