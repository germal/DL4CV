from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report
from MyDLEnv.nn.conv import LeNet
from tensorflow.keras.datasets import mnist
from tensorflow.keras.optimizers import SGD
from tensorflow.keras import backend as K
import matplotlib.pyplot as plt
import numpy as np

#input:
#output:

print("[INFO] accessing mnist...")
((trainData, trainLabels), (testData, testLabels)) = mnist.load_data()

if K.image_data_format() == "channels_first":
    trainData = trainData.reshape((trainData.shape[0],1,28,28))
    testData = testData.reshape((testData.shape[0], 1, 28, 28))
else:
    trainData = trainData.reshape((trainData.shape[0],28,28,1))
    testData = testData.reshape((testData.shape[0], 28, 28, 1))

#load and scale to [0, 1]
trainData = trainData.astype('float32') / 255.0
testData = testData.astype('float32') / 255.0

le = LabelBinarizer()
trainLabels = le.fit_transform(trainLabels)
testLabels = le.transform(testLabels)

print("[INFO] compiling model...")
opt = SGD(lr=0.01)
model = LeNet.build(width=28, height=28, depth=1, classes=10)
model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=["accuracy"])

print("[INFO] training network...")
H = model.fit(trainData, trainLabels, validation_data=(testData, testLabels),
              batch_size=128, epochs=20, verbose=1)

print("[INFO] evaluating network...")
predictions = model.predict(testData, batch_size=128)
print(classification_report(testLabels.argmax(axis=1),
                            predictions.argmax(axis=1),
                            target_names=[str(x) for x in le.classes_]))

plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0,20), H.history['loss'], label="train_loss")
plt.plot(np.arange(0,20), H.history['val_loss'], label="val_loss")
plt.plot(np.arange(0,20), H.history["accuracy"], label="train_acc")
plt.plot(np.arange(0,20), H.history["val_accuracy"], label="val_acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend()
plt.show()
