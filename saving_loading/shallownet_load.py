from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from MyDLEnv.preprocessing import ImageToArrayPreprocessor
from MyDLEnv.preprocessing import SimplePreprocessor
from MyDLEnv.datasets import SimpleDatasetLoader
from tensorflow.keras.models import load_model
from imutils import paths
import numpy as np
import argparse
import cv2

#input: D:\Docs\DigitalBooks\DeepLearningForCV\SB_Code\datasets\animals
#output: D:\Github\DL4CV\saving_loading

ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True,
                help="path to the input dataset")
ap.add_argument("-m", "--model", required=True,
                help="path to the pre-trained model")
args = vars(ap.parse_args())

# initialize class labels
classLabels = ["cat","dog","panda"]

print("[INFO] sampling images...")
imagePaths = np.array(list(paths.list_images(args["dataset"])))
idxs = np.random.randint(0, len(imagePaths), size=(10,))
imagePaths = imagePaths[idxs]

#initialize preprocessors
sp = SimplePreprocessor(32, 32)
iap = ImageToArrayPreprocessor()

#load and scale to [0, 1]
sdl = SimpleDatasetLoader(preprocessors=[sp, iap])
(data, labels) = sdl.load(imagePaths)
data = data.astype("float") / 255.0

# load pre-trained network
print("[INFO] loading pre-trained network...")
model = load_model(args["model"])

# make predictions on images
print("[INFO] predicting...")
preds = model.predict(data, batch_size=32).argmax(axis=1)

# loop over sample images
for (i, imagePath) in enumerate(imagePaths):
    image=cv2.imread(imagePath)
    cv2.putText(image, "Label:{}".format(classLabels[preds[i]]),
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
    cv2.imshow("Image", image)
    cv2.waitKey(0)
