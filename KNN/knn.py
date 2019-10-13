from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from MyDLEnv.preprocessing import SimplePreprocessor
from MyDLEnv.datasets import SimpleDatasetLoader
from imutils import paths
import argparse

ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True,
                help="path to input dataset")
ap.add_argument("-k", "--neighbours", type = int, default=1,
                help="# of nearest neighbours for classification")
ap.add_argument("-j", "--jobs", type = int, default=-1,
                help="# of jobs for k-NN distance calc (-1 uses all available cores)")
args = vars(ap.parse_args())

# grab list of images
print("[INFO] loading images...")
imagePaths = list(paths.list_images(args["dataset"]))

# initialize image preprocessor. Load data from disk and reshape
sp = SimplePreprocessor(32, 32)
sdl = SimpleDatasetLoader(preprocessors=[sp])
(data, labels) = sdl.load(imagePaths, verbose=500)
data = data.reshape((data.shape[0], -1)) #3072 = images of 32 x 32 x 3

# show information on memory consumption
print("[INFO] features matrix: {:.1f}MB".format(
    data.nbytes / (1024 * 1024.0)))

# Training and test split
le = LabelEncoder()
labels = le.fit_transform(labels)

# partition the data into traiing and testing splits using 75% for training
(trainX, testX, trainY, testY) = train_test_split(data, labels,
                                                  test_size=0.25, random_state=42)

# train and evaluate K-NN classifier on pixel intensities
print("[INFO] evaluating k-NN classifier...")
model = KNeighborsClassifier(n_neighbors=args['neighbours'],
                             n_jobs=args["jobs"])
model.fit(trainX, trainY)
print(classification_report(testY, model.predict(testX),
                            target_names=le.classes_))

#dataset see: D:\Docs\DigitalBooks\DeepLearningForCV\SB_Code\datasets\animals