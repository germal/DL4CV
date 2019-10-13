import numpy as np
import cv2

# initialize class labels and set seed
labels = ["dog", "cat", "panda"]
np.random.seed(42)

# randomly initialize weights matrix and bias
# would be learned but for now just use random values
W = np.random.randn(3, 3072) # mean 0 and unit variance
b = np.random.randn(3)

# load example image
orig = cv2.imread("beagle.png")
image = cv2.resize(orig, (32, 32)).flatten()

# compute output scores
scores = W.dot(image) + b

# loop over scores + labels and display
for (label, score) in zip(labels, scores):
    print("[INFO] {}: {:.2f}".format(label, score))

# draw label with highest score
cv2.putText(orig, "Label: {}".format(labels[np.argmax(scores)]),
            (10, 30), cv2.FONT_HERSHEY_COMPLEX, 0.9, (0, 255, 0), 2)
cv2.imshow("Image", orig)
cv2.waitKey(0)

# close window and exit with 0