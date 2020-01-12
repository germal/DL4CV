from skimage.exposure import rescale_intensity
import numpy as np
import argparse
import cv2

def convolve(image, K):
    # grab spatial dimension of image
    (iH, iW) = image.shape[:2]
    (kH, kW) = K.shape[:2]

    # allocate memory for output image
    # need to pad image so spatial size is not reduced
    pad = (kW - 1)//2
    image = cv2.copyMakeBorder(image, pad, pad, pad, pad,
                               cv2.BORDER_REPLICATE)
    output = np.zeros((iH,iW), dtype="float")

    # loop over input image, 'sliding' kernel across
    for y in np.arange(pad, iH + pad):
        for x in np.arange(pad, iW + pad):
            #roi by center + pad
            roi = image[y - pad:y + pad + 1, x - pad:x + pad + 1]

            #convolution
            k = (roi * K).sum()

            #store convolved value at (x,y) coordinate
            #accounting for padding
            output[y - pad, x - pad] = k
            output = rescale_intensity(output, in_range=(0, 255))
            output = (output * 255).astype('uint8')
        print(y)

    return output


ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
                help="path to the input image")
args = vars(ap.parse_args())

smallBlur = np.ones((7,7), dtype='float') * (1.0 / (7 * 7))
largeBlur = np.ones((21,21), dtype='float') * (1.0 / (21 * 21))

sharpen = np.array((
    [0, -1, 0],
    [-1, 5, -1],
    [0, -1, 0]), dtype='int')

laplacian = np.array((
    [0, 1, 0],
    [1, 4, 1],
    [0, 1, 0]), dtype='int')

sobelX = np.array((
    [-1, 0, 1],
    [-2, 0, 2],
    [-1, 0, 1]), dtype='int')

sobelY = np.array((
    [-1, -2, -1],
    [0, 0, 0],
    [1, 2, 1]), dtype='int')

emboss = np.array((
    [-2, -1, 0],
    [-1, 1, 1],
    [0, 1, 2]), dtype='int')

kernelBank = (
    ("small_blur", smallBlur),
    ("large_blur", largeBlur),
    ("sharpen", sharpen),
    ("laplacian", laplacian),
    ("sobelX", sobelX),
    ("sobelY", sobelY),
    ("emboss", emboss))

image = cv2.imread(args["image"])
scale_percent = 50 # percent of original size
width = int(image.shape[1] * scale_percent / 100)
height = int(image.shape[0] * scale_percent / 100)
dim = (width, height)
# resize image
resized = cv2.resize(image, dim, interpolation = cv2.INTER_AREA)
gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)

for (kernelName, K) in kernelBank:
    print("[INFO] applying {} kernel...".format(kernelName))
    convolveOutput = convolve(gray, K)
    opencvOutput = cv2.filter2D(gray, -1, K)
    print("[INFO] finished applying convolutions...")

    cv2.imshow("original", gray)
    cv2.imshow("{} - convolve".format(kernelName), convolveOutput)
    cv2.imshow("{} - opencv".format(kernelName), opencvOutput)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

